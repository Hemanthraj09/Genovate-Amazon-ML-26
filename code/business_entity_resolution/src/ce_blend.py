"""Tune how much to trust the cross-encoder (ce.py) against the tree probability.

For every band pair with an out-of-half cross-encoder logit, fit
    logit(p_new) = a * logit(p_tree) + b * ce + c
by logistic regression on one half of the S1 entities and measure macro F0.5 on
the other half (and vice versa), against the tree probability alone. Everything
here is out of sample: p_tree is honest OOF, ce comes from the model that never
saw that half, and the blend is scored on entities it was not fitted on.

    BER_VARIANT=fix BER_MODEL_TAG=v4bh python ce_blend.py
Writes WORK_DIR/ce/blend_<tag>.json with the coefficients refitted on all pairs.
"""
import json
import os
import numpy as np
import polars as pl
from sklearn.linear_model import LogisticRegression
import config as C
import decide as D
import tune


def logit(p):
    p = np.clip(p, 1e-6, 1 - 1e-6)
    return np.log(p / (1 - p))


def design(df):
    return np.column_stack([logit(df["p"].to_numpy()), df["ce"].to_numpy()])


def apply(df, coef):
    """Replace p by the blended probability where a cross-encoder score exists."""
    z = coef["a"] * logit(df["p"].to_numpy()) + coef["b"] * df["ce"].fill_null(0).to_numpy() + coef["c"]
    pb = 1 / (1 + np.exp(-z))
    return df.with_columns(pl.when(pl.col("ce").is_not_null()).then(pl.Series(pb)).otherwise(pl.col("p")).alias("p"))


def average(paths):
    """Mean cross-encoder logit over several runs (pairs present in all)."""
    fr = [pl.read_parquet(p).rename({"ce": f"ce{i}"}) for i, p in enumerate(paths)]
    j = fr[0]
    for f in fr[1:]:
        j = j.join(f, on=["qid", "s1"])
    return j.select("qid", "s1", pl.mean_horizontal([f"ce{i}" for i in range(len(fr))]).alias("ce"))


def main():
    oof, s1, truth = tune.load_oof("oof2")
    # BER_CE_RUNS="" or ",b": average the out-of-half logits of several ce.py runs
    runs = os.environ.get("BER_CE_RUNS", "").split(",")
    outname = C.MODEL_TAG + ("_" + "".join(r or "a" for r in runs) if len(runs) > 1 else "")
    ce = average([C.work("ce", f"oof_ce_{C.MODEL_TAG}{r}.parquet") for r in runs])
    dec = json.load(open(C.work(C.MODEL_DIR, "decision_oof2.json")))
    miss = dec["miss"] if dec.get("use_miss") else None
    a = D.assign_argmax(oof).join(s1.rename({"idx": "s1"}), on="s1", how="left")
    x = oof.join(ce, on=["qid", "s1"], how="left").with_columns(tune.entity_half().alias("half"))
    band = x.filter(pl.col("ce").is_not_null())
    print(f"pairs {x.height:,}, with cross-encoder score {band.height:,}")
    tot = {"tree": 0.0, "blend": 0.0}; n = 0
    for h in (0, 1):
        fit = band.filter(pl.col("half") != h)
        lr = LogisticRegression(C=1.0, max_iter=200).fit(design(fit), fit["y"].to_numpy())
        coef = {"a": float(lr.coef_[0][0]), "b": float(lr.coef_[0][1]), "c": float(lr.intercept_[0])}
        ev = x.filter(pl.col("half") == h)
        s1h = s1.filter(tune.entity_half("idx") == h); trh = truth.filter(tune.entity_half("s1") == h)
        for name, frame in (("tree", ev), ("blend", apply(ev, coef))):
            aa = D.assign_argmax(frame.select("qid", "s1", "p", "y")).join(s1h.rename({"idx": "s1"}), on="s1", how="left")
            sc = tune.report(D.by_expected_f(aa, miss_odds=miss), trh, s1h, "", quiet=True)
            tot[name] += sc * s1h.height
        n += s1h.height
        print(f"half {h}: coef {coef}", flush=True)
    for k in tot:
        print(f"{k:6s} held-out F0.5 {tot[k] / n:.5f}")
    lr = LogisticRegression(C=1.0, max_iter=200).fit(design(band), band["y"].to_numpy())
    coef = {"a": float(lr.coef_[0][0]), "b": float(lr.coef_[0][1]), "c": float(lr.intercept_[0]),
            "gain": (tot["blend"] - tot["tree"]) / n}
    json.dump(coef, open(C.work("ce", f"blend_{outname}.json"), "w"), indent=1)
    if len(runs) > 1:     # the matching test file for ensemble.py BER_CE=<outname>
        average([C.work("ce", f"test_ce_{C.MODEL_TAG}{r}.parquet") for r in runs]).write_parquet(C.work("ce", f"test_ce_{outname}.parquet"))
    print("saved", coef)


if __name__ == "__main__":
    main()
