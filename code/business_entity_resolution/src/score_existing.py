"""Dev utility: score an already-trained model set on the corrected train world.

v04, v05 and r2 were trained and tuned on the legacy "tl" variant, whose candidate
lists are ~2.3 shorter per query than test's (see feedback6.md). Their local scores
are therefore not comparable with a model built on the corrected variant. This
script replays each old pipeline -- its fold models, its stage-2 context module and
the decision rule it actually ships -- over the corrected world, so every candidate
is measured on the same test-shaped data.

    BER_VARIANT=fix python score_existing.py model_tl_r2
    BER_VARIANT=fix python score_existing.py model_tl stage2_v04

Caveats, both of which flatter the old models slightly:
  * stage-1/2 probabilities use the single non-owning fold model, which is how
    their OOF was produced; on test they average both, which is a little sharper.
  * v04/v05 were blocked with absolute key caps, while the corrected world uses
    r2's size-scaled caps, so only r2 is a like-for-like blocking comparison.
"""
import json
import pickle
import sys
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import blocking as B
import decide as D
import train as TR
import tune


def _fold_predict(chunks, models, feats, fold_col="fold"):
    """OOF prediction: each row scored by the one model that did not see its fold."""
    out = []
    for d in chunks:
        X = TR.design(d, feats)
        fold = d[fold_col].to_numpy()
        P = np.stack([m.predict(X, num_threads=C.N_THREADS) for m in models])
        p = (P.sum(0) - P[fold, np.arange(len(fold))]) / (len(models) - 1)
        keep = ["qid", "s1"] + (["y"] if "y" in d.columns else [])
        out.append(d.select(keep).with_columns(pl.Series("p", p.astype(np.float32))))
        del X, P
    return pl.concat(out)


def _chunks(ctx=None):
    """Stream the corrected world's feature chunks, labelled and folded."""
    for f in sorted(TR.feat_dir("train").glob("part_*.parquet")):
        d = TR.add_folds(pl.read_parquet(f).lazy()).collect()
        yield d.join(ctx, on=["qid", "s1"], how="inner") if ctx is not None else d


def main(mdir, s2mod="stage2"):
    if s2mod != "stage2":
        sys.path.insert(0, str(C.WORK_DIR / "dev"))
    S2 = __import__(s2mod)
    m1 = [lgb.Booster(model_file=str(C.work(mdir, f"lgb_fold{k}.txt"))) for k in range(TR.NFOLD)]
    f1 = m1[0].feature_name()
    print(f"[{mdir}] stage-1: {len(m1)} folds, {len(f1)} features", flush=True)
    p1 = _fold_predict(_chunks(), m1, f1)

    ctx = S2.context(p1.select("qid", "s1", "p"), "train")
    m2 = [lgb.Booster(model_file=str(C.work(mdir, f"s2_fold{k}.txt"))) for k in range(TR.NFOLD)]
    f2 = m2[0].feature_name()
    print(f"[{mdir}] stage-2: {len(f2)} features", flush=True)
    pairs = _fold_predict(_chunks(ctx), m2, f2)

    drop = B.dropped_s1()
    s1 = (pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
            .filter(~pl.col("idx").is_in(drop.implode())))
    truth = B.true_pairs().filter(~pl.col("s1").is_in(drop.implode()))
    dec = json.load(open(C.work(mdir, "decision_oof2.json")))
    a = D.assign_argmax(pairs).join(s1.rename({"idx": "s1"}), on="s1", how="left")
    rule = dec["rule"]
    if rule.startswith("ef_iso"):
        blob = pickle.load(open(C.work(mdir, "isotonic_oof2.pkl"), "rb"))
        isos = blob["isos"] if isinstance(blob, dict) else {tune.POOLED: blob}
        a = tune.calibrate(a, isos, dec.get("per_country", False), None)
    pred = D.by_expected_f(a) if rule.startswith("ef") else D.by_threshold(a, dec["tau"])
    tune.report(pred, truth, s1, f"[{mdir}] rule={rule} on CORRECTED world")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "stage2")
