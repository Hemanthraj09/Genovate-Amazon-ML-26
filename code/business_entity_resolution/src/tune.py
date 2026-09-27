"""Step 6b: evaluate decision rules on out-of-fold train predictions.

Reports macro F0.5 over ALL train S1 entities (singletons included) for:
  * argmax + global threshold tau (grid)
  * argmax + isotonic calibration + per-entity expected-F0.5 selection

Calibration honesty: the isotonic map is fitted on one entity half and the score
is reported on the other, so the reported number is out-of-sample. (It used to be
fitted and scored on the same rows, which made every expected-F number optimistic.)
The calibrator that ships for inference is then refitted on everything.

Calibration is available pooled or per country. Per country matters because the
score distributions differ (France's mean p is roughly half the US's), but a
per-country map also bakes in that country's *training* decoy rate -- and India's
corrected world sits at 32% decoys against test's estimated 40.5%, because train
has too few India records to reach it. `prior_shift` corrects for that with an
odds multiplier. All variants are scored and the best on held-out entities wins.

Saves the chosen settings to WORK_DIR/<model>/decision_<name>.json.
"""
import json
import pickle
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
import config as C
import blocking as B
import decide as D
import evaluate as E

POOLED = "_pooled"


def load_oof(name="oof"):
    """OOF pair probabilities and the evaluation universe."""
    oof = pl.read_parquet(C.work(C.MODEL_DIR, f"{name}.parquet"))
    drop = B.dropped_s1()
    s1 = (pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
            .filter(~pl.col("idx").is_in(drop.implode())))
    truth = B.true_pairs().filter(~pl.col("s1").is_in(drop.implode()))
    return oof, s1, truth


def entity_half(col="s1"):
    """Stable half split of S1 entities, for out-of-sample calibration."""
    return (pl.col(col).cast(pl.UInt64).hash(seed=31) % 2).cast(pl.UInt8)


def _iso(p, y):
    """Fit one isotonic regression."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(p, y)
    return iso


def fit_calibrator(assigned):
    """Pooled-only isotonic map (for callers with no country column)."""
    return {POOLED: _iso(assigned["p"].to_numpy(), assigned["y"].to_numpy())}


def fit_calibrators(assigned):
    """Pooled and per-country isotonic maps from raw p to P(match)."""
    isos = fit_calibrator(assigned)
    for (c,), g in assigned.group_by("country"):
        isos[c] = _iso(g["p"].to_numpy(), g["y"].to_numpy())
    return isos


def train_decoy_share(assigned):
    """Share of this world's query records that match nothing, per country.

    Measured on the argmax-assigned rows, which hold exactly one row per query
    record that reached the candidate set; y=1 means that record's true entity is
    the one it was assigned to. Records whose true entity lost the argmax are
    counted as decoys here, so this slightly overstates the decoy share -- it is
    used only to size a correction, never as a label.
    """
    g = assigned.group_by("country").agg(pl.col("y").mean().alias("m"))
    return {r["country"]: 1.0 - r["m"] for r in g.iter_rows(named=True)}


def miss_odds(assigned, truth, s1):
    """(1-rho)/rho per country, rho = share of true pairs surviving blocking+argmax.

    Feeds decide.by_expected_f so the selector knows some of an entity's true
    matches are not on its candidate list at all.
    """
    den = dict(truth.join(s1.rename({"idx": "s1"}), on="s1", how="left")
                    .group_by("country").len().iter_rows())
    num = dict(assigned.filter(pl.col("y") == 1).group_by("country").len().iter_rows())
    return {c: (1 - num.get(c, 0) / n) / max(num.get(c, 0) / n, 1e-9)
            for c, n in den.items() if n}


def prior_odds(assigned):
    """Odds multiplier per country taking this world's decoy rate to test's."""
    got = train_decoy_share(assigned)
    out = {}
    for c, d_tr in got.items():
        d_te = C.DECOY_SHARE.get(c)
        if d_te is None or not 0 < d_tr < 1:
            continue
        out[c] = ((1 - d_te) / d_te) * (d_tr / (1 - d_tr))
    return out


def calibrate(assigned, isos, per_country=False, prior=None):
    """Map p through the calibrator; optionally per country and prior-corrected.

    A country with no calibrator of its own (France at inference) falls back to
    the pooled map and gets no prior correction. `isos` may also be a bare
    estimator, which is how the pre-feedback6 model folders saved it.
    """
    if not isinstance(isos, dict):
        isos = {POOLED: isos}
    p = assigned["p"].to_numpy()
    out = np.empty(len(p), dtype=np.float64)
    if per_country and "country" in assigned.columns:
        cc = assigned["country"].to_numpy()
        for c in np.unique(cc):
            m = cc == c
            out[m] = isos.get(c, isos[POOLED]).predict(p[m])
            if prior and c in prior:
                r = prior[c]
                q = out[m]
                out[m] = (q * r) / (q * r + (1 - q) + 1e-12)
    else:
        out = isos[POOLED].predict(p)
    return assigned.with_columns(pl.Series("p", out.astype(np.float32)))


def report(pred, truth, s1, label, quiet=False):
    """Print overall and per-country macro F0.5."""
    score, ent = E.macro_f05(pred, truth, s1["idx"])
    ent = ent.join(s1.rename({"idx": "s1"}), on="s1")
    by_c = {c: round(v, 5) for c, v in ent.group_by("country").agg(pl.col("f").mean()).iter_rows()}
    sing = ent.filter(pl.col("G") == 0)["f"].mean()
    if not quiet:
        print(f"{label:44s} F0.5={score:.5f}  by country={by_c}  singletons={sing:.4f}", flush=True)
    return score


def main(name="oof"):
    """Tune the decision rule on OOF predictions `model/{name}.parquet`."""
    oof, s1, truth = load_oof(name)
    assigned = (D.assign_argmax(oof)
                .join(s1.rename({"idx": "s1"}), on="s1", how="left")
                .with_columns(entity_half().alias("half")))
    print(f"OOF pairs {oof.height:,}; assigned {assigned.height:,}")
    print("world decoy share by country:",
          {c: round(v, 4) for c, v in train_decoy_share(assigned).items()},
          "| test target:", C.DECOY_SHARE)
    prior = prior_odds(assigned)
    print("prior odds multiplier:", {c: round(v, 4) for c, v in prior.items()})

    # --- threshold grid (no calibration involved, so no split needed)
    best = (None, -1.0)
    for tau in (0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85):
        sc = report(D.by_threshold(assigned, tau), truth, s1, f"argmax + tau={tau}")
        if sc > best[1]:
            best = (tau, sc)

    # --- expected-F variants, each scored on the half its calibrator never saw
    miss = miss_odds(assigned, truth, s1)
    print("missing-match odds by country:", {c: round(v, 4) for c, v in miss.items()})
    halves = {h: (assigned.filter(pl.col("half") == h),
                  s1.filter(entity_half("idx") == h),
                  truth.filter(entity_half("s1") == h)) for h in (0, 1)}

    def evaluate(cal, use_prior, use_miss):
        """Mean held-out macro F0.5 over both halves for one decision config."""
        tot = wtot = 0.0
        for h in (0, 1):
            a_ev, s1_ev, tr_ev = halves[h]
            a = a_ev if cal is None else calibrate(
                a_ev, fit_calibrators(halves[1 - h][0]), cal == "pc",
                prior if use_prior else None)
            pred = D.by_expected_f(a, miss_odds=miss if use_miss else None)
            tot += report(pred, tr_ev, s1_ev, "", quiet=True) * s1_ev.height
            wtot += s1_ev.height
        return tot / wtot

    configs = {"ef_raw": (None, False), "ef_iso": ("pooled", False),
               "ef_iso_pc": ("pc", False), "ef_iso_pc_prior": ("pc", True)}
    scores = {}
    for vname, (cal, up) in configs.items():
        scores[vname] = evaluate(cal, up, False)
        print(f"{'argmax + ' + vname:44s} F0.5={scores[vname]:.5f}  (held-out halves)", flush=True)
    # the missing-match correction is orthogonal: test it on the best calibration only
    bestcal = max(configs, key=lambda v: scores[v])
    cal, up = configs[bestcal]
    scores[bestcal + "_miss"] = evaluate(cal, up, True)
    print(f"{'argmax + ' + bestcal + '_miss':44s} F0.5={scores[bestcal + '_miss']:.5f}  "
          f"(held-out halves)", flush=True)

    dec = {"tau": best[0], "threshold": best[1], **scores}
    dec["rule"] = max(["threshold", *scores], key=lambda r: dec[r])
    dec["per_country"] = "_pc" in dec["rule"]
    dec["use_prior"] = "prior" in dec["rule"]
    dec["use_miss"] = dec["rule"].endswith("_miss")
    dec["miss"] = miss
    # the shipped calibrator is refitted on everything (more data, same mapping)
    with open(C.work(C.MODEL_DIR, f"decision_{name}.json"), "w") as f:
        json.dump(dec, f, indent=1)
    with open(C.work(C.MODEL_DIR, f"isotonic_{name}.pkl"), "wb") as f:
        pickle.dump({"isos": fit_calibrators(assigned), "prior": prior}, f)
    print(dec)
    return dec


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "oof")
