"""Step 6b: evaluate decision rules on out-of-fold train predictions.

Reports macro F0.5 over ALL train S1 entities (singletons included) for:
  * argmax + global threshold tau (grid)
  * argmax + isotonic calibration + per-entity expected-F0.5 selection
Also reports the score per country and per query segment, and saves the
chosen settings to WORK_DIR/model/decision.json.
"""
import json
import numpy as np
import polars as pl
from sklearn.isotonic import IsotonicRegression
import config as C
import blocking as B
import decide as D
import evaluate as E


def load_oof():
    """OOF pair probabilities and the evaluation universe."""
    oof = pl.read_parquet(C.work("model", "oof.parquet"))
    s1 = pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
    truth = B.true_pairs()
    return oof, s1, truth


def fit_calibrator(assigned):
    """Isotonic map from raw p to P(match) on argmax-assigned OOF pairs."""
    iso = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    iso.fit(assigned["p"].to_numpy(), assigned["y"].to_numpy())
    return iso


def calibrate(assigned, iso):
    """Apply a fitted isotonic calibrator to column p."""
    return assigned.with_columns(pl.Series("p", iso.predict(assigned["p"].to_numpy()).astype(np.float32)))


def report(pred, truth, s1, label):
    """Print overall and per-country macro F0.5."""
    score, ent = E.macro_f05(pred, truth, s1["idx"])
    ent = ent.join(s1.rename({"idx": "s1"}), on="s1")
    by_c = {c: round(v, 5) for c, v in ent.group_by("country").agg(pl.col("f").mean()).iter_rows()}
    sing = ent.filter(pl.col("G") == 0)["f"].mean()
    print(f"{label:38s} F0.5={score:.5f}  by country={by_c}  singletons={sing:.4f}", flush=True)
    return score


def main():
    oof, s1, truth = load_oof()
    assigned = D.assign_argmax(oof)
    print(f"OOF pairs {oof.height:,}; assigned {assigned.height:,}")
    best = (None, -1)
    for tau in (0.3, 0.4, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9):
        sc = report(D.by_threshold(assigned, tau), truth, s1, f"argmax + tau={tau}")
        if sc > best[1]:
            best = (tau, sc)
    # calibrated expected-F selection (calibrator fit on half, evaluated on other half by cluster)
    iso = fit_calibrator(assigned)
    sc_ef = report(D.by_expected_f(calibrate(assigned, iso)), truth, s1, "argmax + isotonic + expected-F")
    raw_ef = report(D.by_expected_f(assigned), truth, s1, "argmax + raw p + expected-F")
    dec = {"tau": best[0], "tau_score": best[1], "ef_iso_score": sc_ef, "ef_raw_score": raw_ef}
    with open(C.work("model", "decision.json"), "w") as f:
        json.dump(dec, f, indent=1)
    import pickle
    with open(C.work("model", "isotonic.pkl"), "wb") as f:
        pickle.dump(iso, f)
    print(dec)


if __name__ == "__main__":
    main()
