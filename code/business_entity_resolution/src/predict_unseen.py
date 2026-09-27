"""Rebuild the submission with a separate decision rule for unseen countries.

France is 15% of test and appears in no training data. Leave-one-country-out
(loco.py) shows what that costs: scoring India with a model that never saw India
drops it from 0.985 to ~0.92, and its singleton score from 0.98 to ~0.70 -- the
model is over-confident on an unseen country and false-merges, which F0.5 punishes
twice as hard as a miss.

So a country with no training data should not share US/India's decision rule. Two
ways to set its rule, both calibrated on the LOCO predictions (a genuinely unseen
country) rather than on US/India:

  tau     : the plain threshold that maximises the LOCO score
  iso     : an isotonic map fitted on LOCO, then the ordinary expected-F selector

Reruns only the decision step from the cached test stage-2 probabilities, so it
costs minutes and never touches the models.

    BER_VARIANT=fix python predict_unseen.py tau 0.9 output_fr
"""
import json
import pickle
import sys
import polars as pl
import config as C
import decide as D
import output as O
import tune


def unseen_calibrator(loco_files):
    """Isotonic p -> P(match) fitted on argmax-assigned LOCO pairs."""
    parts = [pl.read_parquet(f) for f in loco_files]
    a = D.assign_argmax(pl.concat(parts).select("qid", "s1", "y", "p"))
    return tune._iso(a["p"].to_numpy(), a["y"].to_numpy())


def main(mode="tau", value=0.9, out="output_unseen", unseen=("France",)):
    dec = json.load(open(C.work(C.MODEL_DIR, "decision_oof2.json")))
    blob = pickle.load(open(C.work(C.MODEL_DIR, "isotonic_oof2.pkl"), "rb"))
    pairs = pl.read_parquet(C.work(C.MODEL_DIR, "test_pred2.parquet"))
    s1c = (pl.read_parquet(C.work("raw", "test_s1.parquet"))
             .select(pl.col("idx").alias("s1"), "country"))
    a = D.assign_argmax(pairs).join(s1c, on="s1", how="left")
    seen = a.filter(~pl.col("country").is_in(list(unseen)))
    uns = a.filter(pl.col("country").is_in(list(unseen)))
    print(f"seen {seen.height:,} assigned records; unseen {uns.height:,}")

    # seen countries: exactly the rule tune.py chose
    s = tune.calibrate(seen, blob["isos"], dec.get("per_country", False),
                       blob["prior"] if dec.get("use_prior") else None)
    m_seen = D.by_expected_f(s, miss_odds=dec["miss"] if dec.get("use_miss") else None)

    # unseen countries: rule calibrated on LOCO
    if mode == "tau":
        m_uns = D.by_threshold(uns, float(value))
    else:
        iso = unseen_calibrator(list(C.WORK_DIR.glob(f"{C.MODEL_DIR}/loco_*_s2.parquet")))
        m_uns = D.by_expected_f(tune.calibrate(uns, {tune.POOLED: iso}, False, None))
    print(f"matches: seen {m_seen.height:,}  unseen {m_uns.height:,} "
          f"(was {D.by_expected_f(tune.calibrate(uns, blob['isos'], dec.get('per_country', False), None), miss_odds=dec['miss'] if dec.get('use_miss') else None).height:,})")

    matches = pl.concat([m_seen, m_uns])
    return O.write(matches, pairs.select("s1", "qid"), split="test",
                   out_dir=C.REPO_DIR / out)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "tau",
         sys.argv[2] if len(sys.argv) > 2 else 0.9,
         sys.argv[3] if len(sys.argv) > 3 else "output_unseen")
