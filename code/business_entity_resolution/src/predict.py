"""Step 8: score test candidates, decide matches, write the submission files.

candidate_pairs.tsv = every (S1, S2/S3) pair the model scores (the pruned
blocking output, i.e. exactly the rows of the test feature files).
matching_results.tsv = the decided subset.
"""
import json
import os
import pickle
import polars as pl
import config as C
import decide as D
import output as O
import train as TR


def main(stage=1, rule=None):
    """Predict on test and write output/ files using the tuned decision rule.

    stage=1: stage-1 fold-average probabilities; stage=2: stage-2 re-scoring
    (requires stage2.fit_predict_oof() to have been run).
    """
    if os.environ.get("BER_S1_FROM_FILE"):
        # stacked model sets (stack.py) have no stage-1 models of their own
        pairs = pl.read_parquet(C.work(C.MODEL_DIR, "test_pred.parquet"))
    else:
        models = TR.load_models()
        pairs = TR.predict_split(models, "test")
        pairs.write_parquet(C.work(C.MODEL_DIR, "test_pred.parquet"))
    name = "oof"
    if stage >= 2:
        import stage2
        for level in range(2, stage + 1):
            pairs = stage2.predict_test(level)
        name = f"oof{stage}"
    with open(C.work(C.MODEL_DIR, f"decision_{name}.json")) as f:
        dec = json.load(f)
    rule = rule or dec["rule"]
    assigned = D.assign_argmax(pairs)
    if rule.startswith("ef"):
        # the country label only routes calibration and is never a model feature;
        # a country with no calibrator of its own (France) falls back to pooled
        s1c = (pl.read_parquet(C.work("raw", "test_s1.parquet"))
                 .select(pl.col("idx").alias("s1"), "country"))
        assigned = assigned.join(s1c, on="s1", how="left")
    if rule.startswith("ef_iso"):
        with open(C.work(C.MODEL_DIR, f"isotonic_{name}.pkl"), "rb") as f:
            blob = pickle.load(f)
        import tune
        assigned = tune.calibrate(assigned, blob["isos"], dec.get("per_country", False),
                                  blob["prior"] if dec.get("use_prior") else None)
    # test-decoy-density correction (see decide.shrink_odds); 1.0 = off
    assigned = D.shrink_odds(assigned, D.parse_odds(os.environ.get("BER_ODDS")))
    if rule.startswith("ef"):
        matches = D.by_expected_f(assigned, miss_odds=dec["miss"] if dec.get("use_miss") else None)
    else:
        matches = D.by_threshold(assigned, dec["tau"])
    print(f"stage={stage} rule={rule}: {matches.height:,} matched pairs over "
          f"{matches['s1'].n_unique():,} S1 entities")
    return O.write(matches, pairs.select("s1", "qid"), split="test")


if __name__ == "__main__":
    import sys
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1, sys.argv[2] if len(sys.argv) > 2 else None)
