"""Step 8: score test candidates, decide matches, write the submission files.

candidate_pairs.tsv = every (S1, S2/S3) pair the model scores (the pruned
blocking output, i.e. exactly the rows of the test feature files).
matching_results.tsv = the decided subset.
"""
import json
import pickle
import polars as pl
import config as C
import decide as D
import output as O
import train as TR


def main(rule=None):
    """Predict on test and write output/ files using the tuned decision rule."""
    models = TR.load_models()
    pairs = TR.predict_split(models, "test")
    pairs.write_parquet(C.work("model", "test_pred.parquet"))
    with open(C.work("model", "decision.json")) as f:
        dec = json.load(f)
    rule = rule or dec.get("rule", "threshold")
    assigned = D.assign_argmax(pairs)
    if rule == "ef_iso":
        with open(C.work("model", "isotonic.pkl"), "rb") as f:
            iso = pickle.load(f)
        import tune
        matches = D.by_expected_f(tune.calibrate(assigned, iso))
    elif rule == "ef_raw":
        matches = D.by_expected_f(assigned)
    else:
        matches = D.by_threshold(assigned, dec["tau"])
    print(f"rule={rule}: {matches.height:,} matched pairs over {matches['s1'].n_unique():,} S1 entities")
    return O.write(matches, pairs.select("s1", "qid"), split="test")


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else None)
