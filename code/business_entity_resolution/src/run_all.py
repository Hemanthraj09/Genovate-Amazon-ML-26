"""End-to-end pipeline: raw TSVs -> output/matching_results.tsv + candidate_pairs.tsv.

    python run_all.py            # all steps
    python run_all.py blocking   # resume from a step (prepare, maps, normalize,
                                 # blocking, features, train, tune, predict)
Each step writes its artefacts under WORK_DIR, so steps can be re-run alone.
"""
import sys
import time

STEPS = ["prepare", "maps", "normalize", "blocking", "features", "train", "tune", "predict"]


def run(step):
    """Run one pipeline step by name."""
    if step == "prepare":
        import prepare; prepare.main()
    elif step == "maps":
        import learn_maps; learn_maps.main()
    elif step == "normalize":
        import normalize_all; normalize_all.main()
    elif step == "blocking":
        import blocking, config as C
        for split in ("train", "test"):
            cand = blocking.run(split)
            cand.write_parquet(C.work("cand", f"{split}.parquet"))
            if split == "train":
                blocking.oracle_report(cand, label="train")
    elif step == "features":
        import features
        for split in ("train", "test"):
            features.build(split)
    elif step == "train":
        import train
        train.predict_oof(train.fit_folds())
    elif step == "tune":
        import tune; tune.main()
    elif step == "predict":
        import predict; predict.main()


if __name__ == "__main__":
    start = STEPS.index(sys.argv[1]) if len(sys.argv) > 1 else 0
    t0 = time.time()
    for s in STEPS[start:]:
        print(f"===== {s} =====", flush=True)
        run(s)
        print(f"===== {s} done ({time.time() - t0:.0f}s) =====", flush=True)
