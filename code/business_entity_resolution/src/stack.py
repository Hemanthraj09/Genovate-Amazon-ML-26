"""Seed a model set whose stage-1 probabilities are the average of several builds.

Only for builds that share one training world (same blocking, same feature files,
same fold hash), e.g. v4 and v4b: their OOF rows line up one to one and every
row's probability is still out-of-fold, so the average is a legitimate OOF and
stage 2 can be trained and tuned on it as usual.

    BER_VARIANT=fix BER_MODEL_TAG=v4s python stack.py model_fix_v4 model_fix_v4b
    BER_VARIANT=fix BER_MODEL_TAG=v4s python stage2.py 2
    BER_VARIANT=fix BER_MODEL_TAG=v4s python tune.py oof2
    BER_VARIANT=fix BER_MODEL_TAG=v4s BER_S1_FROM_FILE=1 python predict.py 2
"""
import sys
import polars as pl
import config as C


def average(tags, name, keys):
    frames = [pl.read_parquet(C.work(t, f"{name}.parquet")) for t in tags]
    j = frames[0].select(*keys, pl.col("p").alias("p0"))
    for i, f in enumerate(frames[1:], 1):
        j = j.join(f.select("qid", "s1", pl.col("p").alias(f"p{i}")), on=["qid", "s1"], how="inner")
    heights = [f.height for f in frames]
    assert j.height == max(heights) == min(heights), f"{name}: rows differ {heights} -> {j.height}"
    out = j.select(*keys, pl.mean_horizontal([f"p{i}" for i in range(len(tags))]).cast(pl.Float32).alias("p"))
    out.write_parquet(C.work(C.MODEL_DIR, f"{name}.parquet"))
    print(f"{name}: {out.height:,} rows averaged over {len(tags)} builds -> {C.MODEL_DIR}")


if __name__ == "__main__":
    tags = sys.argv[1:]
    average(tags, "oof", ["qid", "s1", "y"])
    average(tags, "test_pred", ["qid", "s1"])
