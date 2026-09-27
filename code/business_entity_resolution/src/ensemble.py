"""Average the stage-2 test probabilities of several builds, then decide.

v2 and v3 were blocked identically on the test side, so they score the exact same
candidate pairs and their probabilities can be averaged pair-for-pair. They differ
on the train side (v3 rebuilt the decoy pool from natural decoys rather than
orphans), which is what makes them worth averaging: two models that disagree
slightly for a structural reason, not just a seed.

Their OOF sets live in different training worlds, so the ensemble cannot be tuned
locally end to end. It is scored on the pairs the two worlds share, and otherwise
inherits the decision rule of the build named first.

    BER_VARIANT=fix python ensemble.py output_ens model_fix_v3 model_fix_v2
"""
import json
import os
import pickle
import sys
import polars as pl
import config as C
import decide as D
import evaluate as E
import output as O
import blocking as B
import tune


def load(tag, name):
    return pl.read_parquet(C.work(tag, f"{name}.parquet"))


def local_check(tags):
    """Score each build, and their average, on the OOF pairs the worlds share."""
    frames = [load(t, "oof2").select("qid", "s1", "y", pl.col("p").alias(t)) for t in tags]
    j = frames[0]
    for f in frames[1:]:
        j = j.join(f.drop("y"), on=["qid", "s1"], how="inner")
    print(f"shared OOF pairs: {j.height:,}")
    drop = B.dropped_s1()
    s1 = (pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
            .filter(~pl.col("idx").is_in(drop.implode())))
    truth = B.true_pairs().filter(~pl.col("s1").is_in(drop.implode()))
    s1 = s1.filter(pl.col("idx").is_in(j["s1"].unique().implode()))
    truth = truth.filter(pl.col("s1").is_in(s1["idx"].implode()))
    j = j.with_columns(pl.mean_horizontal(tags).alias("ens"))
    for col in tags + ["ens"]:
        a = D.assign_argmax(j.select("qid", "s1", "y", pl.col(col).alias("p")))
        tune.report(D.by_expected_f(a), truth, s1, f"  {col} (shared OOF universe)")


def main(out, tags):
    local_check(tags)
    parts = [load(t, "test_pred2").select("qid", "s1", pl.col("p").alias(t)) for t in tags]
    j = parts[0]
    for p in parts[1:]:
        j = j.join(p, on=["qid", "s1"], how="full", coalesce=True)
    # union, not intersection: blocking breaks ties at the top-K boundary
    # nondeterministically, so ~12% of pairs exist in only one build's candidate
    # set. Intersecting would throw that recall away; a pair only one build scored
    # keeps that build's probability.
    print(f"test pairs: union {j.height:,}, in all builds "
          f"{j.drop_nulls(tags).height:,}, per build {[p.height for p in parts]}")
    pairs = j.select("qid", "s1", pl.mean_horizontal(pl.col(t) for t in tags).alias("p"))
    pairs = pairs.filter(pl.col("p").is_not_null())

    dec = json.load(open(C.work(tags[0], "decision_oof2.json")))
    blob = pickle.load(open(C.work(tags[0], "isotonic_oof2.pkl"), "rb"))
    s1c = (pl.read_parquet(C.work("raw", "test_s1.parquet"))
             .select(pl.col("idx").alias("s1"), "country"))
    a = D.assign_argmax(pairs).join(s1c, on="s1", how="left")
    if dec["rule"].startswith("ef_iso"):
        a = tune.calibrate(a, blob["isos"], dec.get("per_country", False),
                           blob["prior"] if dec.get("use_prior") else None)
    odds = D.parse_odds(os.environ.get("BER_ODDS"))    # see decide.shrink_odds; 1.0 = off
    a = D.shrink_odds(a, odds)
    print(f"odds multiplier {odds}")
    matches = (D.by_expected_f(a, miss_odds=dec["miss"] if dec.get("use_miss") else None)
               if dec["rule"].startswith("ef") else D.by_threshold(a, dec["tau"]))
    # Countries with no training labels (open set: whatever test has that train
    # lacks) can be decided from STAGE-1 probabilities with a plain threshold.
    # Train-on-one-country experiments (loco.py) showed stage 2's context features
    # can hurt a country the model never saw, and preferred a strict tau.
    unseen_tau = os.environ.get("BER_UNSEEN_S1_TAU")
    if unseen_tau:
        seen = pl.read_parquet(C.work("raw", "train_s1.parquet"), columns=["country"])["country"].unique()
        p1 = [load(t, "test_pred").select("qid", "s1", pl.col("p").alias(t)) for t in tags]
        j1 = p1[0]
        for p in p1[1:]:
            j1 = j1.join(p, on=["qid", "s1"], how="full", coalesce=True)
        s1p = j1.select("qid", "s1", pl.mean_horizontal(pl.col(t) for t in tags).alias("p"))
        a1 = D.assign_argmax(s1p).join(s1c, on="s1", how="left")
        unseen = ~pl.col("country").is_in(seen.implode())
        new = D.by_threshold(a1.filter(unseen), float(unseen_tau))
        keep = matches.join(s1c, on="s1", how="left").filter(~unseen).select("s1", "qid")
        print(f"unseen countries from stage 1, tau={unseen_tau}: {new.height:,} pairs "
              f"(stage 2 had {matches.height - keep.height:,})")
        matches = pl.concat([keep, new])
        pairs = pl.concat([pairs.select("qid", "s1"), s1p.select("qid", "s1")]).unique()
    print(f"rule={dec['rule']}: {matches.height:,} matched pairs over {matches['s1'].n_unique():,} entities")
    return O.write(matches, pairs.select("s1", "qid"), split="test", out_dir=C.REPO_DIR / out)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2:])
