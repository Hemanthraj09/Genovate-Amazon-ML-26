"""Dev utility: leave-one-country-out, the only local proxy we have for France.

France is 15% of test and appears in no training data, so nothing in our ordinary
validation says how a model behaves on a country it has never seen. This trains
stage 1 on ONE country's clusters and scores the OTHER, which isolates the
model-transfer half of that question (it does not remove the held-out country's
learned maps, so it is an optimistic bound -- France gets no maps at all).

    BER_VARIANT=fix python loco.py US India      # train on US, score India

Compare the printed score with that country's ordinary score from tune.py.
"""
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import blocking as B
import decide as D
import train as TR
import tune


def country_map():
    """s1 -> country for the train split."""
    return pl.read_parquet(C.work("raw", "train_s1.parquet")).select(
        pl.col("idx").alias("s1"), "country")


def main(train_c, score_c, frac=0.6):
    t0 = time.time()
    cm = country_map()
    feats = TR.feature_names()
    lf = TR.add_folds(TR.feat_scan("train")).join(cm.lazy(), on="s1", how="left")

    tr = lf.filter((pl.col("country") == train_c) & (pl.col("u") < frac)).collect()
    va = lf.filter((pl.col("country") == train_c) & (pl.col("u") >= 1 - TR.VAL_FRAC)).collect()
    print(f"[loco] train on {train_c}: {tr.height:,} pairs (pos {tr['y'].sum():,}), "
          f"valid {va.height:,} ({time.time() - t0:.0f}s)", flush=True)
    dtr = lgb.Dataset(TR.design(tr, feats), tr["y"].to_numpy(), feature_name=feats)
    dva = lgb.Dataset(TR.design(va, feats), va["y"].to_numpy(), reference=dtr)
    del tr, va
    m = lgb.train(TR.PARAMS, dtr, TR.ROUNDS, valid_sets=[dva],
                  callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=False)])
    print(f"[loco] best iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    del dtr, dva

    te = lf.filter(pl.col("country") == score_c).collect()
    p = m.predict(TR.design(te, feats), num_threads=C.N_THREADS)
    pairs = te.select("qid", "s1", "y").with_columns(pl.Series("p", p.astype(np.float32)))
    pairs.write_parquet(C.work(C.MODEL_DIR, f"loco_{train_c}_{score_c}_s1.parquet"))
    del te

    # stage 2, same held-out discipline: context from the stage-1 scores of BOTH
    # countries (as on test, where context is built over the whole split), but the
    # stage-2 model only ever sees train_c rows.
    import stage2 as S2
    p_all = lf.select("qid", "s1", "y", "u", "country").collect()
    p_tr = m.predict(TR.design(lf.filter(pl.col("country") == train_c).collect(), feats),
                     num_threads=C.N_THREADS)
    allp = pl.concat([
        p_all.filter(pl.col("country") == train_c).select("qid", "s1").with_columns(pl.Series("p", p_tr.astype(np.float32))),
        pairs.select("qid", "s1", "p")])
    ctx = S2.context(allp, "train")
    feats2 = feats + S2.S2_EXTRA
    lab = p_all.select("qid", "s1", "y", "u", "country")
    del p_all, allp

    def chunks():
        """Stream the feature files joined to context and labels (never all at once)."""
        for f in sorted(TR.feat_dir("train").glob("part_*.parquet")):
            yield (pl.read_parquet(f).join(ctx, on=["qid", "s1"], how="inner")
                     .join(lab, on=["qid", "s1"], how="inner"))

    tr_p, va_p = [], []
    for d in chunks():
        keep = d.filter(pl.col("country") == train_c)
        tr_p.append(keep.filter(pl.col("u") < frac))
        va_p.append(keep.filter(pl.col("u") >= 1 - TR.VAL_FRAC))
    tr2, va2 = pl.concat(tr_p), pl.concat(va_p)
    del tr_p, va_p
    print(f"[loco] stage2 train {tr2.height:,} ({time.time() - t0:.0f}s)", flush=True)
    dtr = lgb.Dataset(TR.design(tr2, feats2), tr2["y"].to_numpy(), feature_name=feats2)
    dva = lgb.Dataset(TR.design(va2, feats2), va2["y"].to_numpy(), reference=dtr)
    del tr2, va2
    m2 = lgb.train(TR.PARAMS, dtr, TR.ROUNDS, valid_sets=[dva],
                   callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=False)])
    del dtr, dva
    outs = []
    for d in chunks():
        te2 = d.filter(pl.col("country") == score_c)
        if not te2.height:
            continue
        q = m2.predict(TR.design(te2, feats2), num_threads=C.N_THREADS)
        outs.append(te2.select("qid", "s1", "y").with_columns(pl.Series("p", q.astype(np.float32))))
    pairs2 = pl.concat(outs)
    pairs2.write_parquet(C.work(C.MODEL_DIR, f"loco_{train_c}_{score_c}_s2.parquet"))
    print(f"[loco] stage2 best iter {m2.best_iteration}, saved predictions ({time.time() - t0:.0f}s)", flush=True)

    drop = B.dropped_s1()
    s1 = (pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
            .filter(~pl.col("idx").is_in(drop.implode()) & (pl.col("country") == score_c)))
    truth = B.true_pairs().filter(pl.col("s1").is_in(s1["idx"].implode()))
    print(f"[loco] scoring {score_c}: {s1.height:,} entities, {truth.height:,} true pairs", flush=True)
    for stage, pr in (("stage1", pairs), ("stage2", pairs2)):
        a = D.assign_argmax(pr).join(s1.rename({"idx": "s1"}), on="s1", how="left")
        for tau in (0.6, 0.7, 0.8, 0.85, 0.9, 0.95, 0.97):
            tune.report(D.by_threshold(a, tau), truth, s1, f"[loco {train_c}->{score_c}] {stage} tau={tau}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
