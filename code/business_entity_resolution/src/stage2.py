"""Stage 2: competition-aware re-scoring from stage-1 probabilities.

Stage-1 scores each (record, S1) pair in isolation. Stage 2 adds context
computed from stage-1 probabilities of *neighbouring* pairs:
  record side : p share among the record's candidates, margin to the best
                competing S1, second-best p, is-best flag, candidate count
  entity side : sum / count of confident records, best competing record,
                rank of this record within the entity, number of records for
                which this entity is the best choice
Train context uses OUT-OF-FOLD stage-1 probabilities only (train.py), so the
stage-2 model never learns from over-confident in-sample scores.

Outputs: model/s2_fold{k}.txt, model/oof2.parquet, model/test_pred2.parquet
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import train as TR

S2_EXTRA = ["p", "q_pmax", "q_p2", "q_psum", "q_share", "q_margin", "q_isbest", "q_n",
            "s_psum", "s_nconf", "s_pmax_other", "s_prank", "s_nbest", "s_psum_best"]


def context(pairs):
    """Add stage-2 context columns to a (qid, s1, p) frame."""
    q = pairs.group_by("qid").agg(
        pl.col("p").max().alias("q_pmax"),
        pl.col("p").sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias("q_p2"),
        pl.col("p").sum().alias("q_psum"), pl.len().alias("q_n"))
    x = pairs.join(q, on="qid").with_columns(
        (pl.col("p") / (pl.col("q_psum") + 1e-6)).alias("q_share"),
        (pl.col("p") >= pl.col("q_pmax")).cast(pl.Float32).alias("q_isbest"))
    x = x.with_columns(
        (pl.col("p") - pl.when(pl.col("q_isbest") == 1).then(pl.col("q_p2")).otherwise(pl.col("q_pmax"))).alias("q_margin"))
    s = x.group_by("s1").agg(
        pl.col("p").sum().alias("s_psum"), (pl.col("p") > 0.5).sum().alias("s_nconf"),
        pl.col("q_isbest").sum().alias("s_nbest"),
        (pl.col("p") * pl.col("q_isbest")).sum().alias("s_psum_best"),
        pl.col("p").max().alias("s_pmax"),
        pl.col("p").sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias("s_p2"))
    x = x.join(s, on="s1").with_columns(
        pl.when(pl.col("p") >= pl.col("s_pmax")).then(pl.col("s_p2")).otherwise(pl.col("s_pmax")).alias("s_pmax_other"),
        pl.col("p").rank("ordinal", descending=True).over("s1").alias("s_prank"))
    return x.select("qid", "s1", *[pl.col(c).cast(pl.Float32) for c in S2_EXTRA])


def _design(split, ctx, feats):
    """Stage-1 features joined with stage-2 context, chunk by chunk."""
    for f in sorted(TR.feat_dir(split).glob("part_*.parquet")):
        part = pl.read_parquet(f)
        yield part.join(ctx, on=["qid", "s1"], how="inner")


def fit_predict_oof(frac=TR.TRAIN_FRAC):
    """Train stage-2 fold models on OOF context; return OOF stage-2 probs."""
    t0 = time.time()
    oof = pl.read_parquet(C.work(C.MODEL_DIR, "oof.parquet"))
    ctx = context(oof.select("qid", "s1", "p"))
    feats = TR.feature_names() + S2_EXTRA
    lab = TR.add_folds(oof.select("qid", "s1").lazy()).select("qid", "s1", "y", "fold", "u").collect()
    ctx = ctx.join(lab, on=["qid", "s1"])
    models, outs = [], []
    del oof
    for k in range(TR.NFOLD):
        # stream the chunk files so the full design matrix never sits in memory
        tr = pl.concat([d.filter((pl.col("fold") != k) & (pl.col("u") < frac))
                        for d in _design("train", ctx, feats)])
        va = pl.concat([d.filter((pl.col("fold") == k) & (pl.col("u") < 0.05))
                        for d in _design("train", ctx, feats)])
        print(f"stage2 fold {k}: train {tr.height:,} ({time.time() - t0:.0f}s)", flush=True)
        dtr = lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy(), feature_name=feats)
        dva = lgb.Dataset(va.select(feats).to_numpy(), va["y"].to_numpy(), reference=dtr)
        del tr
        m = lgb.train(TR.PARAMS, dtr, TR.ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=False)])
        m.save_model(str(C.work(C.MODEL_DIR, f"s2_fold{k}.txt")))
        del dtr, dva
        for d in _design("train", ctx, feats):
            te = d.filter(pl.col("fold") == k)
            p = m.predict(te.select(feats).to_numpy(), num_threads=C.N_THREADS)
            outs.append(te.select("qid", "s1", "y").with_columns(pl.Series("p", p.astype(np.float32))))
        models.append(m)
        print(f"stage2 fold {k}: best iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    oof2 = pl.concat(outs)
    oof2.write_parquet(C.work(C.MODEL_DIR, "oof2.parquet"))
    return models, oof2


def predict_test():
    """Stage-2 probabilities on test from stage-1 fold-average predictions."""
    models = [lgb.Booster(model_file=str(C.work(C.MODEL_DIR, f"s2_fold{k}.txt"))) for k in range(TR.NFOLD)]
    p1 = pl.read_parquet(C.work(C.MODEL_DIR, "test_pred.parquet"))
    ctx = context(p1)
    feats = TR.feature_names("test") + S2_EXTRA
    outs = []
    for part in _design("test", ctx, feats):
        X = part.select(feats).to_numpy()
        p = np.mean([m.predict(X, num_threads=C.N_THREADS) for m in models], axis=0)
        outs.append(part.select("qid", "s1").with_columns(pl.Series("p", p.astype(np.float32))))
    out = pl.concat(outs)
    out.write_parquet(C.work(C.MODEL_DIR, "test_pred2.parquet"))
    return out


if __name__ == "__main__":
    fit_predict_oof()
