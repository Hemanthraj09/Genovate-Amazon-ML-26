"""Step 5: train the pair classifier (LightGBM) with out-of-fold predictions.

Folds are grouped by *query cluster*: a query that truly matches S1 entity s
belongs to cluster s; an unmatched (decoy) query is its own cluster. All
candidates of a query, and all queries of the same entity, therefore land in
the same fold, so every train pair gets an out-of-fold probability from a
model that never saw that entity. These OOF probabilities are used to tune
the decision rule (and later for stage-2 features).

Outputs (WORK_DIR/model/):
    lgb_fold{k}.txt     one model per fold (averaged for test inference)
    oof.parquet         qid, s1, y, p   for every train candidate pair
"""
import json
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import blocking as B

NFOLD = 2
TRAIN_FRAC = 0.6           # share of each training fold's clusters used for fitting
ID_COLS = ("qid", "s1")
PARAMS = dict(objective="binary", learning_rate=0.08, num_leaves=255, min_data_in_leaf=200,
              feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=C.N_THREADS, verbose=-1, seed=C.SEED)
ROUNDS = 800


def feat_dir(split):
    """Feature folder of a split (train uses the active variant's folder)."""
    return C.work("feat", C.TRAIN_TAG if split == "train" else split, "x").parent


def feat_scan(split):
    """Lazy scan over the feature chunk files of a split."""
    return pl.scan_parquet(str(feat_dir(split) / "part_*.parquet"))


def cluster_table():
    """qid -> cluster id (true S1 for matched queries) and label lookup."""
    tp = B.true_pairs()
    return tp.select("qid", pl.col("s1").alias("cluster"))


def add_folds(lf):
    """Attach label y and fold id (hash of the query cluster) to a lazy frame."""
    tp = B.true_pairs().with_columns(pl.lit(1, pl.UInt8).alias("y")).lazy()
    cl = cluster_table().lazy()
    lf = (lf.join(tp, on=["qid", "s1"], how="left")
            .join(cl, on="qid", how="left")
            .with_columns(pl.col("y").fill_null(0),
                          pl.coalesce(pl.col("cluster").cast(pl.UInt64),
                                      pl.col("qid").cast(pl.UInt64) + (1 << 32)).alias("cluster")))
    return lf.with_columns((pl.col("cluster").hash(seed=11) % NFOLD).cast(pl.UInt8).alias("fold"),
                           ((pl.col("cluster").hash(seed=23) % 1000) / 1000.0).alias("u"))


def feature_names(split="train"):
    """Model feature columns (everything except ids)."""
    return [c for c in feat_scan(split).collect_schema().names() if c not in ID_COLS]


def fit_folds():
    """Train one model per fold on a cluster-sample of the other folds."""
    t0 = time.time()
    feats = feature_names()
    lf = add_folds(feat_scan("train"))
    C.work(C.MODEL_DIR, "x")
    models = []
    for k in range(NFOLD):
        tr = lf.filter((pl.col("fold") != k) & (pl.col("u") < TRAIN_FRAC)).collect()
        va = lf.filter((pl.col("fold") == k) & (pl.col("u") < 0.05)).collect()
        print(f"fold {k}: train {tr.height:,} (pos {tr['y'].sum():,})  valid {va.height:,}  ({time.time() - t0:.0f}s)", flush=True)
        dtr = lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy(), feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(va.select(feats).to_numpy(), va["y"].to_numpy(), reference=dtr)
        del tr
        m = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.log_evaluation(100), lgb.early_stopping(50, verbose=False)])
        m.save_model(str(C.work(C.MODEL_DIR, f"lgb_fold{k}.txt")))
        print(f"fold {k}: best iter {m.best_iteration}  ({time.time() - t0:.0f}s)", flush=True)
        models.append(m)
    return models


def load_models():
    """Load the saved fold models."""
    return [lgb.Booster(model_file=str(C.work(C.MODEL_DIR, f"lgb_fold{k}.txt"))) for k in range(NFOLD)]


def predict_oof(models):
    """Out-of-fold probability for every train candidate pair."""
    feats = feature_names()
    lf = add_folds(feat_scan("train")).select("qid", "s1", "y", "fold", *feats)
    out = []
    for k, m in enumerate(models):
        part = lf.filter(pl.col("fold") == k).collect()
        p = m.predict(part.select(feats).to_numpy(), num_threads=C.N_THREADS)
        out.append(part.select("qid", "s1", "y").with_columns(pl.Series("p", p.astype(np.float32))))
        del part
    oof = pl.concat(out)
    oof.write_parquet(C.work(C.MODEL_DIR, "oof.parquet"))
    return oof


def predict_split(models, split):
    """Average of fold models' probabilities for every pair of a split."""
    feats = feature_names(split)
    files = sorted(feat_dir(split).glob("part_*.parquet"))
    out = []
    for f in files:
        part = pl.read_parquet(f)
        X = part.select(feats).to_numpy()
        p = np.mean([m.predict(X, num_threads=C.N_THREADS) for m in models], axis=0)
        out.append(part.select("qid", "s1").with_columns(pl.Series("p", p.astype(np.float32))))
    return pl.concat(out)


if __name__ == "__main__":
    ms = fit_folds()
    oof = predict_oof(ms)
    imp = sorted(zip(ms[0].feature_name(), ms[0].feature_importance("gain")), key=lambda x: -x[1])
    print("top features:", [(n, int(g)) for n, g in imp[:25]])
    with open(C.work(C.MODEL_DIR, "feature_importance.json"), "w") as f:
        json.dump([(n, float(g)) for n, g in imp], f)
