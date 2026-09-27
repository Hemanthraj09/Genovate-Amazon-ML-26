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
import os
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import blocking as B

NFOLD = int(os.environ.get("BER_NFOLD", "4"))
TRAIN_FRAC = float(os.environ.get("BER_TRAIN_FRAC", "0.6"))   # share of eligible clusters fitted
VAL_FRAC = 0.04            # early-stopping split, carved from the TRAINING folds only
ID_COLS = ("qid", "s1")
PARAMS = dict(objective="binary",
              learning_rate=float(os.environ.get("BER_LR", "0.08")),
              num_leaves=int(os.environ.get("BER_LEAVES", "255")),
              min_data_in_leaf=int(os.environ.get("BER_MIN_LEAF", "200")),
              feature_fraction=float(os.environ.get("BER_FEAT_FRAC", "0.8")),
              bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
              max_bin=255, num_threads=C.N_THREADS, verbose=-1,
              seed=int(os.environ.get("BER_SEED", str(C.SEED))),
              extra_trees=os.environ.get("BER_EXTRA_TREES", "0") == "1")
ROUNDS = int(os.environ.get("BER_ROUNDS", "1800"))   # early stopping usually halts first
# BER_LEGACY_OOF=1 reproduces the OOF averaging used for the v4/v4b model sets
# (models j != k, which leaks -- see predict_oof). Only for regenerating those.
LEGACY_OOF = os.environ.get("BER_LEGACY_OOF", "0") == "1"


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
    """Model feature columns (everything except ids, and blocking artefacts if ROBUST)."""
    drop = set(ID_COLS) | (set(C.BLOCKING_ARTEFACTS) if C.ROBUST else set())
    return [c for c in feat_scan(split).collect_schema().names() if c not in drop]


def design(df, feats):
    """Feature matrix as float32 (polars upcasts mixed dtypes to float64 otherwise)."""
    return df.select([pl.col(c).cast(pl.Float32) for c in feats]).to_numpy()


def fit_folds():
    """Train one model per fold on a cluster-sample of the *other* folds.

    The early-stopping set is carved out of the training folds (u >= 1 - VAL_FRAC),
    never out of fold k. Validating on fold k would choose best_iteration using the
    very rows that fold's model is about to score out-of-fold, which made the local
    number optimistic.
    """
    t0 = time.time()
    feats = feature_names()
    lf = add_folds(feat_scan("train"))
    C.work(C.MODEL_DIR, "x")
    models = []
    for k in range(NFOLD):
        other = pl.col("fold") != k
        tr = lf.filter(other & (pl.col("u") < TRAIN_FRAC)).collect()
        va = lf.filter(other & (pl.col("u") >= 1 - VAL_FRAC)).collect()
        print(f"fold {k}: train {tr.height:,} (pos {tr['y'].sum():,})  valid {va.height:,}  "
              f"({time.time() - t0:.0f}s)", flush=True)
        # drop each polars frame before LightGBM bins its matrix, otherwise the
        # polars copy, the float32 copy and the binned copy are all live at once --
        # which is what put a 5-fold/80% run into swap on a 16 GB box
        ytr, yva = tr["y"].to_numpy(), va["y"].to_numpy()
        Xtr = design(tr, feats); del tr
        Xva = design(va, feats); del va
        dtr = lgb.Dataset(Xtr, ytr, feature_name=feats, free_raw_data=True)
        dva = lgb.Dataset(Xva, yva, reference=dtr)
        del Xtr, Xva
        m = lgb.train(PARAMS, dtr, ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=False)])
        m.save_model(str(C.work(C.MODEL_DIR, f"lgb_fold{k}.txt")))
        print(f"fold {k}: best iter {m.best_iteration}  ({time.time() - t0:.0f}s)", flush=True)
        del dtr, dva
        models.append(m)
    return models


def load_models():
    """Load the saved fold models."""
    return [lgb.Booster(model_file=str(C.work(C.MODEL_DIR, f"lgb_fold{k}.txt"))) for k in range(NFOLD)]


def predict_oof(models):
    """Out-of-fold probability for every train candidate pair.

    A fold-k pair is scored by model k, the ONLY model that never trained on fold k
    (model j trains on every fold except j). Test pairs get the average of all NFOLD
    models, which is slightly sharper than one model; that is the standard, safe
    direction of mismatch.

    Between 26 Sep and 27 Sep 12:45 this averaged the models j != k -- exactly the
    ones that HAD trained on fold k. That leaked the training rows into the OOF,
    inflated every local score (v4b 0.99044 leaked vs ~0.988 honest), and trained
    stage 2 on over-confident stage-1 scores.
    """
    feats = feature_names()
    lf = add_folds(feat_scan("train")).select("qid", "s1", "y", "fold", *feats)
    out = []
    for k in range(NFOLD):
        part = lf.filter(pl.col("fold") == k).collect()
        X = design(part, feats)
        if LEGACY_OOF:   # reproduces the v4/v4b uploads only; leaks (see docstring)
            p = np.mean([m.predict(X, num_threads=C.N_THREADS) for j, m in enumerate(models) if j != k], axis=0)
        else:
            p = models[k].predict(X, num_threads=C.N_THREADS)
        out.append(part.select("qid", "s1", "y").with_columns(pl.Series("p", p.astype(np.float32))))
        del part, X
    oof = pl.concat(out)
    oof.write_parquet(C.work(C.MODEL_DIR, "oof.parquet"))
    return oof


def predict_split(models, split):
    """Average of all fold models' probabilities for every pair of a split."""
    feats = feature_names(split)
    files = sorted(feat_dir(split).glob("part_*.parquet"))
    out = []
    for f in files:
        part = pl.read_parquet(f)
        X = design(part, feats)
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
