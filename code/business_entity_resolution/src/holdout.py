"""Dev utility: honest-holdout check of the stacked pipeline.

Half of the train S1 entities (H) and every record whose cluster belongs to H
are excluded from all model training. The pipeline is then applied to H
exactly as it is applied to test:
  * stage-1: 2 fold models trained on the non-H part (T) -> OOF p1 on T,
    fold-average p1 on everything else (as on test);
  * stage-2: context from those p1, 2 fold models trained on T,
    fold-average p2 elsewhere; decision rule tuned on T only.
Scores on H are then compared with the ordinary out-of-fold scores on the same
H entities. A large difference means the OOF validation is optimistic.
(Remaining known leak, not removed here: learned maps / dictionary were fitted
on all train pairs.)

    BER_VARIANT=tl python holdout.py
"""
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import blocking as B
import decide as D
import stage2 as S2
import train as TR
import tune

OUT = "model_holdout"


def tag_rows(df):
    """Add fold, y, and the T (train-allowed) flag to a feature frame."""
    x = TR.add_folds(df.lazy()).collect()
    s_h = (pl.col("s1").cast(pl.UInt64).hash(seed=5) % 2) == 0
    c_h = (pl.col("cluster").hash(seed=5) % 2) == 0
    return x.with_columns((~s_h & ~c_h).alias("T"), s_h.alias("sH"))


def run_level(feats_fn, ctx=None, prefix="h1"):
    """Train 2 fold models on T rows; return p for all rows (OOF on T, avg elsewhere)."""
    t0 = time.time()
    files = sorted(TR.feat_dir("train").glob("part_*.parquet"))
    def rows():
        for f in files:
            d = tag_rows(pl.read_parquet(f))
            if ctx is not None:
                d = d.join(ctx, on=["qid", "s1"], how="inner")
            yield d
    feats = feats_fn()
    models = []
    for k in range(TR.NFOLD):
        tr = pl.concat([d.filter(pl.col("T") & (pl.col("fold") != k) & (pl.col("u") < TR.TRAIN_FRAC)) for d in rows()])
        va = pl.concat([d.filter(pl.col("T") & (pl.col("fold") == k) & (pl.col("u") < 0.05)) for d in rows()])
        dtr = lgb.Dataset(tr.select(feats).to_numpy(), tr["y"].to_numpy(), feature_name=feats)
        dva = lgb.Dataset(va.select(feats).to_numpy(), va["y"].to_numpy(), reference=dtr)
        del tr
        m = lgb.train(TR.PARAMS, dtr, TR.ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        m.save_model(str(C.work(OUT, f"{prefix}_fold{k}.txt")))
        models.append(m)
        print(f"  {prefix} fold {k}: best iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    outs = []
    for d in rows():
        X = d.select(feats).to_numpy()
        ps = [m.predict(X, num_threads=C.N_THREADS) for m in models]
        avg = np.mean(ps, axis=0)
        # model k trained on folds != k, so it is out-of-fold for fold k rows.
        # T rows get OOF (as in training); all other rows the fold average (as on test)
        oof = np.where(d["fold"].to_numpy() == 0, ps[0], ps[1])
        p = np.where(d["T"].to_numpy(), oof, avg)
        outs.append(d.select("qid", "s1", "y", "T", "sH").with_columns(pl.Series("p", p.astype(np.float32))))
    return pl.concat(outs)


def score_H(pairs, T_pairs_for_tuning, label):
    """Tune rules on T entities, report macro F0.5 on H entities."""
    drop = B.dropped_s1()
    s1 = (pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
            .filter(~pl.col("idx").is_in(drop)))
    s1 = s1.with_columns(((pl.col("idx").cast(pl.UInt64).hash(seed=5) % 2) == 0).alias("sH"))
    truth = B.true_pairs().filter(~pl.col("s1").is_in(drop))
    sT, sH = s1.filter(~pl.col("sH")), s1.filter(pl.col("sH"))
    assigned = D.assign_argmax(pairs.select("qid", "s1", "y", "p"))
    # tune tau on T entities
    best = max(((tau, tune.E.macro_f05(D.by_threshold(assigned, tau).filter(pl.col("s1").is_in(sT["idx"])),
                                       truth.filter(pl.col("s1").is_in(sT["idx"])), sT["idx"])[0])
                for tau in (0.5, 0.6, 0.65, 0.7, 0.75, 0.8)), key=lambda t: t[1])
    iso = tune.fit_calibrator(assigned.filter(pl.col("s1").is_in(sT["idx"])))
    tH = truth.filter(pl.col("s1").is_in(sH["idx"]))
    predH_thr = D.by_threshold(assigned, best[0]).filter(pl.col("s1").is_in(sH["idx"]))
    predH_ef = D.by_expected_f(tune.calibrate(assigned, iso)).filter(pl.col("s1").is_in(sH["idx"]))
    tune.report(predH_thr, tH, sH.drop("sH"), f"{label} H tau={best[0]}")
    tune.report(predH_ef, tH, sH.drop("sH"), f"{label} H ef_iso")


def main():
    t0 = time.time()
    print("== honest stage 1", flush=True)
    p1 = run_level(TR.feature_names, prefix="h1")
    p1.write_parquet(C.work(OUT, "p1.parquet"))
    score_H(p1, None, "[honest stage1]")
    print(f"== honest stage 2 ({time.time() - t0:.0f}s)", flush=True)
    ctx = S2.context(p1.select("qid", "s1", "p"), "train")
    p2 = run_level(lambda: TR.feature_names() + S2.S2_EXTRA, ctx=ctx, prefix="h2")
    p2.write_parquet(C.work(OUT, "p2.parquet"))
    score_H(p2, None, "[honest stage2]")
    print(f"== ordinary OOF on the same H entities ({time.time() - t0:.0f}s)", flush=True)
    for name, lab in (("oof", "[OOF stage1]"), ("oof2", "[OOF stage2]")):
        o = pl.read_parquet(C.work(C.MODEL_DIR, f"{name}.parquet"))
        score_H(o, None, lab)


if __name__ == "__main__":
    main()
