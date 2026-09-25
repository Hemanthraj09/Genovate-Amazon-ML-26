"""Dev utility: score models trained on one variant against another variant's data.

Run with BER_VARIANT set to the *evaluation* variant (e.g. tl) and pass the
model folder to evaluate (e.g. model = trained on the original train mix):
    BER_VARIANT=tl python crosseval.py model
Fold models are applied out-of-fold (fold-k rows scored by the fold-k model,
which never trained on fold k -- folds are hashed by query cluster, identical
across variants), then stage-2 context is rebuilt and the decision rules are
scored with the exact macro F0.5 on the variant's evaluation universe.
"""
import sys
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import decide as D
import stage2 as S2
import train as TR
import tune


def oof_with(model_dir, prefix, extra_ctx=None):
    """Out-of-fold probabilities of `model_dir`'s fold models on this variant."""
    models = [lgb.Booster(model_file=str(C.work(model_dir, f"{prefix}{k}.txt"))) for k in range(TR.NFOLD)]
    feats = models[0].feature_name()
    outs = []
    for f in sorted(TR.feat_dir("train").glob("part_*.parquet")):
        part = TR.add_folds(pl.read_parquet(f).lazy()).collect()
        if extra_ctx is not None:
            part = part.join(extra_ctx, on=["qid", "s1"], how="inner")
        for k, m in enumerate(models):
            sub = part.filter(pl.col("fold") == k)
            p = m.predict(sub.select(feats).to_numpy(), num_threads=C.N_THREADS)
            outs.append(sub.select("qid", "s1", "y").with_columns(pl.Series("p", p.astype(np.float32))))
    return pl.concat(outs)


def main(model_dir):
    """Report stage-1 and stage-2 scores of model_dir on the active variant."""
    drop = tune.B.dropped_s1()
    s1 = (pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
            .filter(~pl.col("idx").is_in(drop)))
    truth = tune.B.true_pairs().filter(~pl.col("s1").is_in(drop))
    p1 = oof_with(model_dir, "lgb_fold")
    a1 = D.assign_argmax(p1)
    for tau in (0.6, 0.7, 0.8):
        tune.report(D.by_threshold(a1, tau), truth, s1, f"[{model_dir}] stage1 tau={tau}")
    ctx = S2.context(p1.select("qid", "s1", "p"))
    p2 = oof_with(model_dir, "s2_fold", extra_ctx=ctx)
    a2 = D.assign_argmax(p2)
    for tau in (0.6, 0.7, 0.8, 0.85):
        tune.report(D.by_threshold(a2, tau), truth, s1, f"[{model_dir}] stage2 tau={tau}")
    tune.report(D.by_expected_f(a2), truth, s1, f"[{model_dir}] stage2 expected-F")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "model")
