"""Stage 2: competition-aware re-scoring from stage-1 probabilities.

Stage-1 scores each (record, S1) pair in isolation. Stage 2 adds context
computed from stage-1 probabilities of *neighbouring* pairs:
  record side : p share among the record's candidates, margin to the best
                competing S1, second-best p, is-best flag, candidate count
  entity side : sum / count of confident records, best competing record,
                rank of this record within the entity, number of records for
                which this entity is the best choice
  siblings    : how many *other* candidates of the same entity share this
                record's house number / name / (number, name) / address, and
                how many carry the entity's own number / name (plain counts and
                stage-1-probability-weighted). A deviation shared by siblings is
                usually a systematic source format (true), while a name
                deviation shared by siblings signals a look-alike decoy entity.
Train context uses OUT-OF-FOLD stage-1 probabilities only (train.py), so the
stage-2 model never learns from over-confident in-sample scores.

The same code stacks further: level L builds its context from level L-1
probabilities (level 1 = train.py). Outputs per level L >= 2:
model/s{L}_fold{k}.txt, model/oof{L}.parquet, model/test_pred{L}.parquet
"""
import os
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import train as TR

S2_EXTRA = ["p", "q_pmax", "q_p2", "q_psum", "q_share", "q_margin", "q_isbest",
            "s_psum", "s_nconf", "s_pmax_other", "s_prank", "s_nbest", "s_psum_best",
            "sib_hn_same", "sib_hn_same_p", "sib_nm_same", "sib_nm_same_p",
            "sib_both_same", "sib_both_same_p", "sib_ad_same", "sib_ad_same_p",
            "sib_hn_eq_s", "sib_hn_eq_s_p", "sib_nm_eq_s", "sib_nm_eq_s_p",
            "adc_tset", "adc_ratio", "adc_partial"]
COMMON_FRAC = 0.02    # address tokens in more than this share of a country's records are "common"


def _core_strings(s1n, qn, col):
    """`col` with the country's common tokens removed (digit tokens always kept).

    Common = present in > COMMON_FRAC of the split's records of that country
    (S1 + S2 + S3). This strips street types, cities, regions/departments and
    generator-added generic words without any country-specific list.
    """
    both = pl.concat([s1n.select(pl.lit(1).alias("side"), pl.col("idx").alias("id"), "country", col),
                      qn.select(pl.lit(2).alias("side"), pl.col("qid").alias("id"), "country", col)])
    n = both.group_by("country").agg(pl.len().alias("n"))
    tok = both.select("side", "id", "country", pl.col(col).str.split(" ").alias("t")).explode("t").filter(pl.col("t") != "")
    df = (tok.unique(["side", "id", "t"]).group_by("country", "t").agg(pl.len().alias("df"))
             .join(n, on="country").filter((pl.col("df") > COMMON_FRAC * pl.col("n")) & ~pl.col("t").str.contains(r"[0-9]")))
    kept = tok.join(df.select("country", "t"), on=["country", "t"], how="anti")
    core = kept.group_by("side", "id", maintain_order=True).agg(pl.col("t").str.join(" ").alias("core"))
    return (core.filter(pl.col("side") == 1).select(pl.col("id").alias("s1"), "core"),
            core.filter(pl.col("side") == 2).select(pl.col("id").alias("qid"), "core"))


def core_features(pairs, s1n, qn):
    """Fuzzy similarity of addresses after removing very common tokens."""
    from rapidfuzz import fuzz, process
    x = pairs.select("qid", "s1")
    for col, pref in (("ad", "adc"),):
        cs, cq = _core_strings(s1n, qn, col)
        y = (x.join(cq.rename({"core": "a"}), on="qid", how="left")
              .join(cs.rename({"core": "b"}), on="s1", how="left").fill_null(""))
        a, b = y["a"].to_list(), y["b"].to_list()
        ok = ((y["a"] != "") & (y["b"] != "")).to_numpy()
        scorers = [("tset", fuzz.token_set_ratio), ("ratio", fuzz.ratio)] + ([("partial", fuzz.partial_ratio)] if col == "ad" else [])
        cols = {}
        for nm, sc in scorers:
            v = process.cpdist(a, b, scorer=sc, workers=-1).astype(np.float32)
            cols[f"{pref}_{nm}"] = np.where(ok, v, -1).astype(np.float32)
        x = x.with_columns([pl.Series(k, v) for k, v in cols.items()])
    return x


def sibling_features(pairs, s1n, qn):
    """Sibling agreement counts among the candidates of each S1 entity.

    For a pair (q, s): among the OTHER candidate records of s, count those
    sharing q's house number, core name, (number, name) and full address, and
    those carrying s's own house number / name. Also p-weighted sums.
    """
    qa = qn.select("qid", pl.col("ad_hn").alias("hq"), pl.col("nm_cmp").alias("nq"), pl.col("ad").alias("aq"))
    sa = s1n.select(pl.col("idx").alias("s1"), pl.col("ad_hn").alias("hs"), pl.col("nm_cmp").alias("ns"))
    x = pairs.select("qid", "s1", "p").join(qa, on="qid", how="left").join(sa, on="s1", how="left")
    # count only CONFIDENT siblings (stage-1 p > 0.5): raw candidate counts depend
    # on how many low-score candidates blocking produced, which differs by dataset
    x = x.with_columns((pl.col("p") > 0.5).cast(pl.Int32).alias("_conf"))
    x = x.with_columns((pl.col("hq") == pl.col("hs")).alias("_heq"), (pl.col("nq") == pl.col("ns")).alias("_neq"))
    out = [pl.col("qid"), pl.col("s1")]
    for name, keys, valid in (("hn", ["s1", "hq"], pl.col("hq") != ""),
                              ("nm", ["s1", "nq"], pl.col("nq") != ""),
                              ("both", ["s1", "hq", "nq"], pl.col("hq") != ""),
                              ("ad", ["s1", "aq"], pl.col("aq") != "")):
        out.append(pl.when(valid).then(pl.col("_conf").sum().over(keys) - pl.col("_conf")).otherwise(-1).alias(f"sib_{name}_same"))
        out.append(pl.when(valid).then(pl.col("p").sum().over(keys) - pl.col("p")).otherwise(-1).alias(f"sib_{name}_same_p"))
    for name, flag in (("hn", "_heq"), ("nm", "_neq")):
        f_c = pl.col(flag).cast(pl.Int32) * pl.col("_conf")
        cnt = f_c.sum().over("s1") - f_c
        psum = (pl.col("p") * pl.col(flag).cast(pl.Float32)).sum().over("s1") - pl.col("p") * pl.col(flag).cast(pl.Float32)
        out.append(cnt.alias(f"sib_{name}_eq_s"))
        out.append(psum.alias(f"sib_{name}_eq_s_p"))
    return x.select(out)


def context(pairs, split="train"):
    """Add stage-2 context columns to a (qid, s1, p) frame of `split`."""
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
        pl.col("p").rank("ordinal", descending=True).over("s1").clip(upper_bound=8).alias("s_prank"))
    import blocking as B
    s1n, qn = B.load_norm(split)
    # restrict to the records actually in play, so the "common token" document
    # frequencies in _core_strings are measured on the same population on both
    # sides (the corrected train variant subsamples decoy queries)
    s1n = s1n.filter(pl.col("idx").is_in(pairs["s1"].unique().implode()))
    qn = qn.filter(pl.col("qid").is_in(pairs["qid"].unique().implode()))
    x = x.join(sibling_features(pairs, s1n, qn), on=["qid", "s1"], how="left")
    x = x.join(core_features(pairs, s1n, qn), on=["qid", "s1"], how="left")
    del s1n, qn
    return x.select("qid", "s1", *[pl.col(c).cast(pl.Float32) for c in S2_EXTRA])


def _design(split, ctx, feats):
    """Stage-1 features joined with stage-2 context, chunk by chunk."""
    for f in sorted(TR.feat_dir(split).glob("part_*.parquet")):
        part = pl.read_parquet(f)
        yield part.join(ctx, on=["qid", "s1"], how="inner")


def _name(kind, level):
    """File stem of a level's predictions: oof / oof2 / oof3 ..., test_pred / test_pred2 ..."""
    return kind + ("" if level == 1 else str(level))


def fit_predict_oof(frac=None, level=2):
    # stage 2 early-stops around 100 iterations, so it is not data-hungry; it gets
    # its own (smaller) share so stage 1 can take more without the 82-column design
    # matrix blowing the memory budget
    frac = float(os.environ.get("BER_S2_FRAC", TR.TRAIN_FRAC)) if frac is None else frac
    """Train level-L fold models on OOF context from level L-1; return OOF probs."""
    t0 = time.time()
    oof = pl.read_parquet(C.work(C.MODEL_DIR, _name("oof", level - 1) + ".parquet"))
    ctx = context(oof.select("qid", "s1", "p"), "train")
    feats = TR.feature_names() + S2_EXTRA
    lab = TR.add_folds(oof.select("qid", "s1").lazy()).select("qid", "s1", "y", "fold", "u").collect()
    ctx = ctx.join(lab, on=["qid", "s1"])
    models = []
    del oof
    for k in range(TR.NFOLD):
        # one pass over the chunk files per fold; the full design never sits in memory.
        # The early-stopping rows come from the TRAINING folds, not from fold k.
        other = pl.col("fold") != k
        tr_p, va_p = [], []
        for d in _design("train", ctx, feats):
            tr_p.append(d.filter(other & (pl.col("u") < frac)))
            va_p.append(d.filter(other & (pl.col("u") >= 1 - TR.VAL_FRAC)))
        tr, va = pl.concat(tr_p), pl.concat(va_p)
        del tr_p, va_p
        print(f"stage2 fold {k}: train {tr.height:,} valid {va.height:,} ({time.time() - t0:.0f}s)", flush=True)
        dtr = lgb.Dataset(TR.design(tr, feats), tr["y"].to_numpy(), feature_name=feats)
        dva = lgb.Dataset(TR.design(va, feats), va["y"].to_numpy(), reference=dtr)
        del tr, va
        m = lgb.train(TR.PARAMS, dtr, TR.ROUNDS, valid_sets=[dva],
                      callbacks=[lgb.log_evaluation(200), lgb.early_stopping(50, verbose=False)])
        m.save_model(str(C.work(C.MODEL_DIR, f"s{level}_fold{k}.txt")))
        del dtr, dva
        models.append(m)
        print(f"stage2 fold {k}: best iter {m.best_iteration} ({time.time() - t0:.0f}s)", flush=True)
    # OOF: each pair is scored by the one model that never trained on its fold
    # (model k trains on folds != k); see train.predict_oof for the leak this fixed
    outs = []
    for d in _design("train", ctx, feats):
        X = TR.design(d, feats)
        fold = d["fold"].to_numpy()
        P = np.stack([m.predict(X, num_threads=C.N_THREADS) for m in models])
        own = P[fold, np.arange(len(fold))]
        p = (P.sum(0) - own) / (TR.NFOLD - 1) if TR.LEGACY_OOF else own
        outs.append(d.select("qid", "s1", "y").with_columns(pl.Series("p", p.astype(np.float32))))
        del X, P
    oof2 = pl.concat(outs)
    oof2.write_parquet(C.work(C.MODEL_DIR, _name("oof", level) + ".parquet"))
    return models, oof2


def predict_test(level=2):
    """Level-L probabilities on test from level L-1 fold-average predictions."""
    models = [lgb.Booster(model_file=str(C.work(C.MODEL_DIR, f"s{level}_fold{k}.txt"))) for k in range(TR.NFOLD)]
    p1 = pl.read_parquet(C.work(C.MODEL_DIR, _name("test_pred", level - 1) + ".parquet"))
    ctx = context(p1, "test")
    feats = TR.feature_names("test") + S2_EXTRA
    outs = []
    for part in _design("test", ctx, feats):
        X = TR.design(part, feats)
        p = np.mean([m.predict(X, num_threads=C.N_THREADS) for m in models], axis=0)
        outs.append(part.select("qid", "s1").with_columns(pl.Series("p", p.astype(np.float32))))
    out = pl.concat(outs)
    out.write_parquet(C.work(C.MODEL_DIR, _name("test_pred", level) + ".parquet"))
    return out


if __name__ == "__main__":
    import sys
    fit_predict_oof(level=int(sys.argv[1]) if len(sys.argv) > 1 else 2)
