"""Dev utility: size-shift simulation (which model survives test's smaller S1?).

Test's S1 partitions are smaller than train's (US 663K vs 1.32M, India 810K vs
883K), which changes every blocking-derived statistic. This script builds a
"mini world" from train with test's sizes and decoy share, re-runs blocking
with the ORIGINAL absolute caps (as v04/v05 were built), computes features, and
scores two model sets out-of-fold on it. Comparing each model's score on the
mini world with its ordinary out-of-fold score on the same entities shows how
much it degrades under the size shift.

    BER_VARIANT=tl python sizeshift.py
"""
import json
import pickle
import sys
import time
import numpy as np
import polars as pl
import lightgbm as lgb
import config as C
import blocking as B
import decide as D
import features as F
import train as TR
import tune

sys.path.insert(0, str(C.WORK_DIR / "dev"))
import stage2 as S2_NEW          # robust context (v05)
import stage2_v04 as S2_OLD      # v04 context, taken from git tag v04

SIZE = {"US": 663_106 / 1_323_633, "India": 809_986 / 883_188}   # test S1 / train S1
DECOY_SHARE = 0.41
MODELS = {"v04": ("model_tl", S2_OLD), "v05": ("model_tl_robust", S2_NEW)}


def build_world():
    """Sample S1 entities to test's sizes and queries to test's decoy share."""
    s1, q = B.load_norm("train")
    drop = B.dropped_s1()
    s1 = s1.filter(~pl.col("idx").is_in(drop))
    tp = B.true_pairs()
    keep = []
    for c, f in SIZE.items():
        ids = s1.filter((pl.col("country") == c) &
                        ((pl.col("idx").cast(pl.UInt64).hash(seed=77) % 10000) < int(f * 10000)))["idx"]
        keep.append(ids)
    keep = pl.concat(keep)
    s1m = s1.filter(pl.col("idx").is_in(keep))
    true_q = tp.filter(pl.col("s1").is_in(keep))["qid"]
    qs = []
    for c in SIZE:
        qc = q.filter(pl.col("country") == c)
        qt = qc.filter(pl.col("qid").is_in(true_q))
        qd = qc.filter(~pl.col("qid").is_in(tp["qid"]) | pl.col("qid").is_in(tp.filter(~pl.col("s1").is_in(keep))["qid"]))
        n_dec = min(qd.height, int(qt.height * DECOY_SHARE / (1 - DECOY_SHARE)))
        qs += [qt, qd.sample(n_dec, seed=7)]
        print(f"  mini {c}: S1 {s1m.filter(pl.col('country') == c).height:,}  true recs {qt.height:,}  decoys {n_dec:,}", flush=True)
    return s1m, pl.concat(qs), keep


def block_and_featurize(s1m, qm):
    """Blocking with the original absolute caps, pruning and pair features."""
    cands = []
    for c in SIZE:
        s1c, qc = s1m.filter(pl.col("country") == c), qm.filter(pl.col("country") == c)
        B.CAP_REF_N = s1c.height                 # scale = 1 -> original absolute caps
        cands.append(B.block_country(s1c, qc))
    cand = F.prune_candidates(pl.concat(cands), 12, 0.3)
    s1m, qm = F.add_core_address(s1m, qm)
    s1t = s1m.with_row_index("srow")
    qt = qm.with_row_index("qrow")
    nm_f = s1t.group_by("country", "nm").agg(pl.len().alias("nm_freq"))
    ad_f = s1t.filter(pl.col("ad") != "").group_by("country", "ad").agg(pl.len().alias("ad_freq"))
    s1t = s1t.join(nm_f, on=["country", "nm"], how="left").join(ad_f, on=["country", "ad"], how="left").fill_null(0)
    qt = (qt.join(nm_f.rename({"nm_freq": "q_nm_freq"}), on=["country", "nm"], how="left")
            .join(ad_f.rename({"ad_freq": "q_ad_freq"}), on=["country", "ad"], how="left").fill_null(0))
    mats = {"nm_char_cos": F.tfidf_pair(s1t["nm"].to_list(), qt["nm"].to_list(), "char_wb", (3, 3)),
            "nm_word_cos": F.tfidf_pair(s1t["nm"].to_list(), qt["nm"].to_list(), "word", (1, 1)),
            "ad_word_cos": F.tfidf_pair(s1t["ad"].to_list(), qt["ad"].to_list(), "word", (1, 1))}
    scols = ["idx", "srow", "nm", "nm_s", "nm_cmp", "nm_leg", "ad", "ad_core", "ad_nums", "ad_hn", "ad_unit", "nm_freq", "ad_freq"]
    qcols = ["qid", "qrow", "nm", "nm_s", "nm_cmp", "nm_alt", "nm_leg", "flags", "ad", "ad_core", "ad_nums", "ad_hn",
             "ad_unit", "q_nm_freq", "q_ad_freq"]
    s1c = s1t.select(scols).rename({c: c + "_1" for c in scols if c not in ("idx", "srow", "nm_freq", "ad_freq")})
    outs = []
    for i in range(0, cand.height, F.CHUNK):
        p = cand.slice(i, F.CHUNK).join(qt.select(qcols), on="qid", how="left").join(s1c, left_on="s1", right_on="idx", how="left")
        outs.append(F.chunk_features(p, mats))
    return pl.concat(outs)


def score_model(tag, feats, keep, truth, s1u):
    """Out-of-fold stage 1 + stage 2 + the model's own decision rule."""
    mdir, s2mod = MODELS[tag]
    x = TR.add_folds(feats.lazy()).collect()
    m1 = [lgb.Booster(model_file=str(C.work(mdir, f"lgb_fold{k}.txt"))) for k in range(TR.NFOLD)]
    f1 = m1[0].feature_name()
    p = np.zeros(x.height, dtype=np.float32)
    fold = x["fold"].to_numpy()
    for k, m in enumerate(m1):
        sel = fold == k
        p[sel] = m.predict(x.filter(pl.Series(sel)).select(f1).to_numpy(), num_threads=C.N_THREADS)
    p1 = x.select("qid", "s1").with_columns(pl.Series("p", p))
    ctx = s2mod.context(p1, "train")
    x2 = x.join(ctx, on=["qid", "s1"], how="inner")
    m2 = [lgb.Booster(model_file=str(C.work(mdir, f"s2_fold{k}.txt"))) for k in range(TR.NFOLD)]
    f2 = m2[0].feature_name()
    p2 = np.zeros(x2.height, dtype=np.float32)
    fold2 = x2["fold"].to_numpy()
    for k, m in enumerate(m2):
        sel = fold2 == k
        p2[sel] = m.predict(x2.filter(pl.Series(sel)).select(f2).to_numpy(), num_threads=C.N_THREADS)
    pairs = x2.select("qid", "s1").with_columns(pl.Series("p", p2))
    dec = json.load(open(C.work(mdir, "decision_oof2.json")))
    a = D.assign_argmax(pairs)
    if dec["rule"] == "ef_iso":
        iso = pickle.load(open(C.work(mdir, "isotonic_oof2.pkl"), "rb"))
        pred = D.by_expected_f(tune.calibrate(a, iso))
    elif dec["rule"] == "ef_raw":
        pred = D.by_expected_f(a)
    else:
        pred = D.by_threshold(a, dec["tau"])
    shifted = tune.report(pred.filter(pl.col("s1").is_in(keep)), truth, s1u, f"{tag} SIZE-SHIFTED world")
    # same entities, ordinary out-of-fold world
    o = pl.read_parquet(C.work(mdir, "oof2.parquet"))
    a0 = D.assign_argmax(o)
    if dec["rule"] == "ef_iso":
        pred0 = D.by_expected_f(tune.calibrate(a0, iso))
    elif dec["rule"] == "ef_raw":
        pred0 = D.by_expected_f(a0)
    else:
        pred0 = D.by_threshold(a0, dec["tau"])
    normal = tune.report(pred0.filter(pl.col("s1").is_in(keep)), truth, s1u, f"{tag} ordinary OOF world")
    print(f"==> {tag}: degradation under size shift = {normal - shifted:+.5f}", flush=True)


def main():
    t0 = time.time()
    s1m, qm, keep = build_world()
    feats = block_and_featurize(s1m, qm)
    print(f"mini-world pairs {feats.height:,} ({time.time() - t0:.0f}s)", flush=True)
    truth = B.true_pairs().filter(pl.col("s1").is_in(keep))
    s1u = s1m.select("idx", "country")
    for tag in MODELS:
        score_model(tag, feats, keep, truth, s1u)
    print(f"done ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
