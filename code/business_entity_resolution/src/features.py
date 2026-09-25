"""Step 4: pair features for the (query, S1) candidate pairs.

All statistics (TF-IDF weights, name/address sharing counts) are computed on
the split being processed (train stats from train files, test stats from
test files) -- unsupervised, no labels involved.

Feature groups
  blocking : blocking score, shared-key count, rank / relative score within
             the query's candidates, candidate counts on both sides
  name     : rapidfuzz ratios (ratio, token sort/set, partial, Jaro-Winkler),
             compact-name ratios (domains/handles), DBA-alternative match,
             char-3gram / word TF-IDF cosine, legal-form agreement, flags
  address  : rapidfuzz ratios, word TF-IDF cosine, house-number exact /
             truncation / conflict, digit-run overlap, unit overlap
  ambiguity: how many S1 records share this name / address
The country label is never a feature.
"""
import time
import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
import config as C
import blocking as B

CHUNK = 1_500_000
NFEAT_HASH = 1 << 20


# ------------------------------------------------------------ candidates
def prune_candidates(cand, topk, rel):
    """Keep rank <= topk and score >= rel * best; add query-level context."""
    q = cand.group_by("qid").agg(
        pl.col("score").max().alias("q_best"),
        pl.col("score").sort(descending=True).get(1, null_on_oob=True).fill_null(0).alias("q_second"),
        pl.len().alias("q_ncand_raw"))
    c = cand.join(q, on="qid")
    c = c.filter((pl.col("rank") <= topk) & (pl.col("score") >= rel * pl.col("q_best")))
    s = c.group_by("s1").agg(pl.len().alias("s_ncand"))
    c = c.join(s, on="s1").with_columns(
        pl.col("score").rank("ordinal", descending=True).over("s1").alias("s_rank"))
    return c


# --------------------------------------------------------------- tf-idf
def _hash_chunk(texts, analyzer, ngram):
    """Hash a chunk of texts into term-count vectors (runs in a worker)."""
    hv = HashingVectorizer(analyzer=analyzer, ngram_range=ngram, n_features=NFEAT_HASH,
                           alternate_sign=False, norm=None, lowercase=False,
                           token_pattern=r"\S+")
    return hv.transform(texts)


def tfidf_pair(s1_texts, q_texts, analyzer, ngram):
    """L2-normalized TF-IDF matrices for S1 and query texts (shared IDF)."""
    texts = s1_texts + q_texts
    step = 400_000
    mats = Parallel(n_jobs=C.N_THREADS)(
        delayed(_hash_chunk)(texts[i:i + step], analyzer, ngram)
        for i in range(0, len(texts), step))
    X = sp.vstack(mats).tocsr()
    X = TfidfTransformer(sublinear_tf=True).fit_transform(X).astype(np.float32).tocsr()
    n1 = len(s1_texts)
    return X[:n1], X[n1:]


def rowwise_cos(A, ia, Bm, ib):
    """Cosine of row pairs (A[ia[i]], B[ib[i]]) for L2-normalized sparse rows."""
    out = np.empty(len(ia), dtype=np.float32)
    step = 500_000
    for i in range(0, len(ia), step):
        out[i:i + step] = np.asarray(A[ia[i:i + step]].multiply(Bm[ib[i:i + step]]).sum(1)).ravel()
    return out


# ---------------------------------------------------------------- tables
def load_tables(split):
    """Normalized S1 / query tables with row positions and sharing counts."""
    s1, q = B.load_norm(split)
    s1 = s1.with_row_index("srow")
    q = q.with_row_index("qrow")
    nm_f = s1.group_by("country", "nm").agg(pl.len().alias("nm_freq"))
    ad_f = s1.filter(pl.col("ad") != "").group_by("country", "ad").agg(pl.len().alias("ad_freq"))
    s1 = (s1.join(nm_f, on=["country", "nm"], how="left")
            .join(ad_f, on=["country", "ad"], how="left").fill_null(0))
    q = (q.join(nm_f.rename({"nm_freq": "q_nm_freq"}), on=["country", "nm"], how="left")
          .join(ad_f.rename({"ad_freq": "q_ad_freq"}), on=["country", "ad"], how="left")
          .fill_null(0))
    return s1, q


def _cp(a, b, scorer):
    """Pairwise rapidfuzz scores for aligned lists, multithreaded."""
    return process.cpdist(a, b, scorer=scorer, workers=-1).astype(np.float32)


def chunk_features(p, mats):
    """Compute the feature frame for one chunk of joined pairs."""
    f = {}
    nq, ns = p["nm"].to_list(), p["nm_1"].to_list()
    f["nm_ratio"] = _cp(nq, ns, fuzz.ratio)
    f["nm_tsort"] = _cp(nq, ns, fuzz.token_sort_ratio)
    f["nm_tset"] = _cp(nq, ns, fuzz.token_set_ratio)
    f["nm_partial"] = _cp(nq, ns, fuzz.partial_ratio)
    f["nm_jw"] = _cp(nq, ns, JaroWinkler.normalized_similarity)
    cq, cs = p["nm_cmp"].to_list(), p["nm_cmp_1"].to_list()
    f["cmp_ratio"] = _cp(cq, cs, fuzz.ratio)
    f["cmp_partial"] = _cp(cq, cs, fuzz.partial_ratio)
    f["nms_tset"] = _cp(p["nm_s"].to_list(), p["nm_s_1"].to_list(), fuzz.token_set_ratio)
    alt = p["nm_alt"].to_list()
    f["alt_tset"] = np.where(p["nm_alt"].str.len_chars().to_numpy() > 0, _cp(alt, ns, fuzz.token_set_ratio), -1).astype(np.float32)
    aq, as_ = p["ad"].to_list(), p["ad_1"].to_list()
    f["ad_ratio"] = _cp(aq, as_, fuzz.ratio)
    f["ad_tset"] = _cp(aq, as_, fuzz.token_set_ratio)
    f["ad_tsort"] = _cp(aq, as_, fuzz.token_sort_ratio)
    f["ad_partial"] = _cp(aq, as_, fuzz.partial_ratio)
    qr, sr = p["qrow"].to_numpy(), p["srow"].to_numpy()
    for name, (Ms, Mq) in mats.items():
        f[name] = rowwise_cos(Ms, sr, Mq, qr)
    out = pl.DataFrame(f)
    # vectorized structural features
    e = p.select(
        "qid", "s1",
        pl.col("score").cast(pl.Float32), pl.col("nk").cast(pl.Float32), pl.col("rank").cast(pl.Float32),
        (pl.col("score") / pl.col("q_best")).cast(pl.Float32).alias("rel"),
        (pl.col("q_best") - pl.col("score")).cast(pl.Float32).alias("gap_best"),
        (pl.col("score") - pl.col("q_second")).cast(pl.Float32).alias("gap_second"),
        pl.col("q_ncand_raw").cast(pl.Float32), pl.col("s_ncand").cast(pl.Float32),
        pl.col("s_rank").cast(pl.Float32),
        (pl.col("nm") == pl.col("nm_1")).cast(pl.Float32).alias("nm_exact"),
        (pl.col("nm_cmp") == pl.col("nm_cmp_1")).cast(pl.Float32).alias("cmp_exact"),
        pl.col("nm").str.count_matches(" ").add(1).cast(pl.Float32).alias("nm_ntok_q"),
        pl.col("nm_1").str.count_matches(" ").add(1).cast(pl.Float32).alias("nm_ntok_s"),
        (pl.col("nm_leg") == pl.col("nm_leg_1")).cast(pl.Float32).alias("leg_eq"),
        (pl.col("nm_leg").str.split(" ").list.set_intersection(pl.col("nm_leg_1").str.split(" "))
         .list.eval(pl.element().filter(pl.element() != "")).list.len()).cast(pl.Float32).alias("leg_common"),
        (pl.col("nm_leg") == "").cast(pl.Float32).alias("leg_q_empty"),
        (pl.col("nm_leg_1") == "").cast(pl.Float32).alias("leg_s_empty"),
        (pl.col("flags") & 1).gt(0).cast(pl.Float32).alias("q_domain"),
        (pl.col("flags") & 2).gt(0).cast(pl.Float32).alias("q_handle"),
        (pl.col("flags") & 4).gt(0).cast(pl.Float32).alias("q_indic"),
        (pl.col("flags") & 8).gt(0).cast(pl.Float32).alias("q_dba"),
        (pl.col("flags") & 16).gt(0).cast(pl.Float32).alias("q_idtag"),
        pl.col("nm_freq").cast(pl.Float32).alias("s_nm_freq"),
        pl.col("q_nm_freq").cast(pl.Float32),
        pl.col("ad_freq").cast(pl.Float32).alias("s_ad_freq"),
        pl.col("q_ad_freq").cast(pl.Float32),
        (pl.col("ad") == "").cast(pl.Float32).alias("ad_q_empty"),
        pl.col("ad").str.count_matches(" ").add(1).cast(pl.Float32).alias("ad_ntok_q"),
        pl.col("ad_1").str.count_matches(" ").add(1).cast(pl.Float32).alias("ad_ntok_s"),
        ((pl.col("ad_hn") != "") & (pl.col("ad_hn") == pl.col("ad_hn_1"))).cast(pl.Float32).alias("hn_eq"),
        ((pl.col("ad_hn") != "") & (pl.col("ad_hn_1") != "") & (pl.col("ad_hn") != pl.col("ad_hn_1")) & (
            pl.col("ad_hn_1").str.ends_with(pl.col("ad_hn")) | pl.col("ad_hn_1").str.starts_with(pl.col("ad_hn")) |
            pl.col("ad_hn").str.ends_with(pl.col("ad_hn_1")) | pl.col("ad_hn").str.starts_with(pl.col("ad_hn_1")))
         ).cast(pl.Float32).alias("hn_trunc"),
        (pl.col("ad_hn") == "").cast(pl.Float32).alias("hn_q_missing"),
        (pl.col("ad_hn_1") == "").cast(pl.Float32).alias("hn_s_missing"),
        pl.col("ad_nums").str.split(" ").list.set_intersection(pl.col("ad_nums_1").str.split(" "))
          .list.eval(pl.element().filter(pl.element() != "")).list.len().cast(pl.Float32).alias("nums_common"),
        pl.col("ad_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.len().cast(pl.Float32).alias("nums_q_n"),
        pl.col("ad_nums_1").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.len().cast(pl.Float32).alias("nums_s_n"),
        pl.col("ad_unit").str.split(" ").list.set_intersection(pl.col("ad_unit_1").str.split(" "))
          .list.eval(pl.element().filter(pl.element() != "")).list.len().cast(pl.Float32).alias("unit_common"),
        (pl.col("qid") >= B.QSHIFT).cast(pl.Float32).alias("src3"),
    )
    return pl.concat([e, out], how="horizontal")


def build(split, topk=6, rel=0.3, tag=None):
    """Compute features for all pruned candidates of a split; save parquet."""
    t0 = time.time()
    cand = pl.read_parquet(C.work("cand", f"{split}.parquet"))
    cand = prune_candidates(cand, topk, rel)
    print(f"[{split}] pruned candidates: {cand.height:,} ({time.time() - t0:.0f}s)", flush=True)
    s1, q = load_tables(split)
    mats = {}
    # (address char n-grams are skipped: ~5 GB for 12M addresses; rapidfuzz covers them)
    mats["nm_char_cos"] = tfidf_pair(s1["nm"].to_list(), q["nm"].to_list(), "char_wb", (3, 3))
    mats["nm_word_cos"] = tfidf_pair(s1["nm"].to_list(), q["nm"].to_list(), "word", (1, 1))
    mats["ad_word_cos"] = tfidf_pair(s1["ad"].to_list(), q["ad"].to_list(), "word", (1, 1))
    print(f"[{split}] tf-idf ready ({time.time() - t0:.0f}s)", flush=True)
    scols = ["idx", "srow", "nm", "nm_s", "nm_cmp", "nm_leg", "ad", "ad_nums", "ad_hn", "ad_unit",
             "nm_freq", "ad_freq"]
    qcols = ["qid", "qrow", "nm", "nm_s", "nm_cmp", "nm_alt", "nm_leg", "flags", "ad", "ad_nums",
             "ad_hn", "ad_unit", "q_nm_freq", "q_ad_freq"]
    s1t = s1.select(scols).rename({c: c + "_1" for c in scols if c not in ("idx", "srow", "nm_freq", "ad_freq")})
    qt = q.select(qcols)
    out_dir = C.work("feat", tag or split, "x").parent
    for old in out_dir.glob("part_*.parquet"):
        old.unlink()
    n = 0
    for i in range(0, cand.height, CHUNK):
        p = (cand.slice(i, CHUNK).join(qt, on="qid", how="left")
                 .join(s1t, left_on="s1", right_on="idx", how="left"))
        fe = chunk_features(p, mats)
        fe.write_parquet(out_dir / f"part_{i // CHUNK:04d}.parquet")
        n += fe.height
        print(f"   chunk {i // CHUNK + 1}/{(cand.height - 1) // CHUNK + 1} ({time.time() - t0:.0f}s)", flush=True)
    print(f"[{split}] features for {n:,} pairs saved to {out_dir} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    import sys
    for split in (sys.argv[1:] or ["train", "test"]):
        build(split)
