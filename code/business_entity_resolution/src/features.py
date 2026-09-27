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
import os
import time
import numpy as np
import polars as pl
import scipy.sparse as sp
from joblib import Parallel, delayed
from rapidfuzz import fuzz, process
from rapidfuzz.distance import JaroWinkler, Levenshtein
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
    # fewer workers than threads: each loky worker commits its own copy of the
    # chunk, and 22 of them exhausted the Windows commit limit (WinError 1450)
    mats = Parallel(n_jobs=int(os.environ.get("BER_TFIDF_JOBS", "8")))(
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
def load_tables(split, drop=None, keep_q=None):
    """Normalized S1 / query tables with row positions and sharing counts.

    `drop`:   S1 ids to remove (test-like variants); their records stay as queries.
    `keep_q`: query ids to retain (corrected variant, which subsamples decoys), so
              the TF-IDF corpus and the S1 name/address frequencies are computed on
              the same record population the model will be trained on.
    """
    s1, q = B.load_norm(split)
    if drop is not None and len(drop):
        s1 = s1.filter(~pl.col("idx").is_in(drop))
    if keep_q is not None:
        q = q.filter(pl.col("qid").is_in(keep_q))
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


CORE_FRAC = 0.02      # address tokens above this share of a country's records are "common"


def add_core_address(s1, q):
    """Add `ad_core`: each address minus the tokens that are common in its country.

    France S1 records carry a region name (Hauts-de-France 39%, Nouvelle-Aquitaine
    33%, Pays de la Loire 28%) where their S2/S3 counterparts carry the department
    instead (Nord 11%, Gironde 11%, Loire-Atlantique 9%) -- those tokens appear on
    one side and essentially never on the other, so they only ever subtract from a
    true pair's address similarity. US and India avoid this because the learned maps
    canonicalise their state names; France has no maps because it is absent from
    train. Dropping country-common tokens removes regions, departments, cities and
    street types symmetrically from both sides, leaving the part of the address that
    actually identifies the building. Digits are always kept.
    """
    both = pl.concat([s1.select(pl.lit(1, pl.UInt8).alias("side"), pl.col("idx").alias("id"), "country", "ad"),
                      q.select(pl.lit(2, pl.UInt8).alias("side"), pl.col("qid").alias("id"), "country", "ad")])
    n = both.group_by("country").agg(pl.len().alias("n"))
    tok = (both.select("side", "id", "country", pl.col("ad").str.split(" ").alias("t"))
               .explode("t").filter(pl.col("t") != ""))
    common = (tok.unique(["side", "id", "t"]).group_by("country", "t").agg(pl.len().alias("df"))
                 .join(n, on="country")
                 .filter((pl.col("df") > CORE_FRAC * pl.col("n")) & ~pl.col("t").str.contains(r"[0-9]"))
                 .select("country", "t"))
    core = (tok.join(common, on=["country", "t"], how="anti")
               .group_by("side", "id", maintain_order=True)
               .agg(pl.col("t").str.join(" ").alias("ad_core")))
    s1 = s1.join(core.filter(pl.col("side") == 1).select(pl.col("id").alias("idx"), "ad_core"),
                 on="idx", how="left").with_columns(pl.col("ad_core").fill_null(""))
    q = q.join(core.filter(pl.col("side") == 2).select(pl.col("id").alias("qid"), "ad_core"),
               on="qid", how="left").with_columns(pl.col("ad_core").fill_null(""))
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
    # same comparisons on the country-common-token-stripped address (see
    # add_core_address): the only address signal France's region/department
    # mismatch cannot corrupt
    kq, ks = p["ad_core"].to_list(), p["ad_core_1"].to_list()
    ok = ((p["ad_core"] != "") & (p["ad_core_1"] != "")).to_numpy()
    for nm, sc in (("tset", fuzz.token_set_ratio), ("ratio", fuzz.ratio), ("partial", fuzz.partial_ratio)):
        f[f"adk_{nm}"] = np.where(ok, _cp(kq, ks, sc), -1).astype(np.float32)
    # numeric near-twin features: noise usually costs one edit on the house
    # number (truncation / one digit), look-alike decoys usually differ more
    hq, hs = p["ad_hn"].to_list(), p["ad_hn_1"].to_list()
    both = ((p["ad_hn"] != "") & (p["ad_hn_1"] != "")).to_numpy()
    f["hn_lev"] = np.where(both, _cp(hq, hs, Levenshtein.distance), -1).astype(np.float32)
    uq, us = p["ad_nums"].to_list(), p["ad_nums_1"].to_list()
    f["nums_tset"] = _cp(uq, us, fuzz.token_set_ratio)
    f["nums_nlev"] = _cp(uq, us, Levenshtein.normalized_distance)
    f["nm_lev"] = _cp(nq, ns, Levenshtein.distance)
    f["cmp_lev"] = _cp(cq, cs, Levenshtein.distance)
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
        ((pl.col("ad_hn").cast(pl.Float64, strict=False) - pl.col("ad_hn_1").cast(pl.Float64, strict=False)).abs()
         / pl.max_horizontal(pl.col("ad_hn").cast(pl.Float64, strict=False),
                             pl.col("ad_hn_1").cast(pl.Float64, strict=False), pl.lit(1.0))
         ).fill_null(-1).cast(pl.Float32).alias("hn_reldiff"),
        # acronym: record name = initials of the S1 name (e.g. 'af' for 'ace foundation')
        (pl.col("nm_1").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join("").alias("_ini")
         == pl.col("nm_cmp")).cast(pl.Float32).alias("acr_eq"),
        (pl.col("nm_cmp").str.starts_with(pl.col("nm_1").str.split(" ").list.eval(pl.element().str.slice(0, 1)).list.join(""))
         & (pl.col("nm_1").str.count_matches(" ") >= 1)).cast(pl.Float32).alias("acr_prefix"),
    )
    return pl.concat([e, out], how="horizontal")


def build(split, topk=12, rel=0.3, tag=None):
    """Compute features for all pruned candidates of a split; save parquet."""
    t0 = time.time()
    cand = pl.read_parquet(B.cand_path(split))
    drop = B.dropped_s1() if split == "train" else None
    keep_q = None
    if split == "train" and C.DROP_BEFORE_BLOCKING:
        # blocking already excluded these entities and thinned the decoy queries,
        # so the candidate lists are already test-shaped: do not touch them here
        keep_q = cand["qid"].unique()
        print(f"[{split}] corrected variant: {len(drop):,} S1 entities and the surplus "
              f"decoy queries were excluded before blocking", flush=True)
    elif drop is not None and len(drop):
        cand = (cand.filter(~pl.col("s1").is_in(drop))
                    .sort(["qid", "score"], descending=[False, True])
                    .with_columns(pl.int_range(1, pl.len() + 1).over("qid").cast(pl.UInt8).alias("rank")))
        print(f"[{split}] legacy test-like variant: dropped {len(drop):,} S1 entities "
              f"AFTER blocking (biases candidate counts, see feedback6.md)", flush=True)
    tag = tag or (C.TRAIN_TAG if split == "train" else split)
    cand = prune_candidates(cand, topk, rel)
    print(f"[{split}] pruned candidates: {cand.height:,} ({time.time() - t0:.0f}s)", flush=True)
    s1, q = load_tables(split, drop, keep_q)
    s1, q = add_core_address(s1, q)
    print(f"[{split}] core addresses ready ({time.time() - t0:.0f}s)", flush=True)
    mats = {}
    # (address char n-grams are skipped: ~5 GB for 12M addresses; rapidfuzz covers them)
    mats["nm_char_cos"] = tfidf_pair(s1["nm"].to_list(), q["nm"].to_list(), "char_wb", (3, 3))
    mats["nm_word_cos"] = tfidf_pair(s1["nm"].to_list(), q["nm"].to_list(), "word", (1, 1))
    mats["ad_word_cos"] = tfidf_pair(s1["ad"].to_list(), q["ad"].to_list(), "word", (1, 1))
    print(f"[{split}] tf-idf ready ({time.time() - t0:.0f}s)", flush=True)
    scols = ["idx", "srow", "nm", "nm_s", "nm_cmp", "nm_leg", "ad", "ad_core", "ad_nums", "ad_hn",
             "ad_unit", "nm_freq", "ad_freq"]
    qcols = ["qid", "qrow", "nm", "nm_s", "nm_cmp", "nm_alt", "nm_leg", "flags", "ad", "ad_core",
             "ad_nums", "ad_hn", "ad_unit", "q_nm_freq", "q_ad_freq"]
    s1t = s1.select(scols).rename({c: c + "_1" for c in scols if c not in ("idx", "srow", "nm_freq", "ad_freq")})
    qt = q.select(qcols)
    out_dir = C.work("feat", tag, "x").parent
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
