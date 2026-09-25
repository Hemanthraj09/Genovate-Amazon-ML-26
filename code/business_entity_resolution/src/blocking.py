"""Step 3: candidate generation (blocking).

Search direction: for every S2/S3 record ("query") find the S1 records it may
match. Each S2/S3 record belongs to at most one S1 entity, so a per-query
top-K is the natural shortlist.

Keys (all computed inside one country partition -- whatever labels exist):
  n:<tok>   core-name token          nb:<a>_<b> adjacent core-name bigram
  n:<cmp>   whole core name without spaces (matches domains / handles)
  a:<tok>   address word             #:<num>    address digit run
  ab:<a>_<b> adjacent address-token bigram
Each key gets IDF weight log(N_s1 / df_s1); keys shared by more than CAP S1
records are ignored. score(q, s) = sum of weights of shared keys.

Output: WORK_DIR/cand/{split}.parquet with
    qid (u32: idx + (src-2)*2^23), s1 (u32), score (f32), nk (u16 shared keys),
    rank (u8, 1 = best for this query)
"""
import math
import time
import polars as pl
import config as C

QSHIFT = 1 << 23
CAP = 200           # max S1 document frequency for a usable key (selective types)
CAP_SINGLE = 60     # tighter cap for single name/address keys (they dominate join cost)
KEY_TYPES = {"n": 0, "nb": 1, "p4": 2, "a": 3, "ab": 4, "#": 5, "nh": 6, "th": 7, "hw": 8, "tw": 9}
SINGLE_TYPES = (0, 3, 4, 5)
CAP_CROSS = 80      # cap for name-token x address and number x word cross keys
CROSS_TYPES = (7, 8, 9)
TOPK = 12           # candidates kept per query (before any relative cut)
QCHUNK = 150_000    # queries per join chunk (memory bound)


def qid_expr(src):
    """Unique query id combining source and row index."""
    return (pl.col("idx") + (src - 2) * QSHIFT).cast(pl.UInt32)


def build_keys(df, id_col, side="q"):
    """Explode a normalized frame into unique (id, key-hash) rows.

    Besides single name/address keys, compound name x address keys
    (name+house number, name word+address word) make generic names that are
    shared by many S1 entities retrievable through their address.
    side="s1" also emits truncated variants of house numbers (drop first or
    last digit), mirroring the noise generator's truncations (1202 -> 202/120).
    """
    tok = df.select(pl.col(id_col).alias("id"),
                    pl.col("nm").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 1)).alias("t"),
                    pl.col("ad").str.split(" ").list.eval(pl.element().filter(pl.element().str.len_chars() > 0)).alias("u"),
                    pl.col("ad_nums").str.split(" ").list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("d"),
                    "nm_cmp")
    tok = tok.with_columns(
        pl.col("u").list.eval(pl.element().filter(
            ~pl.element().str.contains(r"^\d+$") & (pl.element().str.len_chars() > 2))).alias("w"))
    parts = []
    t = tok.select("id", "t").explode("t").drop_nulls()
    parts.append(t.select("id", ("n:" + pl.col("t")).alias("k")))
    cmp_ok = tok.filter(pl.col("nm_cmp").str.len_chars() > 2)
    parts.append(cmp_ok.select("id", ("n:" + pl.col("nm_cmp")).alias("k")))
    nb = tok.select("id", pl.col("t").list.slice(0, pl.col("t").list.len() - 1).alias("a"),
                    pl.col("t").list.slice(1).alias("b")).explode(["a", "b"]).drop_nulls()
    parts.append(nb.select("id", ("nb:" + pl.col("a") + "_" + pl.col("b")).alias("k")))
    # typo/order-robust: sorted set of 4-char prefixes of the name tokens
    p4 = tok.filter(pl.col("t").list.len() >= 2).select(
        "id", ("p4:" + pl.col("t").list.eval(pl.element().str.slice(0, 4)).list.sort().list.join("_")).alias("k"))
    parts.append(p4)
    # address words, bigrams and digit runs
    parts.append(tok.select("id", "w").explode("w").drop_nulls()
                 .select("id", ("a:" + pl.col("w")).alias("k")))
    ab = tok.select("id", pl.col("u").list.slice(0, pl.col("u").list.len() - 1).alias("a"),
                    pl.col("u").list.slice(1).alias("b")).explode(["a", "b"]).drop_nulls()
    parts.append(ab.select("id", ("ab:" + pl.col("a") + "_" + pl.col("b")).alias("k")))
    if side == "s1":
        tok = tok.with_columns(pl.concat_list(
            "d",
            pl.col("d").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3).str.slice(1).str.strip_chars_start("0")),
            pl.col("d").list.eval(pl.element().filter(pl.element().str.len_chars() >= 3).str.head(-1)),
        ).list.eval(pl.element().filter(pl.element() != "")).list.unique().alias("d"))
    d = tok.select("id", "d", "t", "w", "nm_cmp").explode("d").drop_nulls("d")
    parts.append(d.select("id", ("#:" + pl.col("d")).alias("k")))
    # compound name x address keys
    parts.append(d.filter(pl.col("nm_cmp").str.len_chars() > 2)
                 .select("id", ("nh:" + pl.col("nm_cmp") + "|" + pl.col("d")).alias("k")))
    parts.append(d.explode("t").drop_nulls("t")
                 .select("id", ("th:" + pl.col("t") + "|" + pl.col("d")).alias("k")))
    parts.append(d.explode("w").drop_nulls("w")
                 .select("id", ("hw:" + pl.col("d") + "|" + pl.col("w")).alias("k")))
    tw = tok.select("id", "t", "w").explode("t").drop_nulls("t").explode("w").drop_nulls("w")
    parts.append(tw.select("id", ("tw:" + pl.col("t") + "|" + pl.col("w")).alias("k")))
    keys = (pl.concat(parts)
              .select("id", pl.col("k").hash(seed=7).alias("h"),
                      pl.col("k").str.extract(r"^([a-z0-9#]+):", 1).replace_strict(KEY_TYPES).cast(pl.UInt8).alias("ty"))
              .unique(["id", "h"]))
    return keys


def block_country(s1n, qn, cap=CAP, topk=TOPK):
    """Top-K S1 candidates for every query record of one country partition."""
    n1 = s1n.height
    sk = build_keys(s1n, "idx", side="s1").rename({"id": "s1"})
    dfk = sk.group_by("h").agg(pl.len().alias("df"), pl.col("ty").first())
    dfk = dfk.filter(pl.when(pl.col("ty").is_in(SINGLE_TYPES)).then(pl.col("df") <= CAP_SINGLE)
                     .when(pl.col("ty").is_in(CROSS_TYPES)).then(pl.col("df") <= CAP_CROSS)
                     .otherwise(pl.col("df") <= cap)).with_columns(
        (pl.lit(math.log(n1)) - pl.col("df").cast(pl.Float64).log()).cast(pl.Float32).alias("w"))
    sk = sk.drop("ty").join(dfk.select("h", "w"), on="h")
    out = []
    for i in range(0, qn.height, QCHUNK):
        qk = build_keys(qn.slice(i, QCHUNK), "qid").drop("ty").rename({"id": "qid"})
        j = qk.join(sk, on="h")
        sc = (j.group_by("qid", "s1")
               .agg(pl.col("w").sum().alias("score"), pl.len().cast(pl.UInt16).alias("nk")))
        sc = (sc.sort(["qid", "score"], descending=[False, True])
                .with_columns(pl.int_range(1, pl.len() + 1).over("qid").alias("rank"))
                .filter(pl.col("rank") <= topk)
                .with_columns(pl.col("rank").cast(pl.UInt8)))
        out.append(sc)
    return pl.concat(out)


def load_norm(split):
    """Normalized S1 and query (S2+S3) frames for a split."""
    s1 = pl.read_parquet(C.work("norm", f"{split}_s1.parquet"))
    qs = []
    for src in (2, 3):
        q = pl.read_parquet(C.work("norm", f"{split}_s{src}.parquet"))
        qs.append(q.with_columns(qid_expr(src).alias("qid")))
    return s1, pl.concat(qs)


def run(split, cap=CAP, topk=TOPK, countries=None):
    """Block every country partition of a split and save candidates."""
    t0 = time.time()
    s1, q = load_norm(split)
    res = []
    for country in sorted(s1["country"].unique().to_list()):
        if countries and country not in countries:
            continue
        c = block_country(s1.filter(pl.col("country") == country),
                          q.filter(pl.col("country") == country), cap, topk)
        print(f"  {split} {country}: {c.height:,} cand pairs  ({time.time() - t0:.0f}s)", flush=True)
        res.append(c)
    cand = pl.concat(res)
    return cand


def dropped_s1():
    """Train S1 ids removed in the test-like variant (deterministic hash)."""
    s1 = pl.read_parquet(C.work("raw", "train_s1.parquet"), columns=["idx"])
    if C.DROP_FRAC <= 0:
        return s1.head(0)["idx"]
    return s1.filter((pl.col("idx").hash(seed=99) % 1000) < int(C.DROP_FRAC * 1000))["idx"]


def true_pairs():
    """Train ground truth as (qid, s1) rows."""
    p = pl.read_parquet(C.work("raw", "train_pairs.parquet"))
    return p.select((pl.col("rec") + (pl.col("src").cast(pl.UInt32) - 2) * QSHIFT)
                    .cast(pl.UInt32).alias("qid"), "s1")


def oracle_report(cand, s1_ids=None, label=""):
    """Blocking quality: pair recall and oracle macro F0.5 over S1 entities.

    Oracle = a perfect matcher restricted to the candidates (keeps exactly the
    true pairs present). Singletons score 1.0 (oracle predicts empty).
    """
    tp = true_pairs()
    s1 = pl.read_parquet(C.work("raw", "train_s1.parquet")).select("idx", "country")
    if s1_ids is not None:
        s1 = s1.filter(pl.col("idx").is_in(s1_ids))
        tp = tp.filter(pl.col("s1").is_in(s1_ids))
    hit = tp.join(cand.select("qid", "s1", "rank"), on=["qid", "s1"], how="left")
    g = hit.group_by("s1").agg(pl.len().alias("G"), pl.col("rank").is_not_null().sum().alias("TP"))
    e = s1.join(g, left_on="idx", right_on="s1", how="left").fill_null(0)
    e = e.with_columns(pl.when(pl.col("G") == 0).then(1.0).otherwise(
        1.25 * pl.col("TP") / (0.25 * pl.col("G") + pl.col("TP"))).alias("f"))
    rec = hit["rank"].is_not_null().mean()
    lost = e.filter((pl.col("G") > 0) & (pl.col("TP") == 0)).height
    print(f"[{label}] pair recall={rec:.5f}  oracle F0.5={e['f'].mean():.5f}  "
          f"entities losing all matches={lost:,}  cand/query={cand.height / cand['qid'].n_unique():.2f}")
    for k in (1, 2, 3, 5, 8, 12):
        r = hit.filter(pl.col("rank") <= k).height / hit.height
        print(f"    recall@{k}={r:.5f}", end="")
    print()
    by_c = e.group_by("country").agg(pl.col("f").mean()).sort("country")
    print("    oracle by country:", dict(by_c.iter_rows()))
    return e


if __name__ == "__main__":
    import sys
    for split in (sys.argv[1:] or ["train", "test"]):
        cand = run(split)
        cand.write_parquet(C.work("cand", f"{split}.parquet"))
        if split == "train":
            oracle_report(cand, label="train")
        del cand
