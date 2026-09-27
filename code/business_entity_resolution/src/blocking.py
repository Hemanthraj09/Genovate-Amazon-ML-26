"""Step 3: candidate generation (blocking).

Search direction: for every S2/S3 record ("query") find the S1 records it may
match. Each S2/S3 record belongs to at most one S1 entity, so a per-query
top-K is the natural shortlist.

Keys (all computed inside one country partition -- whatever labels exist):
  n:<tok>   core-name token          nb:<a>_<b> adjacent core-name bigram
  nc:<cmp>  whole core name without spaces (matches domains / handles)
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
# "nc" (the whole compact name) is a *signature* of the record, not a single
# token, so it keeps the generous CAP rather than the tight CAP_SINGLE. It used to
# share the "n:" prefix and therefore the tight cap, which is what left
# address-less records with generic names unreachable: they emit no address key at
# all, so the compact name is often their only selective key. Blocking recall on
# records with an empty address was 75.7% against 99.5% for the rest.
KEY_TYPES = {"n": 0, "nb": 1, "p4": 2, "a": 3, "ab": 4, "#": 5, "nh": 6, "th": 7,
             "hw": 8, "tw": 9, "nc": 10}
SINGLE_TYPES = (0, 3, 4, 5)
CAP_CROSS = 80      # cap for name-token x address and number x word cross keys
CROSS_TYPES = (7, 8, 9)
CAP_REF_N = 1_000_000   # caps above are defined for an S1 partition of this size
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
    parts.append(cmp_ok.select("id", ("nc:" + pl.col("nm_cmp")).alias("k")))
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
    # caps scale with the S1 partition size so that key selection depends on a
    # key's *relative* frequency; absolute caps made test (smaller S1) keep far
    # more keys than train, shifting every blocking-derived statistic
    scale = n1 / CAP_REF_N
    c_single = max(10, round(CAP_SINGLE * scale))
    c_cross = max(10, round(CAP_CROSS * scale))
    c_other = max(20, round(cap * scale))
    sk = build_keys(s1n, "idx", side="s1").rename({"id": "s1"})
    dfk = sk.group_by("h").agg(pl.len().alias("df"), pl.col("ty").first())
    dfk = dfk.filter(pl.when(pl.col("ty").is_in(SINGLE_TYPES)).then(pl.col("df") <= c_single)
                     .when(pl.col("ty").is_in(CROSS_TYPES)).then(pl.col("df") <= c_cross)
                     .otherwise(pl.col("df") <= c_other)).with_columns(
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


def cand_path(split):
    """Candidate file of a split (train files are per-variant, so they coexist)."""
    return C.work("cand", f"{C.TRAIN_TAG if split == 'train' else split}.parquet")


def kept_s1():
    """Train S1 ids the corrected variant keeps: each country cut to test's size."""
    s1 = pl.read_parquet(C.work("raw", "train_s1.parquet"), columns=["idx", "country"])
    keep = [s1.filter(~pl.col("country").is_in(list(C.SIZE_MATCH)))["idx"]]
    for c, f in C.SIZE_MATCH.items():
        keep.append(s1.filter((pl.col("country") == c) &
                              ((pl.col("idx").cast(pl.UInt64).hash(seed=77 + C.WORLD) % 10_000)
                               < int(f * 10_000)))["idx"])
    return pl.concat(keep)


def select_queries(q):
    """Corrected variant: all records of kept entities + decoys cut to DECOY_SHARE.

    Every record that truly matches a kept entity is retained, so the evaluation
    truth stays complete. The unmatched pool (records matching nothing, plus the
    records of the entities we removed) is thinned per country to that country's
    target decoy share by a deterministic hash of qid, so the selection is
    reproducible and independent of frame order. Where the pool is too small to
    reach the target -- India, see config.DECOY_SHARE -- the whole pool is used
    and the achieved share is reported.
    """
    tp = true_pairs()
    true_q = tp.filter(pl.col("s1").is_in(kept_s1().implode()))["qid"]
    if C.ANCHORED:
        return select_anchored(q, tp, true_q)
    q = q.with_columns(pl.col("qid").is_in(true_q.implode()).alias("_true"),
                       pl.col("qid").is_in(tp["qid"].unique().implode()).alias("_hasmatch"),
                       (pl.col("qid").cast(pl.UInt64).hash(seed=88 + C.WORLD) % 10_000).alias("_h"))
    out = []
    for country in sorted(q["country"].unique().to_list()):
        qc = q.filter(pl.col("country") == country)
        n_true = int(qc["_true"].sum())
        # two kinds of decoy, and the mix matters more than the total. A NATURAL
        # decoy matches nothing even in full train -- a generated look-alike, so it
        # still scores well against something and its candidate list stays sharp.
        # An ORPHAN only became a decoy because we deleted its entity, so its best
        # remaining candidate is unrelated, its blocking profile goes flat, and far
        # more candidates survive the relative prune. Drawing decoys at random left
        # train US at 4.62 pruned candidates per query against test's 2.38. Test
        # looks like full train plus withheld entities, so aim for full train's
        # natural-decoy rate and use orphans only to top up.
        nat = qc.filter(~pl.col("_true") & ~pl.col("_hasmatch"))
        orp = qc.filter(~pl.col("_true") & pl.col("_hasmatch"))
        nat_rate = 1.0 - qc["_hasmatch"].mean()          # this country's rate in full train
        target = C.DECOY_SHARE.get(country, 0.40)
        want = min(nat.height + orp.height, round(n_true * target / (1 - target)))
        want_nat = min(nat.height, round(nat_rate * (n_true + want)))
        want_orp = min(orp.height, want - want_nat)
        parts = [qc.filter(pl.col("_true"))]
        for label, pool, k in (("natural", nat, want_nat), ("orphan", orp, want_orp)):
            thr = 0 if pool.height == 0 else math.ceil(10_000 * k / pool.height)
            parts.append(pool.filter(pl.col("_h") < thr))
        sel = pl.concat(parts)
        got = sel.height - n_true
        print(f"    select {country}: true {n_true:,}  decoys {got:,} "
              f"({parts[1].height:,} natural + {parts[2].height:,} orphan, "
              f"natural rate in full train {nat_rate:.3f})  -> share {got / sel.height:.4f} "
              f"(target {target:.3f}{', pool exhausted' if want >= nat.height + orp.height else ''})",
              flush=True)
        out.append(sel)
    return pl.concat(out).drop("_true", "_hasmatch", "_h")


def select_anchored(q, tp, true_q):
    """Anchored world: true records of kept entities + the decoys that imitate them.

    Test's decoys are all look-alikes of entities PRESENT in its S1: its pruned
    candidates per query (US 2.38, India 3.12) are exactly what true copies plus
    anchored look-alikes produce (2.45 / 3.19), and DBA-flagged records show it
    has no orphans. Orphans and look-alikes of removed entities have no strong
    candidate, so ~11 weak ones survive the prune each; they inflated our world to
    4.35 candidates per query and skewed every stage-2 context feature. A natural
    decoy's anchor is its rank-1 S1 in full-train blocking (all S1 present).
    """
    anchor = (pl.scan_parquet(str(C.work("cand", "train.parquet"))).filter(pl.col("rank") == 1)
                .select("qid", pl.col("s1").alias("anchor")).collect())
    nat = (anchor.join(tp.select("qid").unique(), on="qid", how="anti")
                 .filter(pl.col("anchor").is_in(kept_s1().implode()))["qid"])
    sel = q.filter(pl.col("qid").is_in(true_q.implode()) | pl.col("qid").is_in(nat.implode()))
    for country in sorted(sel["country"].unique().to_list()):
        sc = sel.filter(pl.col("country") == country)
        n_true = int(sc["qid"].is_in(true_q.implode()).sum())
        print(f"    anchored {country}: true {n_true:,}  anchored decoys {sc.height - n_true:,} "
              f"-> share {(sc.height - n_true) / sc.height:.4f}", flush=True)
    return sel


def run(split, cap=CAP, topk=TOPK, countries=None):
    """Block every country partition of a split and save candidates."""
    t0 = time.time()
    s1, q = load_norm(split)
    if split == "train" and C.DROP_BEFORE_BLOCKING:
        s1 = s1.filter(pl.col("idx").is_in(kept_s1().implode()))
        q = select_queries(q)
        print(f"  corrected variant: S1 {s1.height:,} entities, queries {q.height:,} "
              f"({time.time() - t0:.0f}s)", flush=True)
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
    """Train S1 ids excluded from the evaluation universe by the active variant.

    "fix": the complement of kept_s1() (already absent from the candidate sets,
    because they were removed before blocking).
    "tl":  a flat DROP_FRAC hash sample, removed from candidates in features.py.
    """
    s1 = pl.read_parquet(C.work("raw", "train_s1.parquet"), columns=["idx"])
    if C.DROP_BEFORE_BLOCKING:
        return s1.filter(~pl.col("idx").is_in(kept_s1().implode()))["idx"]
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
        cand.write_parquet(cand_path(split))
        if split == "train":
            oracle_report(cand, s1_ids=kept_s1() if C.DROP_BEFORE_BLOCKING else None, label="train")
        del cand
