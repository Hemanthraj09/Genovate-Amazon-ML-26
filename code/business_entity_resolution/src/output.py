"""Step 7: write the two submission files (format rules from the challenge).

matching_results.tsv : source1_entity_id <TAB> matched_entity_ids
candidate_pairs.tsv  : source1_entity_id <TAB> candidate_entity_ids

Guarantees (checked here, then re-checked with utils/validate_submission.py):
  * exactly one row per test Source-1 entity (every country, incl. unseen ones)
  * empty list when there is no match / no candidate
  * only S2-/S3- IDs that exist in the test files, no duplicates in a list
  * matches are a subset of candidates (both come from the same pair table)
"""
import polars as pl
import config as C
import blocking as B


def id_tables(split):
    """(s1 idx -> entity_id) and (qid -> entity_id) lookup tables."""
    s1 = pl.read_parquet(C.work("raw", f"{split}_s1.parquet")).select("idx", "entity_id")
    qs = []
    for src in (2, 3):
        q = pl.read_parquet(C.work("raw", f"{split}_s{src}.parquet")).select("idx", "entity_id")
        qs.append(q.select(B.qid_expr(src).alias("qid"), pl.col("entity_id").alias("rec_id")))
    return s1, pl.concat(qs)


def _lists(pairs, s1_ids, q_ids, col):
    """One row per S1 entity with a comma-joined, de-duplicated ID list."""
    g = (pairs.select("s1", "qid").unique()
              .join(q_ids, on="qid", how="inner")
              .sort(["s1", "rec_id"])
              .group_by("s1").agg(pl.col("rec_id").str.join(",").alias(col)))
    out = (s1_ids.join(g, left_on="idx", right_on="s1", how="left")
                 .select(pl.col("entity_id").alias("source1_entity_id"), pl.col(col).fill_null("")))
    return out


def write(matches, candidates, split="test", out_dir=None):
    """Write matching_results.tsv and candidate_pairs.tsv; return their paths."""
    out_dir = out_dir or C.OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    s1_ids, q_ids = id_tables(split)
    # matches must be a subset of candidates
    extra = matches.select("s1", "qid").join(candidates.select("s1", "qid"), on=["s1", "qid"], how="anti")
    assert extra.height == 0, f"{extra.height} matches not in candidates"
    m = _lists(matches, s1_ids, q_ids, "matched_entity_ids")
    c = _lists(candidates, s1_ids, q_ids, "candidate_entity_ids")
    assert m.height == s1_ids.height == c.height
    assert m["source1_entity_id"].n_unique() == m.height
    mp, cp = out_dir / "matching_results.tsv", out_dir / "candidate_pairs.tsv"
    m.write_csv(mp, separator="\t", quote_style="never")
    c.write_csv(cp, separator="\t", quote_style="never")
    n_nonempty = (m["matched_entity_ids"] != "").sum()
    print(f"wrote {mp} ({m.height:,} rows, {n_nonempty:,} non-empty) and {cp}")
    return mp, cp
