"""Step 0: convert the organizer TSVs to parquet.

Each record gets a dense integer row index `idx` (position in its source file)
so later stages can join on small integers instead of string IDs. The original
`entity_id` string is kept for writing the submission.

Output (in WORK_DIR/raw/):
    {split}_s{src}.parquet   columns: idx, entity_id, business_name,
                             business_address, country
    train_pairs.parquet      columns: s1 (idx in train S1), src (2|3),
                             rec (idx in train S2/S3) -- one row per true pair
"""
import time
import polars as pl
import config as C


def read_tsv(path) -> pl.DataFrame:
    """Read an organizer TSV as all-string columns (tab separated, no quoting)."""
    return pl.read_csv(path, separator="\t", quote_char=None, infer_schema=False)


def main():
    t0 = time.time()
    id_maps = {}
    for split in C.SPLITS:
        for src in C.SOURCES:
            df = read_tsv(C.raw_path(split, src))
            df = df.with_columns(pl.col(c).fill_null("") for c in df.columns)
            df = df.with_row_index("idx").with_columns(pl.col("idx").cast(pl.UInt32))
            assert df["entity_id"].n_unique() == df.height, "duplicate entity ids"
            assert df["entity_id"].str.starts_with(f"S{src}-").all(), "bad id prefix"
            df.write_parquet(C.work("raw", f"{split}_s{src}.parquet"))
            if split == "train":
                id_maps[src] = df.select("entity_id", "idx")
            print(f"{split} s{src}: {df.height:,} rows  countries="
                  f"{dict(df['country'].value_counts().iter_rows())}  "
                  f"({time.time() - t0:.0f}s)")

    # Ground truth -> long table of true pairs, keyed by integer indices.
    gt = read_tsv(C.gt_path()).with_columns(pl.col("matched_entity_ids").fill_null(""))
    long = (gt.with_columns(pl.col("matched_entity_ids").str.split(","))
              .explode("matched_entity_ids")
              .filter(pl.col("matched_entity_ids") != "")
              .rename({"source1_entity_id": "s1_id", "matched_entity_ids": "rec_id"}))
    long = long.join(id_maps[1].rename({"entity_id": "s1_id", "idx": "s1"}), on="s1_id")
    parts = []
    for src in (2, 3):
        m = id_maps[src].rename({"entity_id": "rec_id", "idx": "rec"})
        parts.append(long.join(m, on="rec_id").select(
            "s1", pl.lit(src, pl.UInt8).alias("src"), "rec"))
    pairs = pl.concat(parts)
    n_ids = long.height
    assert pairs.height == n_ids, f"unmapped gt ids: {n_ids - pairs.height}"
    pairs.write_parquet(C.work("raw", "train_pairs.parquet"))
    print(f"train pairs: {pairs.height:,}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
