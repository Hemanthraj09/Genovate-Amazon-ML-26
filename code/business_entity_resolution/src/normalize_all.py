"""Step 2: normalize every record of every split/source (parallel).

Output (WORK_DIR/norm/{split}_s{src}.parquet), one row per record:
    idx, country, nm, nm_s, nm_cmp, nm_alt, nm_leg, flags,
    ad, ad_nums, ad_hn, ad_unit
"""
import time
from multiprocessing import Pool
import polars as pl
import config as C
import textnorm as T

CHUNK = 40_000
NAME_KEYS = ("nm", "nm_s", "nm_cmp", "nm_alt", "nm_leg", "flags")
ADDR_KEYS = ("ad", "ad_nums", "ad_hn", "ad_unit")


def _init():
    """Worker initializer: load the learned maps once per process."""
    T.load_maps(C.work("maps.json"))


def _work(batch):
    """Normalize a batch of (name, address, country) tuples -> column lists."""
    cols = {k: [] for k in NAME_KEYS + ADDR_KEYS}
    for name, addr, country in batch:
        n = T.norm_name(name)
        a = T.norm_address(addr, country)
        for k in NAME_KEYS:
            cols[k].append(n[k])
        for k in ADDR_KEYS:
            cols[k].append(a[k])
    return cols


def normalize_frame(df, pool):
    """Normalize a raw source frame using the worker pool."""
    rows = list(zip(df["business_name"].to_list(), df["business_address"].to_list(),
                    df["country"].to_list()))
    batches = [rows[i:i + CHUNK] for i in range(0, len(rows), CHUNK)]
    cols = {k: [] for k in NAME_KEYS + ADDR_KEYS}
    for res in pool.imap(_work, batches):
        for k in cols:
            cols[k].extend(res[k])
    out = df.select("idx", "country").with_columns(
        [pl.Series(k, cols[k], dtype=pl.UInt8 if k == "flags" else pl.String)
         for k in NAME_KEYS + ADDR_KEYS])
    return out


def main():
    t0 = time.time()
    with Pool(C.N_THREADS, initializer=_init) as pool:
        for split in C.SPLITS:
            for src in C.SOURCES:
                df = pl.read_parquet(C.work("raw", f"{split}_s{src}.parquet"))
                out = normalize_frame(df, pool)
                out.write_parquet(C.work("norm", f"{split}_s{src}.parquet"))
                print(f"{split} s{src}: {out.height:,} rows  ({time.time() - t0:.0f}s)",
                      flush=True)


if __name__ == "__main__":
    main()
