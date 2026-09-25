"""Dev harness: blocking quality on a sample of train S1 entities (one country)."""
import sys, time
import polars as pl
import blocking as B
import config as C


def main(country="India", frac=0.25, cap=B.CAP, topk=B.TOPK):
    t0 = time.time()
    s1, q = B.load_norm("train")
    s1 = s1.filter(pl.col("country") == country)
    ids = s1["idx"].sample(fraction=frac, seed=C.SEED)
    tp = B.true_pairs().filter(pl.col("s1").is_in(ids))
    qs = q.filter(pl.col("qid").is_in(tp["qid"]))
    cand = B.block_country(s1, qs, cap, topk)
    print(f"{country} frac={frac} cap={cap} topk={topk}: {time.time() - t0:.0f}s")
    B.oracle_report(cand, s1_ids=ids, label=f"{country} sample")
    return cand


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "India", float(sys.argv[2]) if len(sys.argv) > 2 else 0.25)
