"""Step 6: turn pair probabilities into per-entity match sets.

1. One S1 per record: each S2/S3 record keeps only its highest-probability
   S1 candidate (S1 is deduplicated, so a record belongs to at most one S1).
2. Per S1 entity, choose the match set:
   * `by_threshold`: keep assigned records with p >= tau (simple baseline);
   * `by_expected_f`: sort the entity's assigned records by p and pick the
     prefix size k (0 = predict empty) maximizing the *exact* expected F0.5
     under independent Bernoulli(p) truths:
         E_k = sum_{t,o} P(TP=t) P(O=o) * 1.25 t / (k + 0.25 (t + o))
         E_0 = P(no record is a true match)       (singleton credit)
     where TP ~ PoissonBinomial(top-k p), O ~ PoissonBinomial(rest).
"""
import numpy as np
import polars as pl


def assign_argmax(pairs):
    """Keep, for every query record, only its best-scoring S1 candidate."""
    return (pairs.sort(["qid", "p"], descending=[False, True])
                 .unique("qid", keep="first", maintain_order=False))


def by_threshold(assigned, tau):
    """Baseline decision: accept assigned pairs with probability >= tau."""
    return assigned.filter(pl.col("p") >= tau).select("s1", "qid")


def _pb_left(P):
    """Poisson-binomial distributions of prefix sums: list over k of (m, k+1)."""
    m, n = P.shape
    L = [np.ones((m, 1))]
    for j in range(n):
        prev, pj = L[-1], P[:, j:j + 1]
        new = np.zeros((m, j + 2))
        new[:, :-1] += prev * (1 - pj)
        new[:, 1:] += prev * pj
        L.append(new)
    return L


def _pb_right(P):
    """Distributions of suffix sums: R[k] covers items k..n-1, shape (m, n-k+1)."""
    m, n = P.shape
    R = [None] * (n + 1)
    R[n] = np.ones((m, 1))
    for j in range(n - 1, -1, -1):
        prev, pj = R[j + 1], P[:, j:j + 1]
        new = np.zeros((m, n - j + 1))
        new[:, :-1] += prev * (1 - pj)
        new[:, 1:] += prev * pj
        R[j] = new
    return R


def expected_f_best_k(P):
    """Best prefix size k and its expected F0.5 for each row of sorted probs P."""
    m, n = P.shape
    L, R = _pb_left(P), _pb_right(P)
    E = np.zeros((m, n + 1))
    E[:, 0] = R[0][:, 0]
    for k in range(1, n + 1):
        t = np.arange(k + 1)[:, None]
        o = np.arange(n - k + 1)[None, :]
        F = 1.25 * t / (k + 0.25 * (t + o))
        E[:, k] = np.einsum("mt,mo,to->m", L[k], R[k], F)
    return E.argmax(1), E.max(1)


def by_expected_f(assigned, max_n=16):
    """Per-entity expected-F0.5-optimal subset of its assigned records."""
    a = assigned.sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1").alias("pos"),
        pl.len().over("s1").alias("n"))
    a = a.filter(pl.col("pos") <= max_n).with_columns(pl.col("n").clip(upper_bound=max_n))
    keep = []
    for n in sorted(a["n"].unique().to_list()):
        g = a.filter(pl.col("n") == n).sort(["s1", "pos"])
        P = g["p"].to_numpy().reshape(-1, n).astype(np.float64)
        k, _ = expected_f_best_k(P)
        kk = np.repeat(k, n)
        keep.append(g.filter(pl.Series(g["pos"].to_numpy() <= kk)).select("s1", "qid"))
    return pl.concat(keep) if keep else assigned.head(0).select("s1", "qid")


def _self_test():
    """Sanity checks of the expected-F optimizer."""
    k, e = expected_f_best_k(np.array([[0.02, 0.01]]))
    assert k[0] == 0, k                         # nothing likely -> predict empty
    k, e = expected_f_best_k(np.array([[0.99, 0.98, 0.1]]))
    assert k[0] == 2, k
    # brute-force check against enumeration
    rng = np.random.default_rng(0)
    for _ in range(50):
        p = np.sort(rng.random(4))[::-1]
        best = []
        for kk in range(5):
            tot = 0.0
            for mask in range(16):
                truth = [(mask >> i) & 1 for i in range(4)]
                pr = np.prod([p[i] if truth[i] else 1 - p[i] for i in range(4)])
                G, TP = sum(truth), sum(truth[:kk])
                f = (1.0 if G == 0 else 0.0) if kk == 0 else (0.0 if TP == 0 else 1.25 * TP / (kk + 0.25 * G))
                tot += pr * f
            best.append(tot)
        k, e = expected_f_best_k(p[None, :])
        assert abs(e[0] - max(best)) < 1e-9, (e, best)
    print("decide self-test OK")


if __name__ == "__main__":
    _self_test()
