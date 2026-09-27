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
import math
import numpy as np
import polars as pl


def assign_argmax(pairs):
    """Keep, for every query record, only its best-scoring S1 candidate."""
    return (pairs.sort(["qid", "p"], descending=[False, True])
                 .unique("qid", keep="first", maintain_order=False))


def shrink_odds(assigned, r):
    """Multiply every pair's odds p/(1-p) by r before deciding (r < 1 = stricter).

    Test carries about twice the look-alike decoys per S1 entity that the training
    world does. DBA names, which only true copies carry, show test has no orphan
    records at all, so every extra test record is a natural look-alike; the
    uncertain negative mass per entity is 1.9x (stage 1) to 2.2-2.5x (stage 2) the
    world's. A pair the model scores p on train is therefore less likely to be real
    on test. Injecting that decoy density into the OOF predictions and re-deciding
    puts the optimum near r = 0.4-0.5 (+0.0006 on US/India, singletons 0.975 ->
    0.985), at a cost of only 0.0002 on the unshifted world.

    `r` may be a dict keyed by country (needs a `country` column); "*" is the
    default for countries not listed.
    """
    if isinstance(r, dict):
        r = pl.col("country").replace_strict(r, default=r.get("*", 1.0), return_dtype=pl.Float64)
    elif r == 1.0:
        return assigned
    p = pl.col("p")
    return assigned.with_columns((p * r / (p * r + 1 - p)).alias("p"))


def parse_odds(s):
    """BER_ODDS value: "0.5" -> 0.5, "US:0.5,France:0.3,*:0.5" -> dict."""
    s = (s or "1").strip()
    if ":" not in s:
        return float(s)
    return {k.strip(): float(v) for k, v in (kv.split(":") for kv in s.split(","))}


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


MAX_MISS = 2       # unretrieved true matches modelled per entity (Poisson, truncated)


def expected_f_best_k(P, lam=None):
    """Best prefix size k and its expected F0.5 for each row of sorted probs P.

    `lam`: per-row expected number of true matches that are *not* in P at all --
    lost in blocking, or argmax-assigned to another entity. The metric's G counts
    those, so ignoring them shrinks the denominator and makes every extra
    prediction look dearer than it is; the optimiser then under-predicts, and
    claims singleton credit it has not earned. Modelled as Poisson(lam) truncated
    at MAX_MISS. lam=None reproduces the original, biased behaviour.
    """
    m, n = P.shape
    L, R = _pb_left(P), _pb_right(P)
    if lam is None:
        w = [np.ones(m)]
    else:
        lam = np.asarray(lam, dtype=np.float64).reshape(m)
        w = [np.exp(-lam) * lam ** j / math.factorial(j) for j in range(MAX_MISS + 1)]
        w[-1] = np.maximum(0.0, 1.0 - sum(w[:-1]))      # tail folded onto MAX_MISS
    E = np.zeros((m, n + 1))
    E[:, 0] = R[0][:, 0] * w[0]                          # empty scores 1 only if G == 0
    for k in range(1, n + 1):
        t = np.arange(k + 1)[:, None]
        o = np.arange(n - k + 1)[None, :]
        for j, wj in enumerate(w):
            F = 1.25 * t / (k + 0.25 * (t + o + j))
            E[:, k] += wj * np.einsum("mt,mo,to->m", L[k], R[k], F)
    return E.argmax(1), E.max(1)


def by_expected_f(assigned, max_n=16, miss_odds=None):
    """Per-entity expected-F0.5-optimal subset of its assigned records.

    `miss_odds`: (1-rho)/rho, where rho is the share of true pairs that survive
    blocking and argmax. A float, or a dict keyed by country when `assigned` has a
    country column. The expected number of missed matches for an entity is taken
    as miss_odds * (its retrieved probability mass). None disables the correction.
    """
    a = assigned.sort(["s1", "p"], descending=[False, True]).with_columns(
        pl.int_range(1, pl.len() + 1).over("s1").alias("pos"),
        pl.len().over("s1").alias("n"),
        pl.col("p").sum().over("s1").alias("psum"))
    if miss_odds is not None:
        odds = (pl.col("country").replace_strict(miss_odds, default=0.0)
                if isinstance(miss_odds, dict) and "country" in a.columns
                else pl.lit(float(miss_odds if not isinstance(miss_odds, dict) else 0.0)))
        a = a.with_columns((pl.col("psum") * odds).alias("lam"))
    a = a.filter(pl.col("pos") <= max_n).with_columns(pl.col("n").clip(upper_bound=max_n))
    keep = []
    for n in sorted(a["n"].unique().to_list()):
        g = a.filter(pl.col("n") == n).sort(["s1", "pos"])
        P = g["p"].to_numpy().reshape(-1, n).astype(np.float64)
        lam = g["lam"].to_numpy().reshape(-1, n)[:, 0] if miss_odds is not None else None
        k, _ = expected_f_best_k(P, lam)
        kk = np.repeat(k, n)
        keep.append(g.filter(pl.Series(g["pos"].to_numpy() <= kk)).select("s1", "qid"))
    return pl.concat(keep) if keep else assigned.head(0).select("s1", "qid")


def _self_test():
    """Sanity checks of the expected-F optimizer."""
    k, e = expected_f_best_k(np.array([[0.02, 0.01]]))
    assert k[0] == 0, k                         # nothing likely -> predict empty
    k, e = expected_f_best_k(np.array([[0.99, 0.98, 0.1]]))
    assert k[0] == 2, k
    # brute-force check against enumeration, with and without missing matches
    rng = np.random.default_rng(0)
    for lam in (None, 0.0, 0.05, 0.4, 1.3):
        if lam is None:
            wm = [1.0]
        else:
            wm = [math.exp(-lam) * lam ** j / math.factorial(j) for j in range(MAX_MISS + 1)]
            wm[-1] = max(0.0, 1.0 - sum(wm[:-1]))
        for _ in range(30):
            p = np.sort(rng.random(4))[::-1]
            best = []
            for kk in range(5):
                tot = 0.0
                for mask in range(16):
                    truth = [(mask >> i) & 1 for i in range(4)]
                    pr = np.prod([p[i] if truth[i] else 1 - p[i] for i in range(4)])
                    TP = sum(truth[:kk])
                    for j, wj in enumerate(wm):
                        G = sum(truth) + j
                        f = (1.0 if G == 0 else 0.0) if kk == 0 else (
                            0.0 if TP == 0 else 1.25 * TP / (kk + 0.25 * G))
                        tot += pr * wj * f
                best.append(tot)
            k, e = expected_f_best_k(p[None, :], None if lam is None else [lam])
            assert abs(e[0] - max(best)) < 1e-9, (lam, e, best)
    # a missing-match prior must never make the optimiser predict *fewer* records
    p = np.array([[0.62, 0.41, 0.18]])
    k0, _ = expected_f_best_k(p)
    k1, _ = expected_f_best_k(p, [0.5])
    assert k1[0] >= k0[0], (k0, k1)
    print("decide self-test OK")


if __name__ == "__main__":
    _self_test()
