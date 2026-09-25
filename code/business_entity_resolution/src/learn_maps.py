"""Learn normalization maps from the provided training pairs (no external data).

1. indic_word: Indic-script word -> Latin word. For true pairs whose S2/S3
   name contains Indic script and has the same number of tokens as the S1
   name, tokens are aligned by position and the majority Latin token wins.
   This inverts the noise generator's transliteration on seen vocabulary.
2. addr_comp[country]: whole address component substitutions (e.g. state
   'mh' -> 'maharashtra', 'texas' -> 'tx', 'bombay' -> 'mumbai'), mined from
   pairs whose component sets differ by exactly one component on each side.
   Direction is always S2/S3 form -> S1 (reference) form.
3. addr_tok[country]: single-token substitutions mined the same way at token
   level, after the hand-written abbreviation map has been applied.

Output: WORK_DIR/maps.json
"""
import collections
import json
import re
import time
import polars as pl
import config as C
import textnorm as T

MIN_COUNT = 25     # minimum support for a learned substitution
MIN_SHARE = 0.5    # the substitution must explain >= this share of the source form's diffs


def load_pairs(n_sample=None):
    """True train pairs joined with raw name/address of both sides."""
    s1 = pl.read_parquet(C.work("raw", "train_s1.parquet")).select(
        "idx", pl.col("business_name").alias("n1"), pl.col("business_address").alias("a1"),
        "country")
    pairs = pl.read_parquet(C.work("raw", "train_pairs.parquet"))
    if n_sample:
        pairs = pairs.sample(n_sample, seed=C.SEED)
    parts = []
    for src in (2, 3):
        r = pl.read_parquet(C.work("raw", f"train_s{src}.parquet")).select(
            "idx", pl.col("business_name").alias("n2"), pl.col("business_address").alias("a2"))
        parts.append(pairs.filter(pl.col("src") == src)
                     .join(r, left_on="rec", right_on="idx")
                     .join(s1, left_on="s1", right_on="idx"))
    return pl.concat(parts)


def learn_indic_words(df):
    """Position-aligned Indic->Latin word dictionary from true name pairs."""
    df = df.filter(pl.col("n2").str.contains(r"[ऀ-ൿ]"))
    cnt = collections.defaultdict(collections.Counter)
    for n1, n2 in zip(df["n1"], df["n2"]):
        t1 = [w.strip(".,()[]").lower() for w in n1.split()]
        t2 = [w.strip(".,()[]") for w in n2.split()]
        t1 = [w for w in t1 if w]
        t2 = [w for w in t2 if w]
        if len(t1) != len(t2):
            continue
        for a, b in zip(t2, t1):
            if T.INDIC_RE.search(a) and not T.INDIC_RE.search(b):
                cnt[a][b] += 1
    out = {}
    for w, c in cnt.items():
        (best, n), tot = c.most_common(1)[0], sum(c.values())
        if n >= 2 and n / tot >= 0.5:
            out[w] = best
    print(f"indic words learned: {len(out):,} (from {df.height:,} indic-name pairs)")
    return out


def _mine(pairs_of_sets, s1_count):
    """Count one-to-one substitutions between two sets that differ by one item.

    A substitution x->y is kept when it has enough support and explains most
    of the one-to-one events whose source item is x. The source form must be
    rare on the S1 (reference) side: if S1 itself uses x regularly, x is a
    legitimate value (e.g. a real city), not a noisy variant, so it is kept.
    """
    sub = collections.Counter()
    src_tot = collections.Counter()
    for a1, a2 in pairs_of_sets:
        add, rem = a2 - a1, a1 - a2
        if len(add) == 1 and len(rem) == 1:
            x, y = next(iter(add)), next(iter(rem))
            sub[(x, y)] += 1
            src_tot[x] += 1
    out = {}
    for (x, y), n in sub.most_common():
        if n < MIN_COUNT or x in out or n / src_tot[x] < MIN_SHARE:
            continue
        if any(ch.isdigit() for ch in x + y) or s1_count[x] > 0.2 * n:
            continue
        if x in T.ADDR_ABBR.values():          # never remap hand-written canonicals
            continue
        out[x] = y
    return out


def learn_addr_maps(df):
    """Per-country component-level and token-level address substitution maps."""
    comp_maps, tok_maps = {}, {}
    for country in df["country"].unique().to_list():
        d = df.filter(pl.col("country") == country)
        raw_comps, comps = [], []
        for a1, a2 in zip(d["a1"], d["a2"]):
            if not a2:
                continue
            c1, c2 = T._addr_basic_components(a1), T._addr_basic_components(a2)
            raw_comps.append((c1, c2))
            # compare components after hand-written token normalization
            comps.append((set(" ".join(T._addr_comp_tokens(c)) for c in c1),
                          set(" ".join(T._addr_comp_tokens(c)) for c in c2)))
        s1_comp = collections.Counter(c for c1, _ in comps for c in c1)
        cm = _mine(comps, s1_comp)
        comp_maps[country] = cm
        toks = []
        for c1, c2 in comps:
            t1 = set(t for c in c1 for t in cm.get(c, c).split())
            t2 = set(t for c in c2 for t in cm.get(c, c).split())
            toks.append((t1, t2))
        s1_tok = collections.Counter(t for t1, _ in toks for t in t1)
        tm = _mine(toks, s1_tok)
        # Don't chain onto hand-written canonical forms in a conflicting way.
        tok_maps[country] = {k: v for k, v in tm.items() if k != v}
        print(f"{country}: {len(cm)} component subs, {len(tok_maps[country])} token subs")
        print("   comp:", list(cm.items()))
        print("   tok:", list(tok_maps[country].items()))
    return comp_maps, tok_maps


def main():
    t0 = time.time()
    df = load_pairs()
    print(f"pairs loaded: {df.height:,} ({time.time() - t0:.0f}s)")
    maps = {"indic_word": learn_indic_words(df), "addr_comp": {}, "addr_tok": {}}
    T.MAPS.update(maps)                    # address mining sees transliterated text
    sample = df.sample(min(df.height, 1_500_000), seed=C.SEED)
    maps["addr_comp"], maps["addr_tok"] = learn_addr_maps(sample)
    with open(C.work("maps.json"), "w", encoding="utf-8") as f:
        json.dump(maps, f, ensure_ascii=False)
    print(f"saved maps.json ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
