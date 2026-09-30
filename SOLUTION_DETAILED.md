# Genovate: Business Entity Resolution, detailed write-up

**Amazon ML Challenge 2026**, 25–27 September 2026 (72 hours, Unstop)
**Team Genovate:** Hemanth Raj, Kushal K V, Ayush Khanuja
**Result: rank 677**, public leaderboard **macro F0.5 = 0.984083**, shortlisted in the Top 1,000 teams. For reference, the top three teams scored 0.9918, 0.9915 and 0.9912.

This document explains what we built, what we learned about the data, and what worked and what didn't. For setup and reproduction, see [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md).

---

## 1. The problem

There are three sources of business records (name, address, country), with no shared identifiers. Source 1 (S1) is a deduplicated reference list. For each S1 entity we must list every Source 2/3 record describing the same business: zero, one or many.

- **Metric:** F0.5 per S1 entity, averaged over all entities, with precision weighted twice as much as recall.
  - An entity with no true match (a *singleton*) scores 1.0 only if we predict nothing for it.
  - One wrong match on a singleton costs a full point.
- **Countries:** train covers the US and India. Test adds **France**, which never appears in train, and country must be treated as an open set.
- **Rules:** no external data, APIs or geocoding. Models must be MIT/Apache-2.0 and at most 8B parameters.

## 2. The data

| | S1 | S2 | S3 |
|---|---|---|---|
| Train | 2,206,821 | 5,034,616 | 5,285,603 |
| Test | 1,732,544 (US 663K, India 810K, France 259K) | 4,887,273 | 5,082,316 |

- **Each S2/S3 record matches at most one S1 entity.** So we search from the S2/S3 side and assign every record to its single best entity.
- **3.46 true matches per S1 entity, in both countries, and 5.6% singletons.** The distribution of match counts is identical in the US and India, which marks it as a generator constant.
- **Noise on true copies:**
  - typos and look-alike digits (`cardi0logy`)
  - legal-form swaps (Inc / LLC / Pvt Ltd / SARL)
  - "doing business as" names, and names written as domains or handles
  - Indic-script transliteration (18% of India records)
  - reordered or missing address parts, and house-number truncation
  - empty addresses (4.7% of true copies)
- **Decoys: records that match nothing.** They are look-alikes generated from a *specific* S1 entity: the same name with a nearby house number (`krb india c-115` vs `c-108`), or the same address with a different or invented name. They cause most false positives.
- **France:** S1 carries the *region* (Nouvelle-Aquitaine) where S2/S3 often carry the *department* (Gironde). Names are template compositions ("Nantes Sportive SAS"), so decoys can differ from their entity by a single word or legal form.

## 3. The pipeline

```
raw TSVs → parquet → learned maps → normalisation
        → blocking (per country, top-12 by IDF-weighted keys) → prune (≥ 0.3 × best)
        → 56 pair features → stage-1 LightGBM → +28 context features → stage-2 LightGBM
        → average of 5 model sets → cross-encoder blend on uncertain pairs
        → odds correction → best S1 per record → exact expected-F0.5 set per entity
```

### 3.1 Normalisation

The same rules run for every country:
- **Names:** lowercasing, accent stripping, look-alike digit repair, legal-form canonicalisation, honorific and stop-word removal, DBA splitting, domain and handle extraction.
- **Indic script:** transliterated through a word dictionary *learned from train pairs* (1,347 words, covering 96% of test's Indic words), with a Unicode character-level fallback.
- **Addresses:** street-type abbreviation groups, placeholder removal, and house-number, unit and PO-box parsing.
- **Learned maps:** per-country address substitutions (TX ↔ Texas, Bombay → Mumbai) are learned from train pairs. France simply gets none.

### 3.2 Blocking

For each S2/S3 record we score S1 records of the same country label by summing the IDF weights `log(N/df)` of their shared keys. The keys:
- name tokens and adjacent bigrams
- the whole compact name (catches domains and handles), with its own frequency cap
- sorted 4-character prefixes of name tokens (typo- and order-robust)
- address words, bigrams and digit runs; on the S1 side house numbers also emit truncated variants (1202 → 202 / 120)
- cross keys: compact name × number, name token × number, number × address word, name token × address word

Other details:
- **Caps:** each key type has a document-frequency cap that scales with the S1 partition size, so key selection depends on relative frequency and train and test behave alike.
- **Selection:** we keep the top 12 per record, then a relative cut keeps candidates scoring ≥ 0.3 × the record's best.
- **Result:** 30,008,138 test candidate pairs. On validation, pair recall is 0.9855, and a perfect classifier on these candidates would score **0.9955**.

### 3.3 Stage 1: pair features and LightGBM

56 features:
- **Names:** RapidFuzz ratio, token-sort, token-set, partial and Jaro–Winkler; Levenshtein distance; compact-name ratios; a strict core name without generic words; the DBA alternative name; char-3-gram and word TF-IDF cosine; exact and compact-exact flags; token counts; acronym match; legal-form agreement; source flags (domain, handle, Indic, DBA, id-tag).
- **Addresses:** RapidFuzz ratios and word TF-IDF cosine. The same ratios on a *core address*, with tokens common in that country removed, which strips regions and departments symmetrically. House-number features: exact, truncation, edit distance, relative difference. Digit-run and unit overlap.
- **Ambiguity:** how many S1 records share this name or address.
- **Excluded:** raw blocking scores are *not* features. Their scale depends on index size, and an adversarial check could tell train from test with AUC 0.99 using them.

LightGBM trains on **4 folds grouped by query cluster**, so all records of one entity share a fold. Each fold model uses 67.5% of the clusters.

### 3.4 Stage 2: context features

28 features are computed from out-of-fold stage-1 probabilities:
- **Record side:** the probability share among the record's candidates, the margin to its best competitor, and the second-best probability.
- **Entity side:** probability mass, the count of confident records, and how many records choose this entity.
- **Sibling agreement:** how many other confident candidates of the entity share this record's house number, name or address.
  - A deviation shared by siblings is usually a systematic source format.
  - A *name* deviation shared by siblings marks a look-alike decoy entity.

In practice stage 2 leans mostly on the record-side margin and share. It adds about +0.0036 honest F0.5 over stage 1.

### 3.5 Validation worlds that look like test

The single most important piece of engineering:
- **Size mismatch:** test's S1 is half the size of train's in the US. Every blocking and frequency statistic depends on index size.
- **Our first test-like variant was biased:** it dropped 20% of train entities *after* blocking, which left about 9.5 candidates per query against test's 11.7.
- **The corrected worlds** cut each country's S1 to test's size *before* blocking and sample decoys to test's decoy share.
- **Different seeds** (`BER_WORLD`) give different samples. Models from different worlds ensemble far better than seed-only siblings.
- **The anchored world** (`BER_DECOYS=anchored`) keeps only decoys that imitate entities that are present (§4.3).

### 3.6 Ensemble

The final model averages the stage-2 test probabilities of five model sets:

| Model set | World | Honest local F0.5 |
|---|---|---|
| v4 | 0 | (legacy OOF, §5) |
| v4b | 0 | (legacy OOF) |
| v4bh | 0 | 0.98634 |
| w1h | 1 | 0.98645 |
| aw3 | 3, anchored decoys | 0.98785 |

### 3.7 Cross-encoder on the uncertain pairs

We fine-tuned **intfloat/multilingual-e5-small** (MIT, 118M parameters) as a pair classifier on a 6 GB laptop GPU.
- **Input:** the raw `name | address` strings of both records.
- **Training pairs:** the 1.38M pairs whose tree probability (stage 1 or stage 2) lies in (0.01, 0.99). This is where near-twin decoys, French wording and transliteration live.
- **Honest scores:** the pairs are split into two entity halves; each model trains on one half and scores the other, and test gets the mean of both.
- **Blend:** a logistic regression on `[logit(p_tree), ce_logit]`, fitted on one half of the entities and scored on the other.
- **Fitting a shared laptop GPU:** word embeddings frozen, fixed-shape batches, an out-of-memory guard, and a small host-memory footprint.

| | Held-out macro F0.5 |
|---|---|
| Trees only | 0.98634 |
| + cross-encoder (one run) | 0.98892 |
| + cross-encoder (two runs averaged, **submitted**) | **0.98899** |
| + cross-encoder trained on all pairs (built, never uploaded) | 0.98932 |

On the leaderboard, the blend moved us from 0.98092 to 0.983899, and local held-out gains predicted it well.

### 3.8 Decision layer

1. Each S2/S3 record keeps only its highest-probability S1 entity.
2. For each entity, the assigned records are sorted by probability. We pick the prefix size *k* that maximises the **exact expected F0.5** under independent Bernoulli truths.
   - It is computed with Poisson-binomial dynamic programming and verified against brute force.
   - *k* = 0 earns the singleton credit only when the entity truly has no match.
   - A Poisson term accounts for true matches that blocking or the argmax lost.
3. Before this, every pair's odds `p/(1-p)` are multiplied by a per-country factor: **US/India × 0.35** (from a test-density simulation) and **France × 0.6** (from leaderboard probes).

---

## 4. What we learned about test

### 4.1 The leaderboard split by country

We uploaded one deliberate diagnostic with France's rows emptied. It scored 0.84307, which splits the public score into **US + India ≈ 0.982** and **France ≈ 0.955**, assuming France's singleton rate equals train's. France is 15% of test, so even a perfect US/India score would cap us near 0.993.

### 4.2 Test contains no orphan records

"Doing business as" names appear only on true copies: 2.4% of them, and 0.00% of decoys.
- **In test, 99.99% of DBA records score p > 0.98.** Every test record's entity is present in S1.
- **In a naive validation world** that deletes entities, 19% of those records become orphans: records whose entity is missing.

### 4.3 Test has about twice the look-alike decoys per entity

Empty-address, domain-name and DBA rates differ sharply between true copies and decoys. Solving test's mix from those rates gives about **3.4 true-copy records and 2.3 look-alike decoys per entity**, against 1.2 per entity in train.

Candidate counts confirm it:

| Pruned candidates per query | US | India |
|---|---|---|
| True copies | 1.66 | 2.18 |
| Decoys imitating a present entity | 3.65 | 4.67 |
| Orphans / look-alikes of removed entities | 10–11 | 10–11 |
| **Our old validation world** | 4.35 | 3.50 |
| **Test** | **2.38** | **3.12** |

"True copies plus anchored look-alikes" predicts 2.45 / 3.19 candidates per query, which matches test almost exactly. That led to the **anchored world** and to the **odds correction**: on test, a pair scored *p* is less likely to be real than in training.

---

## 5. The out-of-fold leak we found in our own pipeline

For a period, fold k's out-of-fold scores came from the average of the models j ≠ k. Model j trains on every fold except j, so those were exactly the models that *had* seen fold k; model k was the only honest one. Stage 2 did the same.

| | Leaked | Honest |
|---|---|---|
| Stage 1 local F0.5 | 0.98805 | 0.98293 |
| Stage 2 local F0.5 | 0.99044 | 0.98634 |

Effects:
- Every local number from that period was inflated.
- Stage 2 had been trained on over-confident inputs.
- It explained why single-model local gains reached the leaderboard at only 40–56% while ensembles transferred far better.

The fix scores fold k with model k only. The stage-1 models were unaffected, so we rebuilt stage 2 and all tuning on honest OOF without retraining stage 1. `BER_LEGACY_OOF=1` remains in the code only to regenerate the two early model sets used in the final ensemble.

---

## 6. Leaderboard history

| Upload | Change | Public LB |
|---|---|---|
| v01 | baseline: blocking + 55 features + stage 1 + threshold | 0.976553 |
| v03 | legacy test-like world, stage 2, expected-F0.5 selection | 0.976736 |
| fix | corrected world (S1 cut to test size *before* blocking) | 0.977783 |
| probe | France emptied (diagnostic) | 0.84307 |
| v2 | compact-name blocking key, core-address features | 0.977854 |
| v2 + v3 | ensemble of two worlds | 0.978677 |
| v4 | more training data per fold model | 0.97913 |
| v4 + v4b, odds × 0.5 | second strong model + decoy-density correction | 0.979992 |
| same, odds × 0.35 | stronger correction | 0.980005 |
| v4bh | honest OOF, single model | 0.979403 |
| 5 model sets, odds × 0.3 | + honest v4b, w1, anchored world | 0.98092 |
| + cross-encoder (one run) | e5-small blend on uncertain pairs | 0.983899 |
| + cross-encoder (two runs) | runs averaged | 0.984006 |
| France odds × 0.2 | probe | 0.983818 |
| **France odds × 0.6** | **final** | **0.984083** |

---

## 7. Where the remaining error is

This breakdown was measured on validation with the v4b model set, before the cross-encoder. Total loss was about 0.0096 of F0.5:

| Source | Cost |
|---|---|
| Blocking: the true match is never a candidate | 0.0045 |
| True match seen but rejected | 0.0021 |
| False positives | 0.0016 |
| True match lost the argmax to another entity | 0.0013 |

About 97% of blocking misses are genuinely ambiguous: records with an empty address and a heavily changed or generic name shared by dozens of entities. Extra typo-robust keys (consonant skeleton, sorted-token compact name, 3-character prefixes) would recover only 3% of them. The cross-encoder mostly reduces the *rejected true match* and *false positive* rows.

## 8. What did not work

- **France-specific normalisation**, four attempts, each returning about zero on the leaderboard:
  - rescaling stage-2 features
  - a stricter threshold
  - region/department core-address features
  - learned-map dropout
- **Stage 3** (re-stacking context on stage-2 scores): −0.0027.
- **More blocking keys:** they recover only 3% of misses (§7).
- **A richer cross-encoder blend** (stage-1 score, per-country terms, interactions): +0.00005 at most.
- **Widening the cross-encoder's band:** at most +0.0002.
- **The honest single model alone:** 0.97940 on the leaderboard, which did not beat the earlier two-model ensemble. Honest validation helped mostly by making later gains measurable.

## 9. Engineering notes

- **Hardware:** a laptop with 16 GB RAM, 24 threads, and an RTX 4050 laptop GPU (6 GB). We only noticed the GPU on the final day. Our LightGBM build has no GPU support, so the GPU ran only the cross-encoder.
- **RAM was the binding constraint.** Two heavy jobs at once exhausted it, via a Windows resource error or a CUDA out-of-memory error even with free VRAM, so builds ran strictly one at a time.
- **Build times:** one model set takes about 2 h (blocking 12 min, features 5–8 min, stage 1 about 75 min, stage 2 about 15 min, test prediction about 25 min). One cross-encoder run takes about 70 min.
- **Reproducibility:** `src/run_final.sh` regenerates the submitted file end to end. Blocking breaks ties at the top-12 boundary in an order that can vary between runs, so regenerated candidate sets match to within a small fraction of pairs, not bit for bit.

## 10. Lessons

1. **Measure how test differs from train before tuning anything.** Index size, decoy composition and candidate structure all differed, and fixing the validation world was worth more than any model change.
2. **Audit every out-of-fold path for leakage.** Ours inflated local scores by about 0.004 and quietly mis-trained the stacked model.
3. **A small pretrained cross-encoder, targeted at the uncertain band, can beat hand-crafted features on the hardest pairs.** It was our largest single gain, and we found the GPU only on the last day.
4. **Ensembles of models trained on different data samples transfer to the leaderboard far better than seed-only variations.**
5. **With limited uploads, diagnostic probes pay for themselves.** Examples: the France-empty upload, and one-variable odds probes.

## 11. Repository

| Path | What |
|---|---|
| `code/business_entity_resolution/src/` | the pipeline; `run_final.sh` is the entry point |
| `code/business_entity_resolution/README.md` | setup and reproduction |
| `Documentation_template.md` | methodology document submitted to the organisers |
| `submissions/SUBMISSIONS.md` | every upload with its commit and score |
| `PROJECT_STATUS.md`, `feedback6.md`, `PLAN.md` | working notes kept during the challenge |
| `make_submission_zip.py` | builds the organiser archive and validates it with the official script |

The organiser dataset is not included.
