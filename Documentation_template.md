# ML Challenge 2026: Business Entity Resolution Solution Template

**Team Name:** Genovate  
**Team Members:** Hemanth Raj, Kushal K V, Ayush Khanuja  
**Submission Date:** 27 September 2026

---

## 1. Executive Summary

We generate candidates with IDF-weighted multi-key blocking, then score each (Source 2/3 record, Source 1 entity) pair with a two-stage LightGBM matcher. On the pairs the trees are unsure about, a small pretrained multilingual cross-encoder (multilingual-e5-small, MIT, 118M parameters, fine-tuned on a laptop GPU) adds a second opinion. Stage 1 uses 56 pairwise string features. Stage 2 adds 28 context features computed from the stage-1 probabilities of competing pairs. Each record is assigned to its best entity, and each entity's match set is chosen by maximising the *exact* expected F0.5.

Our main technical contribution is a validation world that behaves like the test set, plus three measured findings about how test differs from train:

- the test S1 is half the size of train's
- test has no "orphan" records
- test has about twice the look-alike decoys per entity

We correct for each of these. We also found and fixed an out-of-fold leak in our own pipeline.

---

## 2. Methodology

### 2.1 Problem Analysis

- **Scale.** Train S1 has 2.21M entities (US 1.32M, India 0.88M) and 10.3M S2/S3 records. Test S1 has 1.73M entities (US 663k, India 810k, France 259k) and 10.0M records.
- **Cluster structure is a generator constant.** Every train S1 entity has 3.46 true matches on average, in both countries independently (US 3.458, India 3.464). 5.6% of entities are singletons. The distribution of match counts is the same in both countries.
- **Noise on true copies.** True copies carry typos and look-alike digits (`cardi0logy`), legal-form changes, word reordering, DBA/FKA prefixes, domain or handle forms (`www.acme.com`, `@acme`), Indic-script transliteration (18% of India records), dropped or reordered address components, abbreviated street types, house-number truncation, and empty addresses (4.7%).
- **Decoys.** About 26% of train S2/S3 records match nothing. They are look-alikes generated from a specific S1 entity: the same name with a different house number, or the same address with a different name. These drive most false positives.
- **Test is not train-shaped.** We checked this in three ways:
  1. *Size.* Test's S1 per country is 50% (US) and 92% (India) of train's. Every blocking and frequency statistic depends on index size, so we cut the validation world to test's size before blocking, not after.
  2. *No orphans.* DBA-style names occur only on true copies (2.4% of them, and 0.00% of decoys). In test, 99.99% of DBA records score p > 0.98, which means every test record's entity is present. A naive "drop entities" validation world instead turns 19% of those records into orphans whose entity is missing.
  3. *Twice the decoys.* Empty-address, domain-name and DBA rates all differ sharply between true copies and decoys. From them we solved for test's composition: about 3.4 true-copy-type records and about 2.3 look-alike decoys per entity, against 1.2 decoys per entity in train.
- **France.** France exists only in test. Its S1 addresses carry the *region* (Hauts-de-France, Nouvelle-Aquitaine, Pays de la Loire), while its S2/S3 records often carry the *department* (Nord, Gironde, Loire-Atlantique). Its names are generic compositions (city + word + legal form), so decoys often differ from their entity by one word or one legal form.

### 2.2 Solution Strategy

**Approach Type:** Hybrid. Blocking, then a two-stage gradient-boosted classifier, then a fine-tuned transformer cross-encoder on uncertain pairs, then an exact expected-F0.5 decision layer.

**Core Innovation:**
1. A **test-shaped validation world**. Each country's S1 is cut to test's size *before* blocking. Decoys are sampled to test's decoy share, with natural look-alikes preferred over orphans. Several such worlds are drawn with different hash seeds, and each gives a strong, diverse model for the ensemble.
2. An **exact expected-F0.5 subset selector** per entity. It uses a Poisson-binomial dynamic program over the entity's assigned records and a Poisson term for true matches that never reached the candidate set.
3. A **test-decoy-density correction**: every pair's odds are multiplied by a constant before selection, because test carries about twice the look-alike decoys the model trained with.
4. A **cross-encoder on the uncertain band**. multilingual-e5-small reads the raw name and address of both records together. It is fine-tuned on the 1.4M training pairs whose tree probability lies in (0.01, 0.99), using two entity halves so every training pair gets an honest score. A logistic blend with the tree logit, tuned on held-out entities, lifts honest macro F0.5 from 0.98634 to 0.98892.

---

## 3. Candidate Generation (Blocking)

Blocking searches from each S2/S3 record ("query") into the S1 records of the same country label. Country is treated as an open set of partition labels, so France is blocked like any other country.

- **Blocking keys used:** every key is IDF-weighted with log(N_S1 / df). A query's score for an entity is the sum of the weights of the keys they share.
  - name tokens, adjacent name bigrams, the whole compact name (which matches domains and handles)
  - sorted 4-character prefixes of the name tokens (robust to typos and word order)
  - address words, address bigrams and digit runs. On the S1 side, house numbers also emit truncated variants, because the noise generator truncates them (1202 → 202 / 120).
  - compound keys: compact name × number, name token × number, number × address word, and name token × address word. These make generic names retrievable through their address.
  - Each key type has a document-frequency cap. Caps scale with the S1 partition size, so key selection depends on relative, not absolute, frequency, and train and test behave alike.
- **Selection:** the top 12 S1 candidates per query, then a relative cut that keeps candidates scoring at least 0.3 × the query's best.
- **Candidate pairs generated:** 30,008,138 test pairs (on average 17.3 candidate records per S1 entity).
- **How you ensured true matches were not lost:** we measured recall on the validation world at every change. Pair recall is 0.9855, and a perfect classifier on these candidates would score F0.5 = 0.9955. Of the remaining misses, 64% are records with an empty address, and most of the rest have a heavily changed or generic name shared by dozens of entities. We tested extra typo-robust keys (consonant skeleton, sorted-token compact name, 3-character prefixes); they would recover only 3% of the misses.

---

## 4. Matching Model

**Normalisation** (identical for every country): lowercasing, accent stripping, look-alike digit repair, legal-form canonicalisation, honorific and stop-word removal, DBA splitting, and domain and handle extraction. It also transliterates Indic script, first with a word dictionary learned from train pairs and then with a character fallback built from `unicodedata`. Addresses get street-type abbreviation groups, placeholder removal, and house-number, unit and PO-box parsing. Per-country address substitution maps (for example TX ↔ Texas, Bombay → Mumbai) are *learned from train pairs*. A country without training data simply gets no learned map.

**Features used (stage 1, 56):**
- Name features: RapidFuzz ratio, token-sort, token-set, partial and Jaro–Winkler similarity. Also Levenshtein distance, compact-name ratios, a "strict" core name without generic words, the DBA alternative name, char-3-gram and word TF-IDF cosine, exact and compact-exact flags, token counts, acronym match, legal-form agreement, and source flags (domain, handle, Indic, DBA, id-tag).
- Address features: RapidFuzz ratios, word TF-IDF cosine, and the same ratios on a "core" address with country-common tokens removed (this drops regions and departments symmetrically). House-number features: exact, truncation, Levenshtein distance, relative difference, missing. Also digit-run overlap and unit overlap.
- Other: how many S1 records share this name or address (ambiguity), and the source (S2 or S3). Raw blocking scores are *not* model features, because their scale shifts with index size.

**Stage 2 (28 context features)** are built from stage-1 probabilities of neighbouring pairs:
- *record side:* share, margin to the best competitor, second best, is-best flag
- *entity side:* probability mass, confident count, best competing record, rank, how many records choose this entity
- *sibling agreement:* how many other confident candidates of the entity share this record's house number, name or address, or carry the entity's own values. A deviation shared by siblings signals a systematic source format; a name deviation shared by siblings signals a look-alike decoy entity.

**Model type:** LightGBM binary classifiers (MIT), plus a fine-tuned cross-encoder, intfloat/multilingual-e5-small (MIT, 118M parameters). Both are far under the 8B limit. There are 4 folds grouped by query cluster, so all records of one entity share a fold. Each fold model trains on 67.5% of clusters. The final submission averages several model sets trained on different validation worlds and seeds.

**Threshold selection method:**
- Each record keeps only its best-scoring entity.
- For each entity, we choose the prefix k of its assigned records, sorted by probability, that maximises the exact expected F0.5 under independent Bernoulli truths.
- That expectation includes the singleton credit (k = 0 scores 1 only if the entity has no true match) and a Poisson term for true matches that blocking or the argmax lost.
- Probability odds are scaled for test's decoy density before selection.
- The rule variant (raw or calibrated, pooled or per country) is chosen on held-out entity halves of the out-of-fold predictions.

---

## 5. Results & Error Analysis

- **F_0.5 Score (macro), honest local validation** (held-out entity halves, out-of-fold, after the leak fix below):

  | Model | Local F0.5 |
  |---|---|
  | honest v4b, stage 2 | 0.98634 |
  | honest world-1 model | 0.98645 |
  | anchored world | 0.98785 |
  | v4b + cross-encoder blend | **0.98892** |

  Under a simulation of test's decoy density, the blend scores 0.98613 against 0.98207 for the trees alone. Public leaderboard: see the table below.
- **Common false positives (wrong merges):** near-twin decoys, meaning the same name with a house number one or two digits off (`c-115` vs `c-108`), or the same address with a different or generated name. Singletons attacked by look-alikes cost a full point each.
- **Common false negatives (missed matches):** true copies with an empty address whose name is generic (shared by 12 to 1,000 entities, so they are genuinely ambiguous), and true copies whose house number was perturbed by the generator.

**What moved the leaderboard** (public LB):

| Submission | Change | LB |
|---|---|---|
| r2 / v04 | legacy validation world (entities dropped *after* blocking) | ~0.978 |
| fix | corrected test-shaped world | 0.9778 |
| v4 | more training data per fold model (67.5% of clusters) | 0.97913 |
| v4 + v4b, odds × 0.5 | ensemble of two seeds and tree shapes + decoy-density correction | 0.979992 |
| same, odds × 0.35 | stronger correction | 0.980005 |
| v4bh | honest OOF (leak fixed), single model, odds × 0.35 | 0.979403 |
| ens5 | v4, v4b, v4bh, w1h, aw3; odds × 0.3; no cross-encoder | 0.98092 |
| ens5 + CE (run a) | + cross-encoder blend, odds × 0.35 | 0.983899 |
| ens5 + CE (runs a+b) | two cross-encoder runs averaged | 0.984006 |
| France × 0.2 | same, France odds 0.2 | 0.983818 |
| **France × 0.6 (final)** | same, France odds 0.6 | **0.984083** |

**Out-of-fold leak we found and fixed.** For a while, fold-k rows were scored by the average of the models j ≠ k. Model j trains on every fold except j, so those were exactly the models that had seen fold k, and model k was the only honest one. This inflated local scores (v4b: 0.99044 leaked vs about 0.988 honest). It also trained stage 2 on over-confident stage-1 scores, which is a mismatch with test, where stage-1 scores are honest. The fix scores fold k with model k at both stages. The stage-1 models themselves were unaffected, so we rebuilt stage 2 and all tuning on honest OOF without retraining stage 1.

---

## 6. Conclusion

Most of the achievable accuracy came from making validation look like test: the right index size, the right decoy mix, and honest out-of-fold scores. Tuning the classifier mattered much less. An exact expected-F0.5 selector and an explicit correction for test's heavier decoy load turn good probabilities into a precision-heavy match set. The main lessons: measure how test differs from train before tuning anything, and audit every out-of-fold path for leakage.

---

## Appendix

### A. Code Artefacts

`code/business_entity_resolution/src/` (entry point `run_final.sh`; see `README.md`):

| Step | Script |
|---|---|
| TSV → parquet, integer ids, true-pair table | `prepare.py` |
| learned Indic dictionary and address maps | `learn_maps.py` |
| normalisation | `normalize_all.py`, `textnorm.py` |
| blocking | `blocking.py` |
| pair features | `features.py` |
| stage-1 LightGBM, grouped folds, OOF | `train.py` |
| decision-rule tuning on OOF | `tune.py` |
| stage-2 context model | `stage2.py` |
| test prediction and decision | `predict.py`, `decide.py` |
| multi-model average, cross-encoder blend and final decision | `ensemble.py` |
| cross-encoder (GPU) and its blend | `ce.py`, `ce_blend.py` |
| submission zip in the required layout, validated | `make_submission_zip.py` (repo root) |
| submission writer | `output.py` |
| exact metric with the worked example as a self-test | `evaluate.py` |

### B. Additional Results

- **The final uploaded file** (public LB 0.984083) is `output/matching_results.tsv` in this archive. `src/run_final.sh` regenerates it.
- **Built but never uploaded (out of submissions):** a stronger cross-encoder trained on all ~690k training pairs per half. It scores 0.98932 held-out, against 0.98899 for the runs a+b blend used in the final file.
