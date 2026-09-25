# Amazon ML Challenge 2026: Entity Resolution Plan (v2)

**Team:** Genovate (Hemanth Raj, Kushal K V, Ayush Khanuja)
**Deadline:** 27 Sep 2026, 23:59 IST. Aim to make the final upload by about 20:00 IST.
**Goal:** the best held-out macro F0.5 we can reach. The public leaderboard (currently #1 = 0.9859, on a subset of test) is a sanity check, not the target.

---

## A. Guideline checklist (check before every step and every upload)

Sources: `context/` (the problem statement, guidelines, Unstop instructions, and the video transcript) plus `student_resource/README.md`.

| # | Rule | How we enforce it |
|---|------|-------------------|
| 1 | Files are tab-separated with exact headers: `source1_entity_id<TAB>matched_entity_ids` and `source1_entity_id<TAB>candidate_entity_ids`. IDs are comma-joined with no quoting. | One shared writer function, UTF-8, `\t`. |
| 2 | Exactly one row per test S1 entity, France included. The list is empty when there is no match. | The writer starts from the full `test_source1` ID list. |
| 3 | IDs are S2-/S3- only and must exist in test. No duplicate IDs in a list and no duplicate rows. | Enforced by the writer, then `validate_submission.py --check-ids`. |
| 4 | Matches ⊆ candidates. `candidate_pairs.tsv` is the **exact** set the final model scores, not an earlier blocking pass. | Both files are written from the same pair table. |
| 5 | `country` is an open set. Never hard-code, filter, or one-hot `{US, India}`. | Country is used only to split blocking (whatever labels exist). It is never a model feature. |
| 6 | No external data, APIs, geocoding, business registries, or internet data augmentation. Only the provided data. | Normalization rules are hand-written abbreviation lists (as the problem statement suggests) or learned from the provided files. No downloaded gazetteers. No pseudo-labeling of test. |
| 7 | The final model is MIT/Apache-2.0 and has at most 8B parameters. | LightGBM (MIT). Any embedding model must have its license checked on the model card and recorded in the docs. |
| 8 | Local validation is a held-out train split scored with the exact macro F0.5, including singletons. | `evaluate.py` implements the formula exactly as the problem statement gives it. |
| 9 | At most 5 uploads per day. Keep a version history of every submission. | Local git with one tag per upload, plus `submissions/` holding each file, its config, local score and leaderboard score. |
| 10 | The final zip is `Genovate_submission.zip` containing `output/{matching_results,candidate_pairs}.tsv`, `code/business_entity_resolution/{src/,README.md,requirements.txt}` and a filled `Documentation_template.md`. | Built by a packaging script and re-validated. |
| 11 | Code runs end to end from the raw data using only that folder. Versions are pinned. Every function has a comment explaining what it does. | A single `run_all` entry point. Fixed random seeds. |
| 12 | The methodology document covers methodology, blocking, model and features, and other details (the guidelines ask for 1–2 pages). | Filled in from real experiment numbers. |
| 13 | Don't publish solution code during the challenge (plagiarism risk). | The GitHub repo is public, so **don't push until it is private or the deadline has passed.** |

---

## B. Data facts (these drive the design)

- Train S1/S2/S3 has 2.21M/5.03M/5.29M records. Test has 1.73M/4.89M/5.08M, and France is 15% of test S1 but absent from train.
- **5.58% of S1 records are singletons** (123,247). An all-empty submission would score 0.056. The average is 3.46 matches per S1 (maximum 11).
- Each S2/S3 record matches **at most one** S1. 27% of S2/S3 records are decoys. Country always agrees within a true pair.
- 40% of S1 names are shared by other S1 records, and 5.5% share an address. Test has 2.85 S2 records per S1 against 2.28 in train, which suggests more decoys.
- **Synthetic noise types:**
  - Typos and look-alike characters (0/o, 1/l, 5/s, 8/B) and added accents.
  - Repeated or shuffled words, and legal suffixes changed or moved.
  - Junk prefixes (`***`, `>>`, `--`, `...`, `<<`) and titles (Mr, Dr, Smt, Sri, Shri).
  - Names given as domains, handles or ID tags, "doing business as" / "formerly known as" names, and invented names.
  - Names written in Indian scripts (24% of Indian S2 names).
  - Addresses reordered, with the house number truncated (607→60, 2007→007/02007, 3317→317).
  - "St" wrongly expanded to "Saint".
  - State written in full, abbreviated, or in a local script.
  - Placeholder addresses (N/A, null, `<NULL>`) and empty addresses (3.4%).
  - Suffixes added (Services/Center/Co/Group).

---

## C. Pipeline

### 0. Setup
- Install polars, rapidfuzz and pyarrow, all MIT/Apache.
- Convert the TSVs to parquet with integer row indices.
- Process one country at a time to fit in 16 GB of RAM.

### 1. Normalization
- **Names:**
  - Lowercase, strip accents, and fix look-alike characters inside words.
  - Remove junk prefixes, brackets, honorifics and ID tags.
  - Split "doing business as" / "formerly known as" names and keep both parts.
  - Standardize legal forms (US, India and French: SARL/SAS/SASU/EURL/SCI/SA/SNC/EI) and keep them as a separate field.
  - Remove repeated words.
  - Keep a no-spaces version of the core name, to match domains and handles.
- **Indian scripts:** a word dictionary learned from train pairs, with a standard-library transliterator plus a consonant-only key as the fallback.
- **Addresses:**
  - Remove placeholder tokens.
  - Standardize street words for all three countries, applied to every record regardless of country. Treat st/street/saint as one token.
  - Extract the house number and its variants.
- **Statistics:** IDF weights and sharing counts are computed on **the dataset being processed**, so test statistics come from the test files. This is unsupervised and uses only provided data.

### 2. Blocking
- For each S2/S3 record, within its country, generate several keys:
  - (a) Rare core-name words.
  - (b) The no-spaces core name, for domains.
  - (c) Rare address words, and house number plus the next word.
  - (d) Pairs of address words.
- Score each S1 by the IDF-weighted sum of keys it shares with the record, and keep the top-K plus a relative cutoff below the best score.
- Optional: add the reverse direction (top-K records per S1) and an embedding kNN pass.
- **Gate:** the **oracle macro F0.5** (a perfect model limited to our candidates) is at least 0.997.
  - Also report pair recall, the number of entities that lose all their matches, and the average number of candidates.

### 3. Features, stage 1
- **Name:**
  - rapidfuzz scores: token set, token sort, full ratio, partial ratio, Jaro-Winkler.
  - IDF word overlap and Jaccard similarity.
  - No-spaces and domain similarity, acronym match, exact core-name match, legal-form agreement.
  - Flags for Indian script, domain or handle.
- **Address:**
  - House number: exact match, prefix/suffix (truncation), or conflict.
  - IDF word overlap, full-string scores, missing-address flags.
- **Ambiguity:**
  - How many S1 records share this core name.
  - How many S1 records share this address key.
  - Blocking score, rank among the record's candidates, and the gap to the next best.
- **Source:** whether the record came from S2 or S3.

### 4. Stage-2 features (from **out-of-fold** stage-1 probabilities)
- Stage 1 uses 5-fold cross-validation grouped by S1. Every train pair's stage-1 probability comes from a model that never saw that S1.
- **Record-side competition:**
  - p ÷ the sum of p over the record's candidates.
  - The margin to the best competing S1.
  - The second-best p.
  - Whether this pair is also the S1's best match for that record ("mutual best").
- **Entity-side:**
  - Rank among the S1's candidates.
  - The number and total p of the S1's confident candidates.
- **Corroboration:** the candidate's name and address similarity to the S1's other confident candidates (S2↔S3 and S2↔S2). This helps with invented or domain names, empty addresses and Indian-script names.

### 5. Deciding matches
- Assign each record to its single best S1 only. This is the optimal assignment given our constraint, so no Hungarian-style algorithm is needed.
- **Calibrate** probabilities with isotonic regression fitted on out-of-fold predictions.
- For each S1, choose the match set that maximizes **expected F0.5**:
  - Per entity, F0.5 = 1.25·TP / (k + 0.25·G).
  - Sort the candidates by probability and evaluate every prefix, including the empty set.
  - Compute the expectation **exactly** with Poisson-binomial distributions rather than Monte Carlo.
- MVP version: a single global threshold τ.

### 6. Validation
- 80/20 split by S1, with blocking over the full pool.
- Report the score for each segment: country, Indian-script names, domains/handles, empty address, "doing business as", and singletons.
- **Test-like variant:** drop about 20% of S1 records so their matches become decoys (S2-per-S1 ratio about 2.85). Weight it more heavily when tuning τ.
- **Leave-one-country-out** (train on India → validate on US, and the reverse). This stands in for France.
- Average over folds and 3–5 seeds.
- Log every run in `experiments.md`.

### 7. Error-analysis loop (most of day 2)
- Sample about 200 false positives and 200 false negatives.
- Tag each by noise type, fix the biggest group, and repeat.

### 8. Optional: learned similarity (day 2, only if Indian-script or domain errors dominate)
- multilingual-e5-small (MIT; check the model card) fine-tuned on train pairs with hard negatives on the GPU, using fp16 and sequences of 32–64 tokens.
- **Train on one half of the data and compute features for the other** to prevent leakage.
- Use it for cosine-similarity features and a kNN blocking pass.
- Run it as a separate script that writes `.npy` files, so it doesn't compete for RAM.
- Keep it only if the out-of-fold macro F0.5 improves.

### 9. Deliverables
- `run_all.py`, `README.md`, and a pinned `requirements.txt` noting the GPU if one is used.
- Fill in `Documentation_template.md`, build the zip, re-validate.

---

## D. Timeline and submission budget (5 per day; 0 used so far)

| When (IST) | Milestone | Uploads |
|---|---|---|
| 25 Sep night | v0: normalization, blocking, oracle gate, stage-1 LightGBM, argmax plus global τ. **Upload #1.** | 1–2 |
| 26 Sep morning | v1: out-of-fold stage 1, stage-2 features, calibration, expected-F0.5 selector | 1–2 |
| 26 Sep day | Error-analysis loop, leave-one-country-out, France review, optional embedding model | 1–2 |
| 27 Sep | Seed averaging, final τ, packaging, documentation. Final upload by about 20:00. | 1–3 |

Each upload tests one hypothesis. Only upload when the validation score improves.

## E. Feedback items not adopted, and why
- **Same-country flag:** blocking is already within country, and no true pair crosses countries, so the flag would be the same for every candidate.
- **Hungarian / global assignment:** the only constraint is that each record matches at most one S1, so the best assignment is simply each record's best S1.
- **Postcode/PIN blocking:** Indian addresses have no PINs, only about 10% of US addresses have a ZIP, and the French samples show none. It stays a minor feature only.
- **Monte Carlo subset selection:** replaced by an exact, deterministic calculation.
- **Pseudo-labeling test:** a grey area under "only the provided training data", so we won't do it.
- **Synthetic French training pairs:** deferred. French variants are standardized before features are computed. Revisit only if leave-one-country-out shows the model depends on country.
