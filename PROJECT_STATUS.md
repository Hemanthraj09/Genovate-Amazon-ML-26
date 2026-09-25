# Genovate: Amazon ML Challenge 2026, project status

**Team:** Hemanth Raj, Kushal K V, Ayush Khanuja
**Status as of:** 26 Sep 2026, 01:05 IST. **Paused at the team's request; nothing is running.** Resume from §7.
**Deadline:** 27 Sep 2026, 23:59 IST. Target the final upload by about 20:00 IST on 27 Sep.
**Uploads:** 5 in total, **2 used, 3 left**.
**Best public leaderboard score:** 0.976736 (v03). The current #1 is 0.9859.
**Ready to upload (not yet uploaded):** v04, v05 and r2 (§5). Which one becomes upload #3 is still open (§7).

This document covers what the problem is, what we have built, how well it works, what we have learned, and what is left to do. The working plan is in [PLAN.md](PLAN.md), and every upload and candidate is logged in [submissions/SUBMISSIONS.md](submissions/SUBMISSIONS.md).

---

## 1. The problem in one paragraph

We get business records (name, address, country) from three sources that share no IDs. Source 1 (S1) is a clean, deduplicated reference list. For each S1 record, we must list every Source 2 and Source 3 record that describes the same business. There may be zero, one or many.

Scoring is **macro F0.5** computed per S1 entity and then averaged. It weights precision twice as much as recall. A singleton (an S1 with no true match) scores 1.0 only if we predict an empty list. Test also contains **France**, which never appears in train.

---

## 2. Rules we must follow (from `context/`)

| Rule | Status |
|---|---|
| Output is tab-separated with exact headers, one row per test S1 (empty list allowed), only S2/S3 IDs that exist in test, no duplicates | ✅ The writer enforces it, and every candidate passes `validate_submission.py --check-ids` |
| `candidate_pairs.tsv` is the exact set the model scores, and matches ⊆ candidates | ✅ Both files are written from the same pair table after pruning, and the writer asserts it |
| `country` is an open set: no hard-coding, filtering or one-hot of {US, India} | ✅ Country only splits the work into partitions. It is never a model feature. Learned maps are keyed by whatever labels exist, so an unseen label simply gets none |
| No external data, APIs, geocoding or registries | ✅ The only knowledge sources are hand-written abbreviation lists, maps learned from the provided train pairs, and unsupervised statistics computed on each split's own files. No pretrained models |
| Learning from our own predictions on test (pseudo-labeling) | ⏸ **Not used.** We have asked the organizers via the query form and are waiting for their answer |
| The final model is MIT/Apache-2.0 and has at most 8B parameters | ✅ LightGBM (MIT). pyarrow is Apache-2.0; the other libraries are MIT or BSD |
| Upload limit and version history | ✅ 5 uploads in total. Local git has one tag per version (`v01`–`v05`), and the files are archived in `submissions/` |
| Final zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md` | ⏳ Code, README and requirements exist. The zip and the filled template are still to do (§8-C) |
| Documentation: the guidelines ask for 1–2 pages, the problem statement says no page limit | ⏳ Plan: a 1–2 page summary at the top of `Documentation_template.md`, with full detail below it |
| Code with "proper comments describing the functions" | ⏳ Most functions have docstrings. A full pass over all 19 files is still to do |
| Don't publish code during the challenge | ✅ Nothing has been pushed. The GitHub repo is public, so we push after the deadline or once it is private |

---

## 3. What the data looks like (key EDA facts)

| | S1 | S2 | S3 |
|---|---|---|---|
| Train records | 2,206,821 | 5,034,616 | 5,285,603 |
| Test records | 1,732,544 (15% France) | 4,887,273 | 5,082,316 |

- **S1 size by country:**
  - US: 1.32M in train vs **663K in test**, half the size.
  - India: 883K in train vs 810K in test.
  - France: 259K in test only.
  - This size difference turned out to matter a lot (§6).
- Each S2/S3 record matches **at most one** S1. That lets us search from the S2/S3 side and assign each record to its single best S1.
- In train, 5.58% of S1 entities are singletons and the average is 3.46 matches per S1. The pair's country always agrees.
- **Decoys:** 27% of train S2/S3 records match nothing. In test we estimate about 41%, based on 2.85 records per S1 against 2.28 in train. This estimate assumes test has train's number of matches per entity (§8-A checks it).
- **Ambiguity:** 40–54% of S1 names are shared by other S1 entities, and 5–12% of addresses are shared. Neither field is enough on its own.
- **Noise the generator uses:**
  - Look-alike digits (0/o, 1/l, 5/s, 8/b, 6/g), accents, typos, repeated or shuffled words.
  - Legal-form swaps (Inc/Corp/LLC/Pvt Ltd/SARL...), junk prefixes (`***`, `>>`, `#`, `@`), titles (Mr, Dr, Smt, Shri).
  - Names given as domains or handles, "doing business as" / "formerly known as" names (the real name is always after the marker), invented names, acronyms.
  - Indian names written in Indian scripts (18% of Indian records).
  - Addresses: reordered components, truncated house numbers (607→60, 2007→007), added `H.no`/`#`/`No` prefixes, state codes vs full names vs local script, placeholders (N/A, null), PO box/PMB/unit, empty addresses (about 3%).
  - **Near-twin decoys:** the same name and street with a nearby house number, or a similar name at the same address.
- **France-specific noise** (found in the unlabeled test data):
  - Departments in place of regions (Gironde / Nouvelle-Aquitaine), `n°5` number prefixes, `st-` vs `saint-`, elisions (l', d').
  - The same avenue typos as the US.
  - Generator-added words: Groupe, Participations, Holding, Distribution, Développement, International, Et Fils, Et Associés. These look like translations of the US noise words (Group, & Sons, & Associates...).

---

## 4. What is built (the pipeline)

Everything is in `code/business_entity_resolution/src/` (19 Python files, about 2,240 lines). `run_all.py` runs everything end to end and defaults to the test-like variant. A full run takes about 1.7 hours on the laptop: 16 GB RAM, 24 threads, CPU only.

```
raw TSV ──prepare──> parquet ──learn_maps──> maps.json ──normalize──> normalized records
     ──blocking──> top-12 S1 per record ──prune + features──> pair features
     ──stage-1 LightGBM (2-fold, out-of-fold)──> p1
     ──stage-2 LightGBM (stage-1 features + context/sibling/core features from p1)──> p2
     ──decide (best S1 per record, then calibrated expected-F0.5 set per S1)──> output/*.tsv
```

| Step | File | What it does |
|---|---|---|
| Prepare | `prepare.py` | TSV → parquet with integer row IDs, plus a table of true pairs (9 s) |
| Learned maps | `learn_maps.py` | From train pairs only: an Indian-script→Latin word dictionary (1,347 words; 96.4% of test's Indian-script words covered), and per-country address substitutions (TX↔Texas, MH↔Maharashtra, Bombay→Mumbai, street-type typos). The street-type typo map is also applied to countries with no map of their own |
| Normalize | `textnorm.py`, `normalize_all.py` | Name: core tokens, strict core (generic words removed, French ones included), compact form, legal-form set, "doing business as" alternative, flags, elision. Address: tokens, digit runs, house number, unit/PO box, `n°` removal, splitting of purely alphabetic hyphenated words. The same rules run for every country (24M records in about 2 min) |
| Blocking | `blocking.py` | IDF-weighted shared keys: name tokens and bigrams, compact name, 4-character prefixes, address words and bigrams, house numbers plus truncated variants, name×number, name×word, number×word. Each key type has its own cap, **scaled with the country's S1 size (r2)**. Keeps the top 12 S1 per record. Train pair recall 98.44%, oracle F0.5 0.9952 |
| Prune + features | `features.py` | Keeps candidates scoring at least 0.3× the record's best, which leaves about 35.6M test-like train and 29.5M test pairs. Then computes 62 features: rapidfuzz ratios, TF-IDF cosines, house-number exact/truncation/edit distance/relative difference, number-set similarity, name edit distance, legal forms, acronym match, name/address sharing counts, blocking context |
| Stage 1 | `train.py` | LightGBM, 2 folds grouped by *query cluster* (all records of one entity stay in the same fold), up to 1200 rounds, producing out-of-fold probabilities. **Robust mode (v05, r2):** the 9 blocking-score features are left out of the model because their scale depends on dataset size, leaving 53 features |
| Stage 2 | `stage2.py` | Re-scores each pair with 28–29 extra features built from the out-of-fold stage-1 scores: (a) the record's share, margin and rank among its candidates; (b) the entity's summed score and confident records; (c) **sibling agreement**: how many other candidates of the same entity share this record's house number, name, number+name or address, and how many carry the entity's own number or name (robust mode counts only confident siblings); (d) **core-address similarity** after removing tokens found in >2% of the country's records |
| Decision | `decide.py`, `tune.py` | Each record keeps only its best S1. Then each entity's match set is chosen to **maximize exact expected F0.5** (Poisson-binomial dynamic programming, tested against brute force), after isotonic calibration. A global threshold τ is the fallback. The rule is chosen on out-of-fold predictions |
| Output | `output.py`, `predict.py` | Writes both TSVs and asserts every rule. The validator runs after each build |
| Metric | `evaluate.py` | The exact macro F0.5, with a self-test on the problem statement's worked example |
| Variants | `config.py` | `BER_VARIANT=tl` (test-like: 20% of train entities removed with a fixed hash, so about 41% of records are decoys), `BER_ROBUST`, `BER_MODEL_TAG` (separate model folders) |
| Dev and diagnostic tools | `crosseval.py`, `holdout.py`, `sizeshift.py`, `bench_block.py` | Cross-variant scoring, the honest-holdout check, the size-shift simulation (written, **not yet run**), and a blocking benchmark |

---

## 5. Results

"Local" means out-of-fold predictions on all train S1 entities (singletons included), scored with the exact metric. "Test-like" is the same data with 20% of entities removed, so the decoy share is about 41%.

| Version | What changed | Local (train mix) | Local (test-like) | Public leaderboard |
|---|---|---|---|---|
| v01 | Baseline: blocking + 55 features + stage 1 + τ=0.7 | 0.98229 | 0.98036 | **0.976553** |
| v02 | + stage 2 (basic context) + expected-F0.5 selection | 0.98537 | 0.98158 | not uploaded |
| v03 | + trained on the test-like variant | — | 0.98316 | **0.976736** |
| v04 | + France normalization, near-twin/acronym features, 1200 rounds; stage 2 + sibling agreement + core-address features | — | **0.98550** (US 0.98639, India 0.98416) | candidate |
| v05 | Robust: blocking-score features out of the model; confident-sibling counts | — | 0.98486 (US 0.98574, India 0.98355) | candidate |
| r2 | Robust + size-scaled blocking caps + hyphen/elision fixes, re-learned maps | — | 0.98464 (US 0.98540, India 0.98351) | candidate |
| (experiment) | Stage 3 (context rebuilt from stage-2 scores) | — | 0.98284 | rejected |

All three candidates pass the validator with `--check-ids`. They are archived in `submissions/v04/`, `submissions/v05_candidate/` and `submissions/r2_candidate/`.

### Confidence on test (share of records whose best candidate scores in the uncertain 0.1–0.9 band)

| Country | v03 | v04 | v05 | Train test-like (reference) |
|---|---|---|---|---|
| France | 10.3% | 8.4% | 7.4% | — |
| India | 5.8% | 4.3% | 4.0% | 3.8% |
| US | 5.3% | 5.1% | 4.6% | 3.2% |

A lower uncertain share means the model is more *confident* on test, not necessarily more *accurate*. It is supporting evidence only.

---

## 6. What we have learned

1. **Local gains did not reach the leaderboard.** From v01 to v03, the test-like local score rose by 0.0028, but the leaderboard only rose by 0.0002. The local-minus-LB gap widened from 0.0038 to 0.0065.
2. **Our validation is honest (no leakage).** In the honest-holdout check, half of the entities were never seen by any model, and stage 1 and stage 2 were run exactly as on test. On those entities, stage 2 scored 0.98494 against 0.98545 out-of-fold, and its gain was +0.0040 honest against +0.0036 out-of-fold.
3. **Test is distribution-shifted, mainly through blocking artefacts.** A model can tell test pairs from test-like train pairs with **AUC 0.99**.
   - The shift comes almost entirely from blocking-derived features: candidate count, blocking score, and the score gap to the runner-up (the model's #1 feature).
   - The cause is that blocking caps are absolute and IDF depends on N, while test's S1 is smaller (half the size in the US). So many more keys pass the caps on test, and every blocking statistic sits on a different scale.
   - Without these features the AUC drops to 0.71–0.73. What remains is mostly name-sharing counts, which reflect a real difference.
   - v05 removes these features from the model, and r2 also scales the caps with S1 size to fix the shift at its source.
4. **Sibling agreement is a strong signal.** Among uncertain pairs whose house number differs from S1's, only 24% are true matches when no other candidate of the same entity shares that number, against 71–81% when some do (a systematic source format). For names it reverses: a deviating name shared by several records is only 14% true, the mark of a look-alike decoy entity.
5. **Where the local loss comes from** (v04 validation, per entity; total 0.0145):
   - Misses-only entities: 59%.
   - Entities with false matches: 16%.
   - Model rejected everything: 11%.
   - Blocking lost every match: 7%.
   - Singleton false matches: 6%.
   - About a third of all loss is blocking misses, mostly no-address, generic-name records.
   - 1-match entities are the weakest group (mean F 0.952).
6. **Ruled out:** tie-breaking by row order (Spearman 0.001, ties split 50/50); dictionary coverage (97.8% train vs 96.4% test); differences in name/address sharing, empty addresses or noise flags; orphan vs natural decoy hardness *in train*.
7. **Rejected:** stage 3 (−0.0027); splitting *all* hyphens (it cost India recall, and r2 splits only purely alphabetic words).

---

## 7. Where we paused (resume here)

Nothing is running. The three candidates are ready. **The open decision is which one becomes upload #3.**

- Local scores favor **v04** (0.98550). **v05** (0.98486) and **r2** (0.98464) are within 0.0009 but are built to survive the train/test shift.
- That is an argument, not a measurement, so we don't choose on it.

**Resume order:**

1. **Quick checks (minutes; no uploads):**
   - **Test singleton rate:** estimate it from the model (sum each test entity's probability of having no match) and compare with train's. A false match on a singleton costs a full 1.0, and the test-like variant never creates extra singletons.
   - **French "doing business as" markers:** scan test France names for words such as "dit", "anciennement", "sous l'enseigne", "ex-" and "nom commercial". Our splitter only knows English markers.
   - **French titles and legal words:** check Mme, Mlle, M., Sté/Société, Cie, BP (as a PO box) and CEDEX against the French data before adding them. They should be common in S2/S3 and rare in S1.
   - **Per-country zero-candidate rates** and low-best-score rates on test (France's blocking health).
2. **`sizeshift.py` (about 1 hour).** It builds a train "mini world" at test's sizes (US S1 at 50%, India at 92%, 41% decoys), re-blocks it, and scores v04, v05 and r2 out-of-fold. The version that degrades least is the evidence-based upload #3.
3. **Leave-one-country-out** (train on US and score India, then the reverse; about 40 min). This is the best proxy for an unseen country. It tells us whether probabilities are over-confident there, which gives an evidence-based strictness setting for France (applied generically to any country without labels).
4. **One rebuild** bundling the confirmed French rules, fuzzy similarity to the entity's other confident records, and "S1 numbers covered by the record". That produces the candidate for upload #4.

**Questions for the organizers** (query form):
- Can we learn maps from confident test predictions? (asked; waiting)
- **Which submission counts on the private leaderboard: the best or the last?** (to ask) Until we know, our last upload must be our best real model.
- **Is there a size limit for the final zip?** (to ask) `candidate_pairs.tsv` is about 420 MB before compression.

---

## 8. What is left to build (prioritized)

### A. Close the local vs leaderboard gap
1. The quick checks, the size-shift simulation and leave-one-country-out in §7.
2. **Harder test-like validation:** if the checks show test's extra decoys are near-twins rather than orphans, rebuild the variant so it reproduces test's uncertain share per country.
3. **Map dropout:** train some rows with the learned maps switched off, because France gets none. Measure it with leave-one-country-out.
4. **Validation leak cleanup** (learn the maps on training folds only). This makes local scores more honest but won't move the leaderboard, so it is low priority.

### B. Model and feature work
5. **France normalization:** the checks in §7, plus departments vs regions learned by unsupervised co-occurrence in the test files (no labels, no external data).
6. **Fuzzy similarity inside the entity:** the candidate's best name and address similarity to the entity's other confident records. This handles landmark-only addresses and one-letter name variants.
7. **"S1 numbers covered by the record":** so an extra `H.no 50` prefix isn't read as a number conflict.
8. **Robustness:** full-data training instead of 60% samples, 2–3 seeds, 3 folds.
9. **Deprioritized:** more blocking recall (except for France, if the checks show a problem), embeddings (too slow on CPU for 24M records), more stacking, CatBoost/XGBoost ensembling.

### C. Deliverables (start 26 Sep, not 27 Sep)
10. `Documentation_template.md`: a 1–2 page summary at the top, then methodology, blocking, features, model, validation (including the honest-holdout and adversarial findings), results and error analysis.
11. Docstrings for every function in all 19 files; final pass on `README.md` and `requirements.txt`.
12. A packaging script that builds `Genovate_submission.zip` in the required layout, re-runs the validator and checks the zip.
13. **One clean end-to-end run of `run_all.py`** for the chosen final version, to prove `output/` can be reproduced from the code. The final upload must be a file the code generates, never a hand-mixed file.
14. Push to GitHub after the deadline (or earlier if the repo is made private).

---

## 9. Upload plan (3 left)

| Upload | What | When |
|---|---|---|
| #3 | The winner of the size-shift simulation among v04 / v05 / r2 | 26 Sep, after §7 steps 1–2 |
| #4 | The rebuild from §7 step 4, if it beats #3 on local, size-shift and leave-one-country-out checks | 26 Sep evening |
| #5 | Final: the best version, re-generated by a clean `run_all.py` run | 27 Sep, by about 20:00 IST |

We decide with local test-like score, size-shift robustness and leave-one-country-out together, never on the public leaderboard alone, because the final ranking uses the private split.

## 10. How to run

See [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md). In short: `pip install -r requirements.txt`, then `cd src && python run_all.py` (it defaults to the test-like variant). The outputs land in `output/`. Then run the organizer's validator with `--check-ids`.
