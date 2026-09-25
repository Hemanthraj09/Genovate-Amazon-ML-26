# Genovate: Amazon ML Challenge 2026, project status

**Team:** Hemanth Raj, Kushal K V, Ayush Khanuja
**Status as of:** 25 Sep 2026, about 22:30 IST
**Deadline:** 27 Sep 2026, 23:59 IST. Target the final upload by about 20:00 IST on 27 Sep.
**Best public leaderboard score:** 0.976736 (v03). The current #1 is 0.9859.

This document covers what the problem is, what we have built, how well it works, what we have learned, and what is left to do. The detailed working plan is in [PLAN.md](PLAN.md), and every upload is logged in [submissions/SUBMISSIONS.md](submissions/SUBMISSIONS.md).

---

## 1. The problem in one paragraph

We get business records (name, address, country) from three sources that share no IDs. Source 1 (S1) is a clean, deduplicated reference list. For each S1 record, we must list every Source 2 and Source 3 record that describes the same business. There may be zero, one or many.

Scoring is **macro F0.5** computed per S1 entity and then averaged. It weights precision twice as much as recall. A singleton (an S1 with no true match) scores 1.0 only if we predict an empty list. Test also contains **France**, which never appears in train.

---

## 2. Rules we must follow (from `context/`)

| Rule | Status |
|---|---|
| Output is tab-separated with exact headers, one row per test S1 (empty list allowed), only S2/S3 IDs that exist in test, no duplicates | ✅ The writer enforces it, and every upload passes `validate_submission.py --check-ids` |
| `candidate_pairs.tsv` is the exact set the model scores, and matches ⊆ candidates | ✅ Both files are written from the same pair table, and the writer asserts it |
| `country` is an open set: no hard-coding, filtering or one-hot of {US, India} | ✅ Country only splits the work into partitions. It is never a model feature, and France goes through the same code |
| No external data, APIs, geocoding or registries | ✅ The only knowledge sources are hand-written abbreviation lists and maps learned from the provided train pairs. No pretrained models |
| The final model is MIT/Apache-2.0 and has at most 8B parameters | ✅ LightGBM (MIT). pyarrow is Apache-2.0; the other libraries are MIT or BSD |
| At most 5 uploads per day, with a version history kept | ✅ Local git with one tag per upload (`v01`–`v03`), plus the files archived in `submissions/vNN/` |
| Final zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md` | ⏳ Code, README and requirements exist. The zip and the filled template are still to do (§8) |
| Don't publish code during the challenge | ✅ Nothing has been pushed. The GitHub repo is public, so we push after the deadline or once it is private |

---

## 3. What the data looks like (key EDA facts)

| | S1 | S2 | S3 |
|---|---|---|---|
| Train records | 2,206,821 | 5,034,616 | 5,285,603 |
| Test records | 1,732,544 (15% France) | 4,887,273 | 5,082,316 |

- Each S2/S3 record matches **at most one** S1. That lets us search from the S2/S3 side and assign each record to its single best S1.
- In train, 5.58% of S1 entities are singletons and the average is 3.46 matches per S1. The pair's country always agrees.
- **Decoys:** 27% of train S2/S3 records match nothing. In test we estimate about 41%, based on 2.85 records per S1 against 2.28 in train.
- **Ambiguity:** 40–54% of S1 names are shared by other S1 entities, and 5–12% of addresses are shared. Neither field is enough on its own.
- **Noise the generator uses:**
  - Look-alike digits (0/o, 1/l, 5/s, 8/b, 6/g), accents, typos, repeated or shuffled words.
  - Legal-form swaps (Inc/Corp/LLC/Pvt Ltd/SARL...), junk prefixes (`***`, `>>`, `#`, `@`), titles (Mr, Dr, Smt, Shri).
  - Names given as domains or handles, "doing business as" / "formerly known as" names (the real name is always after the marker), invented names, acronyms.
  - Indian names written in Indian scripts (18% of Indian records).
  - Addresses: reordered components, truncated house numbers (607→60, 2007→007), added `H.no`/`#`/`No` prefixes, state codes vs full names vs local script, placeholders (N/A, null), PO box/PMB/unit, empty addresses (about 3%).
  - **Near-twin decoys:** the same name and street with a nearby house number, or a similar name at the same address.

---

## 4. What is built (the pipeline)

Everything is in `code/business_entity_resolution/src/` (17 Python files, about 1,900 lines). `run_all.py` runs everything end to end. The full run takes about 1.5 hours on the laptop: 16 GB RAM, 24 threads, CPU only.

```
raw TSV ──prepare──> parquet ──learn_maps──> maps.json ──normalize──> normalized records
     ──blocking──> top-12 S1 per record ──features──> 62 features per pair
     ──stage-1 LightGBM (2-fold, out-of-fold)──> p1 ──stage-2 LightGBM (+context from p1)──> p2
     ──decide (best S1 per record, then expected-F0.5 set per S1)──> output/*.tsv
```

| Step | File | What it does | Key numbers |
|---|---|---|---|
| Prepare | `prepare.py` | TSV → parquet with integer row IDs, plus a table of true pairs | 9 s |
| Learned maps | `learn_maps.py` | From train pairs only: an Indian-script→Latin word dictionary, and per-country address substitutions (TX↔Texas, MH↔Maharashtra, Bombay→Mumbai, street-type typos) | 1,347 Indian-script words (96% test coverage), 55+76 US and 29+6 India component/token maps |
| Normalize | `textnorm.py`, `normalize_all.py` | Name: core tokens, strict core, compact form, legal-form set, "doing business as" alternative, flags. Address: tokens, digit runs, house number, unit/PO box. The same rules run for every country, with French abbreviations included | 24M records in about 2 min |
| Blocking | `blocking.py` | IDF-weighted shared keys (name tokens and bigrams, compact name, 4-character prefixes, address words and bigrams, house numbers plus truncated variants, name×number, name×word, number×word), each type with its own cap. Keeps the top 12 S1 per record | Train: **pair recall 98.45%, oracle F0.5 0.9952**. About 30 min for train plus test |
| Pruning + features | `features.py` | Keeps candidates scoring at least 0.3× the record's best, which leaves 26.8M train and 31.0M test pairs with only 0.01 points of recall lost. Then computes the features: rapidfuzz ratios, TF-IDF cosines, house-number exact/truncation/edit distance, number sets, legal forms, acronyms, name/address sharing counts, blocking context | 55 features in v0, 62 in v1. About 3 min for train and 9 min for test |
| Stage 1 | `train.py` | LightGBM with 2 folds grouped by *query cluster* (all records of one entity stay in the same fold), producing out-of-fold probabilities | about 20–25 min |
| Stage 2 | `stage2.py` | Re-scores each pair using context from the out-of-fold stage-1 scores. Record side: its share, margin and rank among its candidates. Entity side: summed score, number of confident records, competing records | +0.003 local F0.5 |
| Decision | `decide.py`, `tune.py` | Each record keeps only its best S1. Then either a global threshold τ, or each entity's match set is chosen to **maximize exact expected F0.5** (Poisson-binomial dynamic programming, tested against brute force), with isotonic calibration | chosen automatically on out-of-fold predictions |
| Output | `output.py`, `predict.py` | Writes both TSVs and asserts every rule | runs the validator after each build |
| Metric | `evaluate.py` | The exact macro F0.5, with a self-test on the problem statement's worked example | |
| Test-like variant | `config.py` (`BER_VARIANT=tl`) | Drops 20% of train S1 entities so their records become decoys (about 41%, as in test). The model then learns test's decoy mix | |
| Dev tools | `crosseval.py`, `bench_block.py` | Score one training variant's models on another's data; benchmark blocking on a sample | |

Other files: `README.md` (how to reproduce), `requirements.txt` (pinned versions), `PLAN.md`, `submissions/SUBMISSIONS.md`.

---

## 5. Results so far

| Version | What changed | Local F0.5 (train mix) | Local F0.5 (test-like) | Public leaderboard |
|---|---|---|---|---|
| v01 | Baseline: blocking + 55 features + stage 1 + τ=0.7 | 0.98229 | 0.98036 | **0.976553** |
| v02 | + stage 2 + expected-F0.5 selection | 0.98537 | 0.98158 | not uploaded |
| v03 | + trained on the test-like variant, isotonic expected-F | — | **0.98316** | **0.976736** |
| v1 | + France normalization fixes, 7 near-twin/acronym features, 1200 rounds | running | running | — |

"Local" means out-of-fold predictions on all train S1 entities (singletons included), scored with the exact metric. "Test-like" is the same data with 20% of entities removed, so the decoy share matches test.

### Error breakdown (stage 2, train out-of-fold)

- **Precision is 99.7%** (22K wrong matches) and **recall 96.4%** (277K missed pairs).
- 81% of the wrong matches are near-twin decoys: the same name and street with a nearby house number, or a one-letter name change at the same address.
- Of the misses, 43% never reached the shortlist, 60% have no address (just a generic shared name), and the rest carry the same kind of number noise the decoys have.

---

## 6. What we have learned (important)

1. **Local improvements are not reaching the leaderboard.** From v01 to v03, the test-like local score rose by 0.0028, but the leaderboard only rose by **0.0002**. Our validation doesn't yet represent test well. This is now the main thing to understand before building more.
2. **Test is harder than train in ways we haven't reproduced.** We measured the share of records whose best candidate scores in the uncertain 0.1–0.9 band:

   | Data | Uncertain share |
   |---|---|
   | Train, test-like | 3.2–3.8% |
   | Test US/India | 5.3–5.8% |
   | Test France | 10.3% |

   We checked the obvious causes and ruled them out:
   - The rates of shared names and addresses, empty addresses, domains, "doing business as" names, Indian-script names and `H.no` prefixes are the same or lower in test.
   - Dictionary coverage of Indian-script words is similar (97.8% in train vs 96.4% in test).
   - Decoy type barely matters: records orphaned by removing an entity and naturally occurring decoys are about equally hard.
3. **France is about twice as uncertain as the other countries.**
   - French records use departments where S1 uses regions (Gironde vs Nouvelle-Aquitaine), `n°5` number prefixes, `st-` vs `saint-`, the same avenue typos as the US, and extra words (Groupe, Participations, Holding, Et Fils, Et Associés).
   - Some of these are fixed in v1. Others are genuinely ambiguous one-word swaps at the same address.
4. **Validation is somewhat optimistic by construction.** The learned maps and dictionary were fitted on all train pairs, which include the validation folds. The effect is small but real.

---

## 7. In progress right now

- The **v1 rebuild** is running in the background (`work/v1_log.txt`). It includes:
  - Re-learned maps and France normalization fixes (`n°`, hyphens, generic French words, a global street-typo map).
  - Re-blocking. Train recall is 98.40%, slightly down because splitting every hyphen broke some Indian keys. That is already fixed in code for the next rebuild.
  - 62 features and 1200-round training on the test-like variant, then stage 2, prediction and validation.
  - Expected to finish around 23:15 IST.

---

## 8. What is left to build (prioritized)

### A. Close the local vs leaderboard gap (highest priority)

1. **Probe France's contribution with one diagnostic upload.** Submit v03 with every French row set to empty.
   - The score change gives France's real per-entity F0.5: F_France ≈ (LB_v03 − LB_probe) / 0.15 + 0.06.
   - If France is far below US and India, all effort goes to France. If not, France isn't the problem.
2. **Probe the precision/recall balance.** One upload with a stricter decision, for example raising the minimum accepted probability.
   - If the leaderboard goes up, test has more false positives than our validation predicts, so we tighten.
   - If it goes down, we are losing recall.
3. **Remove the dictionary/map leak from validation.** Learn the maps on the training folds only, so local scores are honest.

### B. Model and feature work (after A tells us where the loss is)

4. **Next rebuild:** the hyphen fix (only purely alphabetic words are split), plus more French generic words (fils, assoc, frs) in the strict name.
5. **French address matching:** learn department ↔ region equivalence from the test files themselves (unsupervised co-occurrence with cities; no labels, no external data).
6. **Truncated or added house numbers:** add a "S1 numbers covered by the record" feature, so an extra `H.no 50` prefix isn't read as a number conflict.
7. **Blocking recall:** 43% of the misses never reach the shortlist. Try looser caps for compound keys and top-K by key type.
8. **Model robustness:** train on the full data instead of 60% samples, average several seeds, try more folds (3–5).
9. **Optional (day 2):** a small multilingual embedding model (multilingual-e5-small, MIT) as an extra similarity feature and blocking pass for Indian-script and domain names. Keep it only if the honest local score improves; use half/half training to avoid leakage.

### C. Deliverables (must do by 27 Sep)

10. Fill in `Documentation_template.md` with the final methodology, blocking, features, model, results and error analysis. It is copied into the zip root.
11. Final pass on `README.md` and `requirements.txt`, and check that every function has a comment describing it.
12. A packaging script that builds `Genovate_submission.zip` with the required structure, re-runs the validator, and checks the zip's layout.
13. Clean up the repo, and push to GitHub after the deadline (or earlier if it is made private).

---

## 9. Submission budget

- **Used:** 25 Sep: 2 (v01, v03), with 3 left today. 26 Sep: 5. 27 Sep: 5.
- Every upload tests one hypothesis. The next two proposed uploads are the diagnostic probes in §8-A. They are cheap, and they tell us where the remaining 0.009 points are.

## 10. How to run

See [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md). In short: `pip install -r requirements.txt`, then `cd src && python run_all.py` (it defaults to the test-like variant). The outputs land in `output/`. Then run the organizer's validator with `--check-ids`.
