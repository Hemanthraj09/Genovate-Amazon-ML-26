# Genovate: Amazon ML Challenge 2026, project status

**Team:** Hemanth Raj, Kushal K V, Ayush Khanuja
**Status as of:** 26 Sep 2026, about 00:30 IST (paused; see §7)
**Deadline:** 27 Sep 2026, 23:59 IST. Target the final upload by about 20:00 IST on 27 Sep.
**Best public leaderboard score:** 0.976736 (v03). **v04 is ready but not yet uploaded** (best local score so far). **Uploads: 2 of 5 used, 3 left.** The current #1 is 0.9859.

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
| `country` is an open set: no hard-coding, filtering or one-hot of {US, India} | ✅ Country only splits the work into partitions. It is never a model feature, and France goes through the same code. Learned maps are keyed by whatever labels exist, so an unseen label simply gets none |
| No external data, APIs, geocoding or registries | ✅ The only knowledge sources are hand-written abbreviation lists, maps learned from the provided train pairs, and unsupervised statistics computed on each split's own files. No pretrained models |
| The final model is MIT/Apache-2.0 and has at most 8B parameters | ✅ LightGBM (MIT). pyarrow is Apache-2.0; the other libraries are MIT or BSD |
| At most 5 uploads per day, with a version history kept | ✅ Local git with one tag per upload (`v01`–`v04`), plus the files archived in `submissions/vNN/` |
| Final zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md` | ⏳ Code, README and requirements exist. The zip and the filled template are still to do (§8-C) |
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
- **France-specific noise** (found in the unlabeled test data):
  - Departments in place of regions (Gironde / Nouvelle-Aquitaine), `n°5` number prefixes, `st-` vs `saint-`.
  - The same avenue typos as the US.
  - Generator-added words: Groupe, Participations, Holding, Distribution, Développement, International, Et Fils, Et Associés.

---

## 4. What is built (the pipeline)

Everything is in `code/business_entity_resolution/src/` (17 Python files, about 1,950 lines). `run_all.py` runs everything end to end and defaults to the test-like variant. The full run takes about 1.7 hours on the laptop: 16 GB RAM, 24 threads, CPU only.

```
raw TSV ──prepare──> parquet ──learn_maps──> maps.json ──normalize──> normalized records
     ──blocking──> top-12 S1 per record ──features──> 62 features per pair
     ──stage-1 LightGBM (2-fold, out-of-fold)──> p1
     ──stage-2 LightGBM (62 features + 29 context/sibling/core features from p1)──> p2
     ──decide (best S1 per record, then calibrated expected-F0.5 set per S1)──> output/*.tsv
```

| Step | File | What it does | Key numbers |
|---|---|---|---|
| Prepare | `prepare.py` | TSV → parquet with integer row IDs, plus a table of true pairs | 9 s |
| Learned maps | `learn_maps.py` | From train pairs only: an Indian-script→Latin word dictionary, and per-country address substitutions (TX↔Texas, MH↔Maharashtra, Bombay→Mumbai, street-type typos). The street-type typo map is also applied to countries with no map of their own (France) | 1,347 Indian-script words (97.8% train / 96.4% test coverage) |
| Normalize | `textnorm.py`, `normalize_all.py` | Name: core tokens, strict core (generic words removed, French ones included), compact form, legal-form set, "doing business as" alternative, flags. Address: tokens, digit runs, house number, unit/PO box, `n°` removal, hyphen splitting. The same rules run for every country | 24M records in about 2 min |
| Blocking | `blocking.py` | IDF-weighted shared keys (name tokens and bigrams, compact name, 4-character prefixes, address words and bigrams, house numbers plus truncated variants, name×number, name×word, number×word), each type with its own cap. Keeps the top 12 S1 per record | Train: **pair recall 98.40%, oracle F0.5 0.9951**. About 30 min for train plus test |
| Pruning + features | `features.py` | Keeps candidates scoring at least 0.3× the record's best, which leaves 35.7M test-like train and 31.6M test pairs. Then computes **62 features**: rapidfuzz ratios, TF-IDF cosines, house-number exact/truncation/edit distance/relative difference, number-set similarity, name edit distance, legal forms, acronym match, name/address sharing counts, blocking context | about 4 min for train and 3 min for test |
| Stage 1 | `train.py` | LightGBM, 2 folds grouped by *query cluster* (all records of one entity stay in the same fold), up to 1200 rounds, producing out-of-fold probabilities | about 20 min |
| Stage 2 | `stage2.py` | Re-scores each pair with **29 extra features** built from the out-of-fold stage-1 scores: (a) the record's share, margin and rank among its candidates; (b) the entity's summed score and confident records; (c) **sibling agreement**, i.e. how many other candidates of the same entity share this record's house number, name, number+name or full address, and how many carry the entity's own number or name (plain and weighted); (d) **core-address similarity**, fuzzy scores after removing tokens found in >2% of the country's records (regions, departments, big cities, street types) | about 12 min |
| Decision | `decide.py`, `tune.py` | Each record keeps only its best S1. Then each entity's match set is chosen to **maximize exact expected F0.5** (Poisson-binomial dynamic programming, tested against brute force), after isotonic calibration. A global threshold τ is the fallback. The best rule is chosen on out-of-fold predictions | |
| Output | `output.py`, `predict.py` | Writes both TSVs and asserts every rule | the validator runs after each build |
| Metric | `evaluate.py` | The exact macro F0.5, with a self-test on the problem statement's worked example | |
| Test-like variant | `config.py` (`BER_VARIANT=tl`) | Drops 20% of train S1 entities so their records become decoys (about 41%, as in test) | used for v03 onward |
| Dev tools | `crosseval.py`, `bench_block.py` | Score one training variant's models on another's data; benchmark blocking on a sample | |

Other files: `README.md` (how to reproduce), `requirements.txt` (pinned versions), `PLAN.md`, `submissions/SUBMISSIONS.md`.

---

## 5. Results so far

| Version | What changed | Local F0.5 (train mix) | Local F0.5 (test-like) | Public leaderboard |
|---|---|---|---|---|
| v01 | Baseline: blocking + 55 features + stage 1 + τ=0.7 | 0.98229 | 0.98036 | **0.976553** |
| v02 | + stage 2 (basic context) + expected-F0.5 selection | 0.98537 | 0.98158 | not uploaded |
| v03 | + trained on the test-like variant, isotonic expected-F | — | 0.98316 | **0.976736** |
| **v04** | + France normalization fixes, 7 near-twin/acronym features, 1200 rounds; stage 2 + **sibling agreement** + **core-address** features | — | **0.98550** (US 0.98639, India 0.98416) | not yet uploaded |
| **v05 (candidate)** | **Robust**: 9 blocking-score artefacts removed from the model, stage-2 sibling counts over confident siblings only | — | 0.98486 (US 0.98574, India 0.98355) | not yet uploaded, **recommended for upload #3** |
| (experiment) | stage 3: context rebuilt from stage-2 scores | — | 0.98284 | not adopted |

"Local" means out-of-fold predictions on all train S1 entities (singletons included), scored with the exact metric. "Test-like" is the same data with 20% of entities removed, so the decoy share matches test.

v04 gains, broken down on test-like validation:
- **Stage 1:** 0.98128 → 0.98224, from the new features and longer training.
- **Stage 2:** 0.98316 → 0.98550, of which the sibling and core-address features contribute about +0.0023.

### Error breakdown (v02 stage 2, train out-of-fold)

- **Precision is 99.7%** (22K wrong matches) and **recall 96.4%** (277K missed pairs).
- 81% of the wrong matches are near-twin decoys.
- Of the misses, 43% never reached the shortlist, 60% have no address (just a generic shared name), and the rest carry the same kind of number noise the decoys have.

### Confidence on test, as the share of records whose best candidate scores in the uncertain 0.1–0.9 band

| Data | v03 | v04 |
|---|---|---|
| Test France | 10.3% | **8.4%** |
| Test India | 5.8% | **4.3%** |
| Test US | 5.3% | **5.1%** |
| Train, test-like (reference) | 3.2–3.8% | — |

v04 is the first version whose improvement is visible *on test itself*, not only in validation.

---

## 6. What we have learned (important)

1. **Local improvements have not been reaching the leaderboard.** From v01 to v03, the test-like local score rose by 0.0028, but the leaderboard only rose by **0.0002**. We are checking two explanations:
   - *Validation optimism (leakage).* Stage 2 uses the out-of-fold stage-1 scores of *neighbouring* pairs, which come from other folds. We checked the main route: the models' minimum leaf size is 200 examples, while an entity has only 5–12 candidate pairs. So the model cannot memorize individual entities, which makes large leakage unlikely. Two smaller known sources remain: the learned maps and dictionary are fitted on all train pairs, and early stopping uses a slice of the fold being scored.
   - *The leaderboard is dominated by something our validation can't measure, most likely France.* France is 15% of test, has no labels, and is the most uncertain country. If it is weak, or over-represented in the public subset, US/India gains barely move the leaderboard. **v04's score is the first real test of this,** since v04 is the first version that visibly improves France on test.
2. **Sibling agreement is a strong missing signal.** Among uncertain pairs whose house number differs from S1's:

   | Other candidates of the same entity sharing that number | Share that are true matches |
   |---|---|
   | 0 | 24% |
   | 1 or more | 71–81% |

   So shared deviations are systematic source formats. For names the pattern reverses: a deviating name shared by several records falls to 14% true, the signature of a look-alike decoy entity.
3. **Test is harder than train in ways we haven't reproduced.** We checked the obvious causes and ruled them out:
   - The rates of shared names and addresses, empty addresses, domains, "doing business as" names, Indian-script names and `H.no` prefixes are the same or lower in test.
   - Dictionary coverage is similar.
   - Decoy type barely matters: records orphaned by removing an entity and naturally occurring decoys are about equally hard.
4. **France:** the fixes shipped in v04 reduced its uncertainty, but it is still about twice that of the other countries. The remaining French cases are mostly genuinely ambiguous one-word swaps at the same address ("Tourcoing Amicale Groupe" vs "Tourcoing Sport") and near-twin numbers.
5. **More stacking doesn't help.** Stage 3 lost 0.0027 locally.
6. **Blocking:** splitting *all* hyphens in addresses cost 0.0005 of India recall. The code is already fixed so that only purely alphabetic words are split (saint-nazaire, loire-atlantique) and IDs like B-425 stay intact. The fix takes effect in the next rebuild.

---

## 6b. Diagnostics after the third feedback round (26 Sep, about 00:00 IST)

| Check | Result | Meaning |
|---|---|---|
| **Honest holdout**: half of the entities never seen by any model; stage 1 and stage 2 run exactly as on test (averaged fold models, fold-averaged context) | Honest stage 2 scores 0.98494 vs ordinary out-of-fold 0.98545 on the same entities. The **stage-2 gain is +0.0040 honest vs +0.0036 out-of-fold** | No leakage: our validation is trustworthy *for train-like data*. |
| **Adversarial validation**: can a model tell test pairs from test-like train pairs? | **AUC 0.992** (US 0.996, India 0.986). 87% of it comes from `q_ncand_raw`, then `score`, `s_ncand` and `gap_second` | **This is the main problem.** Blocking caps are absolute counts and IDF depends on N. Test's S1 is half train's size (US 663K vs 1.32M), so far more keys pass the caps and every blocking-derived number is on a different scale. The model's #1 feature (`gap_second`) is one of them. |
| Adversarial validation without the 9 blocking artefacts | AUC **0.73** (US), **0.71** (India). The rest is mostly name-sharing counts | The remaining shift is largely real (test US has fewer same-name competitors), so those counts are kept. |
| Adversarial validation on the stage-2 context | AUC 0.95, from the candidate count per record and the raw sibling counts. Test France has 24.7 siblings sharing the entity's house number vs about 5 in the US, while *confident* candidates per entity are the same everywhere (3.4–3.6) | Count only confident siblings; drop the candidate count. |
| Row-order tie-breaking (§2.9 of the feedback) | Spearman(S1 row, S2 row) = 0.001; ties split 50/50 | Ruled out. |
| Entity-level loss (v04 validation, total 0.0145) | Misses-only entities 59%, entities with false matches 16%, model rejected all 11%, blocking lost all 7%, singleton false matches 6%. **About a third of all loss is blocking misses.** 1-match entities are the weakest group (mean F 0.952) | Recall, not precision, dominates the local loss. |

**Fix in progress ("robust" model set, `BER_MODEL_TAG=robust`):**
- Stage 1 no longer uses the 9 blocking-artefact features (`score, nk, gap_best, gap_second, q_ncand_raw, s_ncand, s_rank, rank, rel`). Blocking still uses them to build and prune candidates.
- Stage 2 drops the candidate count, counts only confident siblings (stage-1 p > 0.5), and caps rank at 8.
- **Pending decision on feedback item 8** (learning maps from confident test predictions): this is pseudo-labeling on test, so we have asked the organizers via the query form. It will not be used until they answer.
- **Upload budget confirmed: 5 in total, 2 used, 3 left.** No diagnostic probes; every upload must be a real candidate.

## 7. Where we paused (resume here)

**Paused at 26 Sep, about 00:30 IST, at the team's request.** No new jobs are to be started until we resume.

1. **The overnight rebuild `r2` has finished** (validator PASS; files archived in `submissions/r2_candidate/`, model folder `model_tl_r2`).
   - It adds size-scaled blocking caps, the hyphen fix and French elision, on top of the robust stack.
   - Blocking recall 0.98439 (oracle 0.99518). Local test-like F0.5: stage 1 0.98113, **stage 2 0.98464** (US 0.98540, India 0.98351).
   - Local scores: v04 0.98550, v05 0.98486, r2 0.98464, all within 0.0009. The size-shift simulation decides which of them survives test's sizes best.
   - **Nothing is running.**
2. **Open decision: which version is upload #3.**
   - Local numbers favor **v04** (0.98550 vs 0.98486).
   - The argument for **v05** is that its features are far less shifted (adversarial AUC about 0.85 vs 0.99). That is not yet a measurement.
3. **Resume order (from Feedback4 review):** (a) quick checks: the model-estimated singleton rate on test vs train, a scan of French 'doing business as' markers, French titles/legal words (Mme, Mlle, Sté, Cie, BP, CEDEX) confirmed against the data, per-country zero-candidate rates; (b) `sizeshift.py` for v04, v05 and r2; (c) leave-one-country-out; (d) one rebuild with the confirmed French rules, fuzzy intra-entity similarity and 'S1 numbers covered'. **Also ask the organizers:** which submission counts on the private leaderboard, and whether there is a zip size limit.

   **`sizeshift.py`** (written and committed, not yet run; about 1 hour).
   - It builds a train "mini world" at test's sizes (US S1 at 50%, India at 92%, 41% decoys), re-blocks it with the original caps, and scores v04 and v05 out-of-fold.
   - Whichever degrades less under the size shift is the evidence-based choice for upload #3.
   - Also score r2 the same way once it has finished.
4. **Uploads: 2 of 5 used, 3 left.** Nothing uploaded since v03 (0.976736).
5. **Organizers' answer on item 8** (learning maps from confident test predictions) is pending. Don't use it until they approve.

## 8. What is left to build (prioritized)

### A. Close the local vs leaderboard gap (decided by v04's score)

1. **If v04 moves the leaderboard clearly (+0.002 or more):** France and near-twins were the bottleneck. Continue with B.4–B.6, which target France and near-twin handling.
2. **If v04 barely moves:** the France probe (v04 with every French row set to empty) would diagnose it, but with only 3 uploads left it is **not affordable**. We rely on local analysis instead.
   - France's per-entity F0.5 ≈ (LB_v04 − LB_probe) / 0.15 + 0.06, accurate to about ±0.003.
   - If the probe collapses to about 0.06, the public subset is essentially France-only, which would change our whole strategy.
3. **Strictness probe:** dropped for the same reason (upload budget).
4. **Remove the small validation leaks:** learn the maps on the training folds only, and use a separate early-stopping slice.

### B. Model and feature work

5. **Next rebuild:** the hyphen fix (already coded), plus more French generic words (fils, assoc, frs) in the strict name.
6. **France:** departments vs regions, learned from the test files themselves by unsupervised co-occurrence with cities (no labels, no external data). More sibling-style consistency features for same-address one-word swaps.
7. **Added or truncated house numbers:** a "S1 numbers covered by the record" feature, so an extra `H.no 50` prefix isn't read as a number conflict.
8. **Blocking recall:** 43% of the misses never reach the shortlist. Try looser caps for compound keys and top-K by key type.
9. **Model robustness:** full-data training instead of 60% samples, several seeds, more folds (3–5).
10. **Optional:** a small multilingual embedding model (multilingual-e5-small, MIT) as an extra similarity feature and blocking pass. Keep it only if the honest local score improves.

### C. Deliverables (must do by 27 Sep)

11. Fill in `Documentation_template.md` with the final methodology, blocking, features, model, results and error analysis. It is copied into the zip root.
12. Final pass on `README.md` and `requirements.txt`, and check that every function has a comment describing it.
13. A packaging script that builds `Genovate_submission.zip` with the required structure, re-runs the validator, and checks the zip's layout.
14. Clean up the repo, and push to GitHub after the deadline (or earlier if it is made private).

---

## 9. Submission budget

- **5 uploads in total. 2 used (v01, v03), 3 left.** v04 has not been uploaded yet.
- With only 3 left, diagnostic probes are too expensive: every remaining upload should be a real candidate for the final score. So the France probe (§8-A.2) is dropped, and we decide based on local validation and v04's result.
- **Suggested use:** upload v04 now; use one upload for the best improvement tomorrow; keep the last one for the final version on 27 Sep.

## 10. How to run

See [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md). In short: `pip install -r requirements.txt`, then `cd src && python run_all.py` (it defaults to the test-like variant). The outputs land in `output/`. Then run the organizer's validator with `--check-ids`.
