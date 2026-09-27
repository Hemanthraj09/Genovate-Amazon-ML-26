# Genovate: Amazon ML Challenge 2026, project status

**Team:** Hemanth Raj, Kushal K V, Ayush Khanuja
**Status as of:** 27 Sep 2026, ~15:30 IST. Builds are running (§7). Code is committed and pushed to GitHub (§9).
**Deadline:** 27 Sep 2026, 23:59 IST. Final upload target: ~20:00 IST. Documentation and zip frozen by ~21:00.
**Uploads:** 5 per day; the leaderboard keeps each team's **maximum** score. Today: **3 used, 2 left**.
**Best public leaderboard score:** **0.980005** (v4 + v4b ensemble, odds × 0.35; `output_ens_o035/`). The top 60 teams are at ≥ 0.99.

Earlier write-ups: [feedback6.md](feedback6.md) (the root-cause analysis of the validation world). Candidates are archived in `submissions/`.

---

## 1. The problem in one paragraph

We get business records (name, address, country) from three sources that share no IDs. Source 1 (S1) is a clean, deduplicated reference list. For each S1 record, we must list every Source 2 and Source 3 record that describes the same business, which may be zero, one or many.

Scoring is **macro F0.5**, computed per S1 entity and then averaged; it weights precision twice as much as recall. A singleton (an S1 with no true match) scores 1.0 only if we predict an empty list. Test also contains **France** (15% of test S1), which never appears in train.

---

## 2. Rules we must follow

| Rule | Status |
|---|---|
| Output format: tab-separated, exact headers, one row per test S1, valid S2/S3 IDs, no duplicates | ✅ Enforced by the writer; every candidate passes `validate_submission.py --check-ids` |
| `candidate_pairs.tsv` = exactly what the model scores; matches ⊆ candidates | ✅ Both files are written from the same pair table |
| `country` is an open set (no hard-coding of US/India) | ✅ Country only partitions the work. The optional "unseen country" rule keys on "has no training labels", never on the name France |
| No external data, APIs, geocoding or registries | ✅ |
| Final model MIT/Apache-2.0, at most 8B parameters | ✅ LightGBM (MIT). *If* the cross-encoder is used: multilingual-e5-small (MIT, 118M parameters) |
| Pseudo-labeling (learning from our own test predictions) | ⏸ Not used; still waiting on the organizers |
| Final zip: `output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md` | ⏳ Documentation drafted (repo root); README, `run_final.sh` and the zip are still to do (§8) |
| Publishing code | No rule in `context/` forbids it (checked the guidelines PDF, the Unstop instructions and the deep-dive transcript). The team chose to push on 27 Sep; the commit history also documents authorship |
| Keep version history of all submissions | ✅ `submissions/SUBMISSIONS.md` lists every upload with its commit; the files are archived locally in `submissions/` (TSVs are git-ignored because of size) |

---

## 3. Key data facts

| | S1 | S2 | S3 |
|---|---|---|---|
| Train records | 2,206,821 | 5,034,616 | 5,285,603 |
| Test records | 1,732,544 (15% France) | 4,887,273 | 5,082,316 |

- **Test S1 is smaller than train's:** US 663K vs 1.32M (half), India 810K vs 883K, France 259K (test only).
- Each S2/S3 record matches **at most one** S1.
- **3.46 true matches per S1 in both train countries (a generator constant); 5.6% singletons.** The per-entity match-count distribution is identical in US and India.
- **Decoys are look-alikes of specific S1 entities:** the same name with a nearby house number, or the same address with a different or generated name. They cause most false positives.
- **Test's decoy structure (measured 27 Sep):**
  - DBA-style names appear only on true copies (2.4% of them, 0.00% of decoys). In test, 99.99% of DBA records get p > 0.98, so **test has no orphans**: every test record's entity is present.
  - Solving the mix from the empty-address, domain and DBA marker rates gives about 3.4 true copies plus **about 2.3 look-alike decoys per test entity**, against 1.2 per entity in train.
  - Pruned candidates per query in test (US 2.38, India 3.12) are exactly what true copies plus decoys *anchored on present entities* produce (2.45 / 3.19).
- **France:** S1 carries the region (Nouvelle-Aquitaine) where S2/S3 often carry the department (Gironde). Names are template compositions ("Nantes Sportive SAS"), so decoys differ from their entity by a single word or legal form.

---

## 4. The pipeline

All code is in `code/business_entity_resolution/src/`.

```
raw TSV ─prepare→ parquet ─learn_maps→ maps.json ─normalize→ normalized records
  ─blocking (per country, IDF-weighted keys, top-12)→ ─prune (≥0.3×best) + 56 features→
  ─stage 1 LightGBM (4 folds by query cluster)→ p1 ─stage 2 (+28 context features)→ p2
  ─[ensemble of model sets] → odds correction → best S1 per record → exact expected-F0.5 set per S1 → output/*.tsv
```

| Piece | File | Notes |
|---|---|---|
| Validation world | `config.py`, `blocking.py` | `BER_VARIANT=fix`: each country's S1 is cut to test's size **before** blocking; decoys are sampled to test's share. `BER_WORLD=N` draws a different sample. `BER_DECOYS=anchored` keeps only decoys that imitate kept entities (§6) |
| Blocking | `blocking.py` | name tokens/bigrams, compact name (`nc`, generous cap), 4-char prefixes, address words/bigrams/digits (with truncated variants), and cross keys. Caps scale with S1 size. Pair recall 0.9855, oracle F0.5 0.9955 |
| Features | `features.py` | 56 model features (blocking scores excluded); includes the core-address features `adk_*`. TF-IDF workers are capped by `BER_TFIDF_JOBS` (default 8) |
| Stage 1 | `train.py` | 4 folds, each model on 67.5% of clusters (`BER_TRAIN_FRAC=0.9`). **OOF: fold k is scored by model k only** (§6, the leak) |
| Stage 2 | `stage2.py` | record-side, entity-side and sibling-agreement context from stage-1 OOF |
| Decision | `decide.py`, `tune.py` | exact expected F0.5 (Poisson-binomial DP plus a missing-match term). `BER_ODDS` = odds multiplier (global or per country) |
| Ensembling | `ensemble.py`, `stack.py` | Averages stage-2 test probabilities over model sets (union of pairs) and applies odds. `BER_UNSEEN_S1_TAU` decides label-less countries from stage 1 |
| Cross-encoder (GPU, optional) | `ce.py`, `ce_blend.py` | multilingual-e5-small fine-tuned on uncertain pairs, with out-of-half honest scores and a blend tuned on held-out entities. Runs in a separate venv (`transformers`) |
| Build scripts | `run_fix*.sh`, `run_world.sh`, `run_anch.sh`, `run_honest.sh`, `run_s2var.sh`, `run_stack.sh` | One per build type. `run_final.sh` still to be written (§8) |

**Hardware:** 16 GB RAM, 24 threads, and an **RTX 4050 laptop GPU (6 GB)**, which we only discovered on 27 Sep. LightGBM has no GPU build here. RAM is the binding constraint: never run two heavy jobs at once (twice now we have hit WinError 1450 or a CUDA out-of-memory error that way).

---

## 5. Leaderboard history

| When | Upload | LB |
|---|---|---|
| 25 Sep | v01 baseline | 0.976553 |
| 25 Sep | v03 (legacy test-like world) | 0.976736 |
| 26 Sep | fix: corrected world (S1 cut before blocking) | 0.977783 |
| 26 Sep | France-empty probe: diagnostic only | 0.84307 |
| 26 Sep | v2 (compact-name key, core-address features) | 0.977854 |
| 26 Sep | v2 + v3 ensemble (two worlds) | 0.978677 |
| 26 Sep | v4 (67.5% of clusters per fold model) | 0.97913 |
| 27 Sep 12:15 | v4 + v4b ensemble, odds × 0.5 | 0.979992 |
| 27 Sep 12:19 | same, odds × 0.35 | **0.980005** |
| 27 Sep ~13:35 | honest v4b alone, odds × 0.35 | 0.979403 |

The France-empty probe splits the leaderboard into **US+India ≈ 0.982** and **France ≈ 0.955** (assuming France's singleton rate matches train's).

---

## 6. What we have learned (most important first)

1. **Out-of-fold leak (found 27 Sep, fixed).** `predict_oof` scored fold k with the average of models j ≠ k, which are exactly the models that had trained on fold k. Stage 2 did the same.
   - Every local score from `fix` through `v4b` was inflated: v4b stage 1 was 0.98805 leaked vs 0.98293 honest; stage 2 was 0.99044 leaked vs **0.98634 honest**.
   - Stage 2 was trained on over-confident stage-1 scores.
   - The fix scores fold k with model k. Honest rebuilds (no stage-1 retraining): `model_fix_v4bh` 0.98634, `model_fix_w1h` 0.98645.
   - On the leaderboard, one honest model (0.97940) did not beat the leaked two-model ensemble (0.98000).
2. **Test's candidate structure differs from our world's.** Pruned candidates per query:

   | | Our world (US / India) | Test (US / India) |
   |---|---|---|
   | All queries | 4.35 / 3.50 | **2.38 / 3.12** |
   | Orphans | 10.3 | — |
   | Look-alikes of removed entities | 11.2 | — |

   With honest inputs, stage 2 leans on these context features, so on test it is far less certain (3.6× the uncertain negative mass in the US). **The fix is the anchored world** (`BER_DECOYS=anchored`): kept entities, their true copies, and only the look-alikes of kept entities (26% decoys). It is being built now as `aw3`.
3. **Test has about 2× the look-alike decoys per entity**, so a pair scored p in training is less likely to be real on test. The fix is to multiply the odds (`BER_ODDS`): 0.5 gave +0.00086 together with the ensemble, and 0.35 gave another +0.000013. The honest simulation also prefers about 0.35.
4. **Ensembles of models from different worlds transfer best** (v2 + v3 gave +0.0008 LB). Single-model local gains transferred at only 40–56%, partly because of the leak.
5. **Blocking** costs 0.0045 of local F0.5, the largest single loss. But 97% of the misses are genuinely ambiguous: no address plus a generic or heavily changed name. Extra typo-robust keys would recover only 3% of them, so this is not worth a rebuild.
6. **Local loss breakdown (v4b, leaked OOF):** blocking 0.0045, true matches the model rejected 0.0021, false positives 0.0016, argmax losses 0.0013.
7. **France:** four normalization fixes (feature scale, stricter threshold, region/department core-address features, map dropout) returned about zero on the leaderboard. A stage-1-only rule for label-less countries is wired in (`BER_UNSEEN_S1_TAU`) but untested.
8. **No leaks in the data:** IDs and row order are uncorrelated with matches (|r| < 0.002).

---

## 7. Running now (27 Sep, ~15:30)

**Honest local results so far** (stage 2, expected-F rule, held-out entity halves):

| Model set | World | Local F0.5 |
|---|---|---|
| v4bh | original world, honest OOF | 0.98634 |
| w1h | world 1, honest OOF | 0.98645 |
| **aw3** | **anchored world 3** (only decoys that imitate kept entities) | **0.98785** |

Stage 1 alone for aw3 scored 0.98354. Each world has a different decoy mix, so the scores are indicative rather than directly comparable.

| Resource | Job | ETA |
|---|---|---|
| CPU | aw3: test prediction (`work/aw3_build.log`) | ~15:35 |
| GPU | cross-encoder on v4bh's uncertain pairs (`ce.py`). It starts automatically once aw3 finishes, because the two can't share 16 GB. Its first three attempts ran out of memory next to the CPU jobs | ~16:40 |
| deferred | honest v4 and the stage-2 variants. Paused with `work/SKIP_v4` and `work/SKIP_s2var`; delete a file to re-enable that step | if time allows |

Ready but not uploaded:

| Folder | Contents |
|---|---|
| `output_v4b_h/` | honest v4b, stage-2 rule only (no odds) |
| `output_w1_h/` | honest w1, stage-2 rule only |
| `output_ens_o1/` | leaked v4 + v4b, no odds |
| `output_v4b_o50/` | leaked v4b alone, × 0.5 |

## 8. Plan for the rest of the day

**Uploads (2 left):**

| # | What | When |
|---|---|---|
| 4 | aw3 (anchored world), alone or with v4bh/w1h, at the odds its honest simulation picks. Tests the structural fix | ~16:30 |
| 5 | Final: the broadest ensemble of the strong model sets (leaked v4/v4b, honest v4bh/w1h/v4h and variants, aw3 if it proves out), with the cross-encoder blend if it helps on held-out entities, at × 0.35 | ~19:30 |

**Deliverables (must finish by ~21:00):**

| # | Item | Status |
|---|---|---|
| 1 | `run_final.sh`: regenerates the final upload (shared prep, each member build with its exact settings, the ensemble) | Drafted; the member list is set once #5 is chosen |
| 2 | README rewrite: variant `fix`, 4 folds, 56 features, GPU optional, run time | To do (~17:00) |
| 3 | `Documentation_template.md` (repo root) | Drafted; fill in the final results |
| 4 | `requirements.txt`: add torch/transformers if the cross-encoder is used | To do |
| 5 | **`make_submission_zip.py <output folder>`** builds `Genovate_submission.zip` in the required layout (`output/`, `code/business_entity_resolution/{src,README.md,requirements.txt}`, `Documentation_template.md`) and runs the organizer's validator on the zipped files | **Ready and tested** on `output_ens_o035`: 43 files, 205.6 MB, validator PASS |
| 6 | Check the zip upload size limit on the portal (205.6 MB) | **Team: please check** |

**End-of-day steps (about 10 minutes once the final output exists):**
1. `python make_submission_zip.py <final output folder>` and confirm PASS.
2. Upload `<final output folder>/matching_results.tsv` to the leaderboard.
3. Submit `Genovate_submission.zip`.
4. Log the upload in `submissions/SUBMISSIONS.md`, then commit and push.

## 9. Git history of 27 Sep (pushed to `origin/main`)

| Commit | What |
|---|---|
| `1518d1f` | Ignore experiment output folders and TSVs |
| `1a66ced` | Test-shaped validation worlds (`fix`, `BER_WORLD`, `BER_DECOYS=anchored`), compact-name key, core-address features, build scripts |
| `6b42ea7` | **OOF leak fix** (fold k scored by model k), `BER_LEGACY_OOF` for regenerating v4/v4b, honest-rebuild and stage-2 variant scripts, `.sh` kept LF |
| `9853337` | Decision layer: missing-match term, `BER_ODDS`, ensembling (`ensemble.py`), stacking, unseen-country rule |
| `9798207` | Optional GPU cross-encoder (`ce.py`, `ce_blend.py`) |
| `8c2bd97` | Status, methodology draft, `feedback6.md`, `run_final.sh` draft |
| `3fb66e0` | Submission log for every upload through 13:35 |
| (this commit) | `make_submission_zip.py`, status update |

## 10. How to run

See [code/business_entity_resolution/README.md](code/business_entity_resolution/README.md) (being rewritten). Short form: `pip install -r requirements.txt`; `cd src`; `export BER_VARIANT=fix`; then run the builds and the final ensemble as in `run_final.sh` (to be added). Validate with the organizer's script using `--check-ids`.
