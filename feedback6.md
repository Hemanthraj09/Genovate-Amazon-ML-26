# Feedback 6 — The local/leaderboard gap is a pipeline bug, not distribution shift

**Amazon ML Challenge 2026 — Business Entity Resolution (team Genovate)**
Written after reading the pipeline source and measuring the candidate sets in `work/`.
Companion to [PROJECT_STATUS.md](PROJECT_STATUS.md). Supersedes the priority order in Feedback 5.

---

## 0. Summary

We have been trying to explain a persistent ~0.008 gap between our test-like local score
(0.98464 for r2) and the leaderboard (0.976736 for v03). PROJECT_STATUS.md §6 attributes it to
an intrinsic train/test distribution shift caused by test's smaller S1 partitions.

**That diagnosis is wrong.** The raw blocking output on train and test is nearly identical.
The shift is created by our own code, one step later: the test-like variant removes the dropped
S1 entities *after* blocking instead of *before*, which strips ~20% of every query's candidate
list. We then train on 9.47 candidates per query and predict on 11.71.

Three findings, all measured on the current `work/` artefacts:

| # | Finding | Evidence |
|---|---|---|
| 1 | The test-like variant is a broken simulation of test | train_tl **9.47** cand/query, only **6.7%** of queries full; test **11.71** cand/query, **95.2%** full |
| 2 | Blocking silently fails on address-less records | recall **99.49%** with an address vs **75.65%** without — ~82K true pairs permanently lost |
| 3 | Stage 2 is fed a different stage-1 score distribution at train and test time | OOF `p` mean 0.169 / frac>0.5 = 0.168; test `p` mean 0.202 / frac>0.5 = 0.200 |

Finding 1 explains why our local gains stopped transferring, why `q_ncand_raw` and `q_n` are the
top train/test discriminators, why the "robust" v05/r2 only partly helped, and why the honest-holdout
check found nothing. Finding 2 is the largest single recoverable accuracy loss. Finding 3 is a
classic stacking defect that costs real leaderboard points.

**Recommendation: do not add model machinery. Fix the four defects below and re-run.** Every one of
them is a small, local code change.

---

## 1. Verdict on Feedback 5

Feedback 5 is a competent, well-organised general ER improvement checklist, and a handful of its
sections are pointed in exactly the right direction. But it was written without reading the code,
so it diagnoses the symptom as a property of the dataset and proposes a re-architecture, when the
dominant cause is four bugs in files we own.

**What Feedback 5 gets right and we should keep:**

- **§22 / §W "prefer distribution-stable features."** Correct, and it goes further than we did:
  `C.BLOCKING_ARTEFACTS` only covers stage-1 columns, so `BER_ROBUST` never protected stage 2.
  Stage 2 still carries `q_psum`, `s_psum`, `s_nconf`, `s_nbest`, `s_psum_best`, `sib_*_same`,
  `sib_*_eq_s` — all pool-size dependent. This is confirmed in `work/diag_adversarial_robust.txt`:
  the robust stage-2 design still separates train from test at AUC 0.88 (US) / 0.83 (India).
- **§23 / §X "verify stage-2 input distributions."** This is finding 3. Feedback 5 was one
  measurement away from the bug and never took it.
- **§D adaptive K and the no-address emphasis.** Correct, and finding 2 gives it a hard number.
- **§E audit the `0.3 × best` pruning rule.** Worth one cheap experiment.
- **§S/§T hard-negative mining.** Cheap, because `work/error_analysis_v02.txt` already holds the
  false positives; reusing them costs one training run, not a redesign.

**What to reject for this deadline:**

- Its **P0 list** (blocking recall sweep at K=12/20/30/50, France diagnostics, size-shift,
  leave-one-country-out, adaptive K, two-sided retrieval — six diagnostics before any fix) would
  consume the remaining ~30 hours producing numbers, while the broken variant keeps invalidating
  every number produced. **Fix the measurement instrument before taking more measurements.**
- **§7–§9, §K, §L, §B: null model, ranking reformulation, match-count model, singleton model.**
  Each is a multi-hour rebuild with an uncertain payoff, and each would need revalidation against
  a validation set we now know is broken. With 3 uploads left, added model complexity is the wrong
  risk. Revisit after the deadline.
- **§14–§17, §G–§J: entity profiles, similarity graphs, soft reassignment.** Genuinely promising
  ideas, but they are next-competition work, not next-24-hours work. Our `sib_*` features already
  capture the cheap 80% of this.
- **Several items are already built**: sibling agreement (`stage2.sibling_features`), expected-F0.5
  selection (`decide.by_expected_f`), IDF-weighted rare-token retrieval (`blocking.block_country`),
  acronym features (`features.py:187-191`), house-number truncation/substring/edit-distance/reldiff
  (`features.py:127-186`). Feedback 5 lists these as new work in §26, §28, §30.
- **The France weighting is too high.** France is 15% of test, but its uncertain share is 7.4%
  against 4.6% for the US — worse, not catastrophic. It deserves the cheap checks in
  PROJECT_STATUS.md §7 step 1 and per-country calibration, not a re-architecture.

**Where Feedback 5's core thesis is wrong:** it says "the remaining problem is not simply model
capacity, it is primarily distribution shift." Half right. The shift is real but mostly
*self-inflicted*, and there is also a large uncaptured capacity loss: `TRAIN_FRAC = 0.6` with
`NFOLD = 2` means each fold model is fitted on only ~30% of clusters (see §3.4).

---

## 2. Finding 1 (root cause): the test-like variant drops S1 entities after blocking

### What the code does

`blocking.run()` blocks against the **full** train S1 index (2.21M entities) — it never consults
`dropped_s1()`. The drop happens afterwards, in `features.build()`:

```python
# code/business_entity_resolution/src/features.py:200-205
drop = B.dropped_s1() if split == "train" else None
if drop is not None and len(drop):
    cand = (cand.filter(~pl.col("s1").is_in(drop))      # <-- 20% of candidates deleted
                .sort(["qid", "score"], descending=[False, True])
                .with_columns(pl.int_range(1, pl.len() + 1).over("qid")...alias("rank")))
```

Blocking already truncated each query to `TOPK = 12`. Deleting 20% of S1 entities from an
already-truncated list removes ~2.4 candidates per query **and nothing replenishes them**. In a
genuine blocking run against a 20%-smaller index, the vacated slots would be refilled from ranks
13, 14, 15… up to a full 12.

### The measurement

```
train, raw blocking output        : 11.83 cand/query   97.4% of queries at the full 12   mean best score 208.2
train_tl, after the post-hoc drop :  9.47 cand/query    6.7% of queries at the full 12   mean best score 174.8
test                              : 11.71 cand/query   95.2% of queries at the full 12   mean best score 193.4
```

Raw train and test agree to within 1% — **r2's size-scaled caps already solved the cap/IDF problem.**
The entire remaining shift is manufactured by the post-hoc drop. We train on a world where 93% of
queries have a short candidate list and predict on a world where 95% have a full one.

### Why this explains everything we could not explain

- **`q_ncand_raw` is the #1 train/test discriminator** (gain 0.769 US / 0.899 India, AUC 0.99 in
  `work/diag_adversarial2.txt`). It is literally the candidate count we corrupted.
- **`q_n` is the #1 stage-2 discriminator** (0.568 / 0.694, AUC 0.948). Same cause.
- **`sib_hn_same` 2.175 train vs 1.588 test, `sib_hn_eq_s` 5.543 vs 4.165** (~27% lower on test).
  Sibling counts are counts over the entity's candidate pool. A 20% thinner pool on the train side,
  compounded with the `p > 0.5` threshold shift from finding 3, gives exactly this.
- **Why v05/r2 only partly helped.** Removing the stage-1 blocking artefacts from the *model* hid the
  symptom without removing the cause, and left every pool-size-dependent stage-2 feature in place.
- **Why the honest-holdout check found nothing.** `holdout.py` splits entities *within the same
  broken variant*. Both halves get 9.47 candidates per query. A holdout cannot detect a bias that is
  identical on both sides of the split. Our conclusion "our validation is honest (no leakage)" is
  true as far as it goes — it tested for model-fitting leakage, not construction bias — and it gave
  us false confidence in a number that was measuring the wrong world.
- **Why v01→v03 gained +0.0028 locally and +0.0002 on the leaderboard.** Those gains came from
  stage-2 context and the tl variant: precisely the components most sensitive to pool composition.
  We were tuning against an artefact.

### The fix

`sizeshift.py` already contains the correct construction. `build_world()` (lines 36-58) drops
entities from the S1 frame **before** `block_country()`, sizes each country to test's actual S1 count,
and subsamples decoy queries to the 41% target. That function should not be a diagnostic — it should
*be* the training variant.

Concretely:

1. Move the drop into `blocking.run()`: for `split == "train"`, filter `s1` by `~dropped_s1()` before
   the per-country loop. Delete the post-hoc filter at `features.py:200-205`.
2. Make the drop **per-country and size-matched**, reusing `sizeshift.SIZE`
   (`US: 663106/1323633 = 0.501`, `India: 809986/883188 = 0.917`). Dropping a flat 20% leaves US at
   1.06M against test's 663K; this is the residual size gap that r2's cap scaling had to paper over.
3. Size-matching overshoots the decoy share (dropping ~50% of US entities makes ~63% of US records
   decoys, against a 41% target), so **decouple the two**: after size-matching the S1 side,
   subsample decoy queries to the 41% share, exactly as `sizeshift.py:55` does.
4. Re-run `blocking → features → train → tune → stage2 → tune2` (~2h).

**Expect the local score to fall.** That is the point: it should land near 0.977–0.980 and finally
*mean* something. The leaderboard score should rise, because every learned threshold on
`q_psum`, `s_nconf`, `s_nbest` and `sib_*` will be calibrated to test's actual pool density.

> Once the variant is honest, `CAP_REF_N` scaling becomes a genuinely testable choice rather than a
> compensation for this bug. Keep it for now; re-measure afterwards.

---

## 3. Other confirmed defects

### 3.1 Blocking loses a quarter of all address-less true pairs

```
true pairs with a query address   : 7,301,347   blocking recall 99.49%
true pairs with an EMPTY address  :   337,018   blocking recall 75.65%   <-- ~82,000 pairs lost
```

`work/error_analysis_v02.txt` reports 118,954 false negatives never present in candidates.
Address-less records are ~4.4% of true pairs but account for **~69% of all irrecoverable blocking
misses**, and 60.1% of all false negatives against a 3.3% base rate.

**Mechanism** (`blocking.build_keys`): a record with an empty address emits *only* name keys —
`n:<token>`, `n:<nm_cmp>`, `nb:<bigram>`, `p4:<prefix-set>`. Every address key (`a:`, `ab:`, `#:`)
and every compound key (`nh:`, `th:`, `hw:`, `tw:`) requires an address token or digit run, so all
of them produce nothing. And `n:` is key type 0, which draws the **tightest cap in the system**:

```python
# blocking.py:107
c_single = max(10, round(CAP_SINGLE * scale))     # CAP_SINGLE = 60
```

For test US (`scale = 0.663`) that is 40. A generic-named, address-less record whose name tokens and
compact name each appear in more than 40 S1 records of that country loses **every key it has** and
gets zero candidates. Those matches are unreachable no matter how good the model is.

**Fix (cheap, targeted, no retrain of the main path).** Add a second blocking pass restricted to
queries that emerge from pass 1 with zero or very few candidates — a small subset, so cost is low:

- relax the type-0 cap for this pass only (e.g. `c_single * 5`), and
- keep the selective `p4:` prefix-set and `n:<nm_cmp>` keys, which stay discriminative even for
  generic names, and
- let these queries use a larger `TOPK` (Feedback 5 §D's adaptive K, but applied where we have
  measured that it pays).

Then add a `no_address` indicator feature so the model can learn that name evidence carries the full
weight when there is no address to corroborate it. Verify the recall gain on the same split above
before rebuilding features.

### 3.2 Stage 2 is trained and tested on different stage-1 score distributions

```
OOF   p (single fold model) : mean 0.1691  std 0.3696  frac>0.5 = 0.1679
TEST  p (average of 2)      : mean 0.2017  std 0.3938  frac>0.5 = 0.1999
```

- Stage-2 **training** context comes from `oof.parquet`, where each pair's `p` is produced by the
  *one* fold model that did not see it (`train.predict_oof`).
- Stage-2 **test** context comes from `test_pred.parquet`, which is `np.mean` over *both* fold models
  (`train.predict_split`, line 116).

`p` is `S2_EXTRA[0]`, and all 29 stage-2 features are functions of it (`q_pmax`, `q_p2`, `q_psum`,
`q_share`, `q_margin`, `s_psum`, `s_nconf`, `s_pmax_other`, `sib_*_p`, …). So the entire stage-2
feature block is computed from a systematically different input at inference time. A 2-model average
is sharper and better calibrated than a single model, so test `p` is not just shifted, it is
*qualitatively better* — which is precisely the situation stage 2 was never trained for.

This is made worse by a hard threshold:

```python
# stage2.py:89
x = x.with_columns((pl.col("p") > 0.5).cast(pl.Int32).alias("_conf"))
```

`frac(p > 0.5)` is 19% higher on test (0.200 vs 0.168), so every `sib_*_same` and `sib_*_eq_s` count
is inflated on test relative to training. The v05 "robust" change that switched these to
*confident*-sibling counts traded a blocking-count dependence for a `p`-distribution dependence —
and the `p` distribution is broken, so it was a bad trade until this is fixed.

**Fix, in order of preference:**

1. **`NFOLD = 4` or `5`, `TRAIN_FRAC = 1.0`.** Build the train-side context by averaging the `K-1`
   models that did not see the pair; test averages all `K`. That reduces the ensemble-size mismatch
   from 1-vs-2 (a 100% difference) to 4-vs-5 (25%), and simultaneously fixes §3.4. This is the right
   fix and it is a few lines.
2. If compute is too tight for (1): per-country **quantile-map** test `p` onto the OOF `p`
   distribution before `context()` is called. Cheap, no retraining, removes most of the mismatch.
3. Either way, drop the hard `p > 0.5` threshold in favour of the `_p`-weighted variants we already
   compute, or a within-query rank.

### 3.3 The local score is optimistic for three independent reasons

**(a) Early stopping validates on the fold being scored.**

```python
# train.py:74-75
tr = lf.filter((pl.col("fold") != k) & (pl.col("u") < TRAIN_FRAC)).collect()
va = lf.filter((pl.col("fold") == k) & (pl.col("u") < 0.05)).collect()   # <-- fold k = the OOF fold
```

`best_iteration` for the model that generates fold *k*'s out-of-fold predictions is chosen by early
stopping on fold *k* data. Same defect at `stage2.py:161` and `holdout.py:53` — so the "honest"
holdout inherits it. Use an inner split carved from the *training* folds instead.

**(b) The isotonic calibrator is fit in-sample, and the comment says otherwise.**

```python
# tune.py:61-63
# calibrated expected-F selection (calibrator fit on half, evaluated on other half by cluster)
iso = fit_calibrator(assigned)                                      # fit on ALL assigned
sc_ef = report(D.by_expected_f(calibrate(assigned, iso)), ...)      # scored on ALL assigned
```

The documented half/half split was never implemented. `dec["rule"]` (tune.py:66) and `tau`
(tune.py:57-60) are also selected on the same rows they are scored on. Fit the calibrator on one
cluster-half and report on the other.

**(c) Learned maps are fitted on all train pairs**, including the OOF folds
(acknowledged in `holdout.py:12-13`). Low priority for the leaderboard, but it means our local
number is optimistic in a way test can never be — and France gets **no maps at all**, so the model
has learned to rely on map-normalised address agreement that 15% of test cannot produce.
**Map dropout** (train a fraction of rows with maps disabled) is the cheap mitigation and is already
in PROJECT_STATUS.md §8-A3.

Fixing (a) and (b) will *lower* the local score. Do it anyway — the whole problem is that we cannot
currently tell a real gain from an artefact.

### 3.4 We throw away 70% of the training data

```python
# train.py:22-23
NFOLD = 2
TRAIN_FRAC = 0.6           # share of each training fold's clusters used for fitting
```

Each fold model fits on 60% of the *other* fold — i.e. ~30% of all clusters (~10.7M of 35.6M pairs,
confirmed in `work/r2_log.txt`: "stage2 fold 0: train 10,670,533"). With `NFOLD = 5` and
`TRAIN_FRAC = 1.0`, each model would see 80%. That is a straightforward capacity gain on top of
fixing §3.2, and it is the single change that buys the most accuracy per line edited. Budget the
compute: it is the most valuable place to spend it.

### 3.5 The expected-F0.5 optimiser is systematically too conservative

`decide.by_expected_f` maximises

```
E_k = Σ_t Σ_o P(TP=t) · P(O=o) · 1.25t / (k + 0.25(t + o))
```

where `t` is the true count among the top-`k` selected and `o` among the remaining *assigned*
candidates. But the metric's `G` is the entity's **total** true matches, including the ones that
never reached the candidate set or were argmax-assigned to a different S1. From
`work/error_analysis_v02.txt`: 118,954 true pairs are absent from candidates and 59,058 went to
another S1 — ~178K true matches the DP cannot see.

Underestimating `G` shrinks the denominator, which makes each extra prediction look *more* expensive
than it is, and it inflates `E_0 = P(no candidate is true)` so empty predictions look *safer* than
they are. Both biases push toward under-prediction — and misses-only entities are 59% of our loss,
with "model rejected everything" another 11% (PROJECT_STATUS.md §6.5).

**Fix:** add a per-entity expected-missing-matches term `m` to the `G` used inside the DP, and fold
`P(m > 0)` into the `k = 0` branch so singleton credit is not over-claimed. Estimate `m` on train
from entity-level features (name/address quality, candidate count, best score, whether the entity has
address-less records). This is **post-processing only — no retraining**, testable in minutes against
the existing `oof2.parquet`, and it targets our single largest loss bucket directly.

### 3.6 Smaller items

- **`sizeshift.py` cannot answer the question it was written for.** `MODELS` (line 33) contains only
  v04 and v05 — r2 (`model_tl_r2`) is absent, though PROJECT_STATUS.md §7.2 says the script decides
  between all three. It also hardcodes `B.CAP_REF_N = s1c.height` (line 66, scale = 1), which matches
  how v04/v05 were built but *not* r2's test-time scale. Fix both before trusting its verdict — or
  skip it, because §2's fix makes the ordinary validation trustworthy and supersedes the simulation.
- **`by_expected_f(max_n=16)`** silently truncates entities with more than 16 assigned records.
  Low impact on a precision-weighted metric, but confirm it is rare rather than assuming it.
- **Per-country calibration.** `tune.fit_calibrator` fits one global isotonic curve. France's score
  distribution is a different regime (mean `p` 0.119 against 0.176 train US / 0.246 test US,
  `work/diag_adversarial2.txt:9`) and is being mapped through a US/India-dominated curve. Fit per
  country. Small change, and France is 15% of test.
- **`assign_argmax` is correct but fragile.** `sort(...).unique("qid", keep="first",
  maintain_order=False)` does pick the true maximum on polars 1.44.2 (verified empirically, 0
  mismatches on 2M rows). It relies on undocumented behaviour; prefer an explicit
  `group_by("qid").agg(...)`. Not a current bug — do not spend deadline time on it.

---

## 4. What to do, in order

Roughly 30 hours and 3 uploads remain. This ordering maximises expected leaderboard gain per hour
and front-loads everything that makes our validation trustworthy.

| # | Action | Effort | Why it is worth it |
|---|---|---|---|
| 1 | **Fix the variant** (§2): drop before blocking, size-matched per country, decoys subsampled to 41%. Re-run the pipeline. | ~2 h | Removes the shift that invalidates every local number. The only change likely to move the leaderboard a lot. |
| 2 | **`NFOLD=5`, `TRAIN_FRAC=1.0`**, average `K-1` models for the train context (§3.2, §3.4). | folded into 1 | Fixes the stage-2 handoff *and* triples the training data. |
| 3 | **Un-leak the local score** (§3.3a, §3.3b): inner early-stopping split; fit the calibrator and pick the rule on a held-out cluster-half. | ~30 min | Without this we still cannot tell a real gain from an artefact. |
| 4 | **Per-country isotonic calibration** (§3.6). | ~20 min | France is 15% of test and is being calibrated on the wrong curve. |
| 5 | **Expected-F with a missing-match term** (§3.5). | ~1 h, no retrain | Targets the 59%-of-loss misses-only bucket. Validate on existing `oof2.parquet`. |
| 6 | **Address-less blocking fallback** (§3.1) + a `no_address` feature. | ~1.5 h | ~82K true pairs are currently unreachable. Biggest pure-recall win available. |
| 7 | Drop the `p > 0.5` sibling threshold for the `_p`-weighted forms (§3.2.3). | ~15 min | Removes the last large pool/`p`-dependence in stage 2. |
| 8 | Map dropout for unseen countries (§3.3c). | ~45 min | France has no maps; the model currently assumes it does. |
| 9 | Hard-negative upweighting from `work/error_analysis_v02.txt` (Feedback 5 §S). | ~1 h | Cheap, the data already exists. Only after 1-3 land. |

**Do not start** the ranking reformulation, null model, match-count model, entity profiles, or
two-sided retrieval before the deadline. They are reasonable ideas with multi-hour costs and
uncertain payoffs, and they would be validated against a measurement instrument we have only just
repaired. Note them for a post-deadline writeup.

### Revised upload plan

| Upload | What | Gate |
|---|---|---|
| **#3** | The rebuild from actions 1-4. Do **not** spend an upload on v04/v05/r2 as they stand — all three were tuned against the broken variant. | Local score on the fixed variant, and validator PASS. |
| **#4** | #3 plus actions 5-7 (and 8 if it validates). | Must beat #3 on the *fixed* local validation. |
| **#5** | Best validated version, regenerated by one clean `run_all.py` run. | PROJECT_STATUS.md §8-C13. |

One caution on reading upload #3's result: if local drops to ~0.978 and the leaderboard *rises*,
that is the fix working, not a regression. Expect local and leaderboard to converge; the whole
purpose of action 1 is to make the two numbers comparable so that uploads #4 and #5 can be chosen on
evidence instead of argument.

### Still worth asking the organisers

Unchanged from PROJECT_STATUS.md §7, and now more urgent because the fix ordering depends on it:

- **Does the private leaderboard score the best submission or the last one?** If it is the last, #5
  must be our best validated model, not an experiment.
- Is there a size limit on the final zip? (`candidate_pairs.tsv` is ~420 MB uncompressed.)
- The pending pseudo-labelling question.

---

## 5. Results (measured 26 Sep, after implementing §4 items 1-5 and 7)

The corrected variant is `BER_VARIANT=fix` (config.py). It drops S1 entities **before**
blocking, cuts each country to test's actual S1 size, and thins decoy queries to a per-country
target. Models land in `work/model_fix/`, the submission in `output_fix/` (archived to
`submissions/fix_candidate/`, validator **PASS**).

### The candidate statistics now match test

| | cand/query | q_ncand_raw | pruned pairs | India s_ncand |
|---|---|---|---|---|
| legacy `tl` (v04/v05/r2 trained here) | 9.47 | 9.40 | 35.55M | — |
| **corrected `fix`** | **11.74** | **11.84** | 31.63M | **24.71** |
| **test** | **11.71** | **11.78** | 29.50M | **25.18** |

The residual shift sits in `nk` (8.92 vs 10.70) and `score` (106 vs 128), both already excluded by
`BER_ROBUST`. France is structurally different in a way train cannot reproduce: **`s_ncand` 334.6
against India's 25.2**, because its S1 index is small relative to its query volume.

### Like-for-like comparison, both scored on the corrected world

`score_existing.py` replays an old model set -- its fold models, its stage-2 context and the
decision rule it actually ships -- over the corrected world, so the two are measured on the same
test-shaped data.

| Build | F0.5 | US | India | singletons |
|---|---|---|---|---|
| r2 as shipped | 0.98656 | 0.98682 | 0.98635 | 0.9874 |
| **corrected build** | **0.98830** | **0.98892** | **0.98780** | 0.9866 |
| delta | **+0.00174** | +0.00210 | +0.00145 | −0.0008 |

For reference, r2 scored 0.98464 on the legacy world and 0.98656 on the corrected one. The
corrected world is *easier*, not harder -- halving US S1 removes competing near-twins, which is
equally true of the real test set. **My earlier prediction that the honest local score would fall
to ~0.978 was wrong, and wrong for an instructive reason:** the broken variant was not
uniformly flattering, it was differently shaped.

### What each fix was actually worth

| Fix | Outcome |
|---|---|
| Corrected variant (drop before blocking, size-matched) | Structural. Its value is that every number below can now be trusted; it is not itself a large point gain |
| `NFOLD=4`, `TRAIN_FRAC=0.6`, OOF = mean of the 3 non-owning models | The bulk of the +0.0017. Each model sees 45% of clusters instead of 30%, and stage 2 now gets a 3-model average at train time against 4 at test time instead of 1 against 2 |
| Early stopping moved off the scored fold | Removes optimism; no accuracy change |
| Out-of-sample calibration | **Changed a conclusion.** Scored honestly, expected-F0.5 selection is *not* better than a plain threshold: at stage 1, threshold 0.98570 beat ef_iso 0.98536. r2's apparent ef_iso edge was in-sample calibration optimism |
| Per-country calibration | +0.00000 at stage 2 (0.9882847 vs 0.9882837). Keep it for France's sake, but it earned nothing measurable |
| Prior-shift correction for India's light decoy pool | +0.00006 at stage 1, −0.00003 at stage 2. Not selected |
| Missing-match term in the expected-F DP | **No effect.** Verified correct against brute force, but the odds are only 0.018-0.023, so lambda ≈ 0.07 -- far too small to move any decision. My estimate of its value was simply wrong |
| Compact-name key given its own cap (`nc:`) | Implemented, **not yet in a build.** Raises compact-name key survival for India S1 from 78.8% to 99.4% and unblocks 8.9% of address-less queries. Needs a re-block of train *and* test |

### The real remaining problem is France, and we cannot measure it

Local validation covers US and India only -- 85% of test. If test US/India behave like the
corrected world (0.9883), then the observed leaderboard score implies:

```
LB = 0.85 * 0.9883 + 0.15 * F_france
v03's 0.976736  ->  F_france ~ 0.911
the leader's 0.9859  ->  F_france ~ 0.972
```

France plausibly accounts for most of the remaining gap to the top. It has no learned maps, and its
stage-2 entity-side features (`s_psum`, `s_nconf`, `s_nbest`, `sib_*_same`) are counts over a
334-candidate pool where training only ever saw ~25. **Making those features scale-free (shares and
ranks within the entity rather than raw counts) is the highest-value remaining change**, and unlike
everything in §4 it targets the part of test we have never been able to see.

## 6. What this says about our process

The honest-holdout check was good work and it answered its question correctly: there is no
model-fitting leakage. But we read it as "our validation is trustworthy," and it could never have
shown that — a holdout split inside a badly constructed variant validates the model, not the
variant. We then spent v04, v05 and r2 optimising against a number that was measuring a world with
9.47 candidates per query, and built an increasingly elaborate robustness story
(`BER_ROBUST`, size-scaled caps, adversarial diagnostics, `sizeshift.py`) around a symptom whose
cause was five lines in `features.py`.

The cheap habit that would have caught it: whenever a train/test discriminator flags a feature,
compare that feature's *construction path* on both sides before theorising about why the
distributions differ. `q_ncand_raw` was the #1 discriminator from the very first adversarial run
(`work/diag_tier0.txt`), and the answer was one `group_by` away the whole time.
