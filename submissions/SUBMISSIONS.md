# Submission log (Genovate)

Every leaderboard upload is recorded here with the git commit that produced it.
The uploaded file for each version is kept locally as `submissions/vNN/matching_results.tsv` (git-ignored because of its size).

| Ver | Date (IST) | Git commit | Change tested | Local OOF F0.5 | Public LB | Notes |
|-----|------------|------------|---------------|----------------|-----------|-------|
| v01 | 25 Sep | (tag v01) | Baseline: multi-key blocking (top-12, rel>=0.3), 55 pair features, 2-fold LightGBM, best-S1-per-record + tau=0.7 | 0.98229 (US 0.98316, India 0.98100) | 0.976553 | Validator PASS with --check-ids. Test: 5.8% empty rows, France behaves like US/India |
| v02 | 25 Sep | (tag v02) | + Stage-2 competition-aware re-scoring (OOF stage-1 context) + per-entity exact expected-F0.5 set selection | 0.98537 (US 0.98625, India 0.98406) | not uploaded | Superseded by v03 |
| v03 | 25 Sep | (tag v03) | Test-like training (20% of train S1 dropped -> ~41% decoys like test) + stage 2 + isotonic expected-F0.5 | test-like val 0.98316 (v02 approach on same data: 0.98158) | 0.976736 | Validator PASS --check-ids |
| v04 | 25/26 Sep | (tag v04) | v1 rebuild: France normalization (n°, hyphens, generic words, global typo map), +7 near-twin/acronym features, 1200 rounds; stage 2 + sibling-agreement features + common-token-stripped address similarity | test-like val 0.98550 (US 0.98639, India 0.98416) | not yet uploaded | Validator PASS --check-ids. Test uncertain share: France 8.4% (was 10.3%), India 4.3% (5.8%), US 5.1% (5.3%) |
| v05 (candidate) | 25 Sep | (tag v05) | ROBUST features: 9 blocking-score artefacts removed from the model (adversarial AUC 0.99 -> 0.73), stage-2 sibling counts over confident siblings only, no candidate-count feature | test-like val 0.98486 (US 0.98574, India 0.98355) | not yet uploaded | Validator PASS --check-ids. Stage-2 design adversarial AUC US 0.877 / India 0.825 (v04 ~0.99). Test uncertain: France 7.4%, India 4.0%, US 4.6% |
| r2 (candidate) | 26 Sep | d75657f | Robust stack + size-scaled blocking caps + alphabetic-only hyphen split + French elision; re-learned maps | test-like val 0.98464 (US 0.98540, India 0.98351); blocking recall 0.98439 | not uploaded | Validator PASS --check-ids. Archived in submissions/r2_candidate |
| fix | 26 Sep | 1a66ced (world code) | Corrected validation world: S1 cut to test size BEFORE blocking; decoys to test share | 0.98830 (leaked OOF) | 0.977783 | submissions/fix_candidate |
| probe | 26 Sep | — | France rows emptied (diagnostic): splits LB into US+India ~0.982 / France ~0.955 | — | 0.84307 | output_diag_france |
| fix2 (v2) | 26 Sep | 1a66ced | + compact-name blocking key (own cap), core-address features | leaked | 0.977854 | submissions/fix2_candidate |
| ens v2+v3 | 26 Sep | 9853337 | mean of v2 and v3 (natural-first decoys) stage-2 probabilities | leaked | 0.978677 | submissions/ens_candidate |
| v4 | 26 Sep | 1a66ced | 67.5% of clusters per stage-1 fold model | 0.99005 leaked | 0.97913 | submissions/fix4_candidate |
| ens v4+v4b o0.5 | 27 Sep 12:15 | 9853337 | + v4b (seed 1337, 383 leaves), odds x0.5 (test decoy density) | 0.99029 leaked | 0.979992 | submissions/ens_o50_uploaded |
| ens v4+v4b o0.35 | 27 Sep 12:19 | 9853337 | same, odds x0.35 | — | **0.980005** | submissions/ens_o035_uploaded (best) |
| v4bh o0.35 | 27 Sep ~13:35 | 6b42ea7 | honest OOF (leak fixed), stage 2 retrained; single model | 0.98634 honest | 0.979403 | submissions/v4bh_o035_uploaded |
| ens5 o0.3 | 27 Sep | 161e809 | 5 model sets (v4bh, aw3, w1h, v4b, v4), odds x0.3, no cross-encoder | aw3 0.98785 honest | 0.98092 | output_ens5_o030 |
| ens5 + CE(a) o0.35 | 27 Sep | 161e809 | + e5-small cross-encoder blend on uncertain pairs | 0.98892 honest (v4bh+CE) | 0.983899 | output_ens5ce_o035 |
| final (ens5 + CE a+b) o0.35 | 27 Sep | eae4f3c | two cross-encoder runs averaged | 0.98899 honest | 0.984006 | output_final, submissions/final_candidate |
| France x0.2 | 27 Sep | eae4f3c | as final, France odds 0.2 | — | 0.983818 | output_fr02 |
| France x0.6 | 27 Sep | eae4f3c | as final, France odds 0.6 | — | **0.984083** | output_fr06 (best so far) |
| France x1.0 | 27 Sep | eae4f3c | as final, France odds 1.0 | — | pending | uploads/matching_results_fr10.tsv.gz |
| US/India x0.5 | 27 Sep | eae4f3c | US/India odds 0.5, France 0.6 | — | pending | uploads/matching_results_ui05.tsv.gz |
| final_c | 27 Sep | (this commit) | cross-encoder run c (all ~690k pairs per half; held-out 0.98932), US/India x0.35, France x0.6 | 0.98932 honest (v4bh+CE c) | pending | uploads/matching_results_final_c.tsv.gz, submissions/final_c_candidate; **Genovate_submission.zip built from it (validated)** |
