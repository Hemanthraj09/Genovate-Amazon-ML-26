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
