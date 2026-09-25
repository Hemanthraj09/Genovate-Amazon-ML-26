# Submission log (Genovate)

Every leaderboard upload is recorded here with the git commit that produced it.
The uploaded file for each version is kept locally as `submissions/vNN/matching_results.tsv` (git-ignored because of its size).

| Ver | Date (IST) | Git commit | Change tested | Local OOF F0.5 | Public LB | Notes |
|-----|------------|------------|---------------|----------------|-----------|-------|
| v01 | 25 Sep | (tag v01) | Baseline: multi-key blocking (top-12, rel>=0.3), 55 pair features, 2-fold LightGBM, best-S1-per-record + tau=0.7 | 0.98229 (US 0.98316, India 0.98100) | 0.977 | Validator PASS with --check-ids. Test: 5.8% empty rows, France behaves like US/India |
| v02 | 25 Sep | (tag v02) | + Stage-2 competition-aware re-scoring (OOF stage-1 context) + per-entity exact expected-F0.5 set selection | 0.98537 (US 0.98625, India 0.98406) | _pending_ | Validator PASS --check-ids. Train mix (27% decoys) |
