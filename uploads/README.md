# Files ready to upload

Gzip-compressed `matching_results.tsv` files (git ignores the raw TSVs: each is ~97 MB).
To upload one: decompress it (`gunzip -k matching_results_fr10.tsv.gz`, or 7-Zip on Windows),
rename the result to `matching_results.tsv` if the portal asks for that name, and upload it.

| File | What | Compare against |
|---|---|---|
| **`matching_results_final_c.tsv.gz`** | **Upload this first.** Final recipe with the stronger cross-encoder (run c, all training pairs: held-out +0.0030 vs +0.0026), US/India x0.35, France x0.6 | best so far 0.984083 |
| `matching_results_fr10.tsv.gz` | final recipe, France odds x1.0 (US/India x0.35) | best so far 0.984083 (France x0.6) |
| `matching_results_ui05.tsv.gz` | final recipe, US/India odds x0.5 (France x0.6) | 0.984083 |

Final recipe = mean of 5 LightGBM model sets (v4bh, aw3, w1h, v4b, v4) + e5-small cross-encoder blend
on uncertain pairs (2 runs averaged) + per-country odds, then the expected-F0.5 decision.
