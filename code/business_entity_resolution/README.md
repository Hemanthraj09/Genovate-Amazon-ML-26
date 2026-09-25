# Business Entity Resolution: Team Genovate (Amazon ML Challenge 2026)

This package runs the whole pipeline end to end. It takes the organizer TSVs, runs normalization, blocking, features, the matching model and the decision rule, and writes `output/matching_results.tsv` and `output/candidate_pairs.tsv`.

It uses only the provided training and test data: no external data, APIs, geocoding or pretrained models. The only model is LightGBM (MIT license), a gradient-boosted tree model.

## Environment

- Python 3.14 (developed on 3.14.5, Windows 11). The pipeline runs on CPU only; no GPU is needed.
- Hardware used: 16 GB RAM and 24 threads. Peak memory is about 12 GB.
- Install the pinned dependencies:

```bash
pip install -r requirements.txt
```

## Data layout

By default the code expects the organizer files at `<repo>/dataset/student_resource/dataset/{train,test}/`. You can override every location with environment variables:

| Variable | Meaning | Default |
|---|---|---|
| `BER_DATA_DIR` | folder containing `train/` and `test/` | `<repo>/dataset/student_resource/dataset` |
| `BER_WORK_DIR` | intermediates (parquet, candidates, features, models) | `<repo>/work` |
| `BER_OUT_DIR` | final submission files | `<repo>/output` |
| `BER_VARIANT` | training variant; the final submission uses `tl` (test-like) | empty |

## Reproduce the submission

```bash
cd src
set BER_VARIANT=tl          # PowerShell: $env:BER_VARIANT="tl"   bash: export BER_VARIANT=tl
python run_all.py           # every step, about 1.5 h on the machine above
```

Each step writes its artefacts to `BER_WORK_DIR`, so you can resume from any step with `python run_all.py <step>`.

| Step | Script | What it does |
|---|---|---|
| `prepare` | `prepare.py` | Converts the TSVs to parquet, gives each record an integer row index, and turns the ground truth into a table of true pairs. |
| `maps` | `learn_maps.py` | Learns from training pairs an Indian-script to Latin word dictionary and per-country address substitution maps (for example TX↔Texas, MH↔Maharashtra, Bombay→Mumbai, street-type typos). |
| `normalize` | `normalize_all.py`, `textnorm.py` | Normalizes names and addresses: legal forms, titles, "doing business as" names, domains and handles, look-alike digits, placeholder addresses, house numbers and PO boxes. The rules are the same for every country. |
| `blocking` | `blocking.py` | Candidate generation. For every Source 2/3 record it finds the top-12 Source 1 records by IDF-weighted shared keys (name tokens and bigrams, compact name, typo-robust prefixes, address tokens and bigrams, house numbers including truncated variants, and name×number, name×address-word and number×address-word combinations). |
| `features` | `features.py` | Keeps candidates scoring at least 0.3× the record's best, then computes 62 pair features: rapidfuzz similarities, TF-IDF cosines, house-number and number-set logic, legal forms, acronyms, ambiguity counts and blocking context. |
| `train` | `train.py` | Stage-1 LightGBM, 2 folds grouped by query cluster, producing out-of-fold probabilities. |
| `tune` | `tune.py` | Chooses the decision rule on out-of-fold predictions using the exact macro F0.5. |
| `stage2` | `stage2.py` | Stage-2 LightGBM on stage-1 features plus context features built from the out-of-fold stage-1 probabilities. |
| `tune2` | `tune.py oof2` | Chooses the decision rule for stage 2. |
| `predict` | `predict.py`, `decide.py`, `output.py` | Scores the test pairs, assigns each record to its best Source 1 match, selects each entity's match set by exact expected F0.5 after isotonic calibration, and writes both TSVs. |

Validate the output with the organizer's script (from `dataset/student_resource/`):

```bash
python utils/validate_submission.py --matching ../../output/matching_results.tsv --candidate ../../output/candidate_pairs.tsv --test-dir dataset/test --check-ids
```

## Other files

- `evaluate.py`: the exact macro F0.5 from the problem statement, with a self-test on its worked example.
- `crosseval.py`: scores one variant's models on another variant's data, used to compare training setups.
- `bench_block.py`: a benchmark for blocking recall on a sample.
- `config.py`: paths, seeds and the variant switch.
