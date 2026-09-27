# Business Entity Resolution: Team Genovate (Amazon ML Challenge 2026)

This package regenerates our final submission from the organizer TSVs:

- `output/matching_results.tsv`, the decided matches uploaded to the leaderboard
- `output/candidate_pairs.tsv`, exactly the pairs the models score

It uses only the provided training and test data, with no external data, APIs or geocoding. There are two kinds of model:

- **LightGBM** (MIT) for blocking-candidate scoring
- **intfloat/multilingual-e5-small** (MIT, 118M parameters), a small pretrained encoder, fine-tuned as a cross-encoder on the pairs the tree models are unsure about

Both are within the challenge's licence and ≤ 8B-parameter rule.

## Environment

- Python 3.14 (developed on 3.14.5, Windows 11).
- Tested hardware: 16 GB RAM, 24 threads, and an NVIDIA RTX 4050 laptop GPU (6 GB). The GPU is only used by the cross-encoder; everything else runs on the CPU. Peak RAM is about 13 GB, so run one model build at a time.
- Install the dependencies (see the comments in `requirements.txt` for the CUDA build of torch):

```bash
pip install -r requirements.txt
```

The cross-encoder needs the model files of `intfloat/multilingual-e5-small` (`model.safetensors`, `tokenizer.json`, `sentencepiece.bpe.model`, `config.json`) in a local folder. Its path is passed to `ce.py`.

## Data layout

By default the code expects the organizer files at `<repo>/dataset/student_resource/dataset/{train,test}/`. Every location can be overridden:

| Variable | Meaning | Default |
|---|---|---|
| `BER_DATA_DIR` | folder containing `train/` and `test/` | `<repo>/dataset/student_resource/dataset` |
| `BER_WORK_DIR` | intermediates (parquet, candidates, features, models) | `<repo>/work` |
| `BER_OUT_DIR` | submission files | `<repo>/output` |

## Reproduce the submission

```bash
cd src
bash run_final.sh        # every step; ~12 h on the machine above (model sets run one after another)
```

`run_final.sh` lists every step with its exact settings. Each step writes its artefacts to `BER_WORK_DIR`, so an interrupted run can be resumed.

### Final recipe

1. **Shared preparation:** parquet conversion, learned normalisation maps, normalisation, then test blocking and features.
2. **Five LightGBM model sets.** Each is a stage-1 model (4 folds grouped by query cluster) plus a stage-2 context model, trained on its own *validation world*:

   | Model set | World | Notes |
   |---|---|---|
   | `v4` | 0 | `BER_LEGACY_OOF=1` (see below) |
   | `v4b` | 0 | `BER_LEGACY_OOF=1` |
   | `v4bh` | 0 | honest OOF |
   | `w1h` | 1 | honest OOF |
   | `aw3` | 3, anchored decoys | honest OOF |

3. **Cross-encoder** (`ce.py` then `ce_blend.py`), trained on `v4bh`'s uncertain pairs. The blend with the tree probability is tuned on held-out entities: honest F0.5 goes from 0.98634 to 0.98892.
4. **Ensemble and decision** (`ensemble.py`):
   - mean of the five model sets' stage-2 test probabilities
   - the cross-encoder blend on the uncertain pairs
   - odds multiplied by `BER_ODDS`, because test carries about twice the look-alike decoys per entity of train
   - best S1 per record, then the exact expected-F0.5 set per S1 entity
5. **Validation** with the organizer script:

```bash
python ../../../dataset/student_resource/utils/validate_submission.py --check-ids \
  --test-dir ../../../dataset/student_resource/dataset/test \
  --matching ../../../output/matching_results.tsv --candidate ../../../output/candidate_pairs.tsv
```

### Reproducibility notes

- **`BER_LEGACY_OOF=1`** reproduces how the `v4`/`v4b` model sets were built. Their stage-2 models were trained on out-of-fold predictions averaged over models that had seen those rows, a leak we found and fixed on 27 Sep (`train.predict_oof`). Their stage-1 models are unaffected. All later model sets use honest OOF (fold k scored by model k).
- **Blocking tie-breaks.** At the top-12 boundary, candidates with equal scores are cut in an order that can differ between runs. Regenerated candidate sets therefore match to within a small fraction of pairs, not bit for bit.

## Pipeline steps

| Step | Script | What it does |
|---|---|---|
| prepare | `prepare.py` | TSVs to parquet with integer row ids, plus a table of true pairs |
| maps | `learn_maps.py` | Learns from train pairs an Indic-script to Latin word dictionary and per-country address substitutions (TX ↔ Texas, Bombay → Mumbai, street-type typos) |
| normalize | `normalize_all.py`, `textnorm.py` | Names: legal forms, honorifics, DBA splitting, domains and handles, look-alike digits. Addresses: abbreviations, placeholders, house numbers, units and PO boxes. The same rules apply to every country |
| validation world | `config.py`, `blocking.py` | `BER_VARIANT=fix` cuts each country's S1 to test's size *before* blocking and samples decoys to test's share. `BER_WORLD` reseeds the sample. `BER_DECOYS=anchored` keeps only decoys that imitate kept entities |
| blocking | `blocking.py` | Top-12 S1 per S2/S3 record by IDF-weighted shared keys: name tokens and bigrams, compact name, 4-character prefixes, address words, bigrams and digits (with truncations), and name × number / word keys. Caps scale with S1 size |
| features | `features.py` | Keeps candidates scoring at least 0.3 × the record's best, then computes 56 pair features: RapidFuzz similarities, TF-IDF cosines, house-number logic, legal forms, acronyms, ambiguity counts, core-address similarity |
| stage 1 | `train.py` | LightGBM with 4 folds grouped by query cluster; out-of-fold probabilities |
| tune | `tune.py` | Chooses the decision rule on OOF with the exact macro F0.5, scored on held-out entity halves |
| stage 2 | `stage2.py` | Stage-1 features plus 28 context features built from stage-1 OOF: the record's margin and share, the entity's mass, and sibling agreement |
| predict | `predict.py`, `decide.py`, `output.py` | Scores test, decides the matches, writes both TSVs |
| cross-encoder | `ce.py`, `ce_blend.py` | Fine-tunes e5-small on uncertain pairs over two entity halves, then tunes the blend on held-out entities |
| ensemble | `ensemble.py` | Averages model sets, applies the blend and `BER_ODDS`, decides, writes the output |

## Other files

- `evaluate.py`: the exact macro F0.5 from the problem statement, with its worked example as a self-test.
- `run_*.sh`: one script per build type used during the challenge.
- `stack.py`, `loco.py`, `score_existing.py`, `sizeshift.py`, `holdout.py`, `crosseval.py`, `bench_block.py`, `predict_unseen.py`: experiment and diagnostic tools (not needed for the final file).
