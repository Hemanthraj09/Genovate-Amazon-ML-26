#!/usr/bin/env bash
# Candidate v4: v3's data, more of it per model.
# Reuses v3's blocking and features (work/cand/train_fix.parquet, work/feat/*),
# so only the models are rebuilt. Stage 1 goes from 45% to 67.5% of clusters per
# fold -- capacity is the lever that produced our one real leaderboard gain.
# Stage 2 stays at the smaller share: it early-stops near 100 iterations, so it is
# not data-hungry, and its 82-column design matrix is what threatens the 16 GB.
set -euo pipefail
export BER_VARIANT=fix
export BER_MODEL_TAG=v4b
export BER_SEED=1337
export BER_LEAVES=383
export BER_LR=0.06
export BER_FEAT_FRAC=0.7
export BER_TRAIN_FRAC=0.9
export BER_S2_FRAC=0.6
export PYTHONWARNINGS=ignore
export PYTHONIOENCODING=utf-8
R="$(cd ../../.. && pwd)"
export BER_OUT_DIR="$R/output_fix4b"
L="$R/work/fix4b_build.log"

step () { echo "===== $1 ($(date +%H:%M:%S)) =====" | tee -a "$L"; }

step "stage 1 (train_frac=$BER_TRAIN_FRAC)"
python -u train.py 2>&1 | tee -a "$L"
step "tune stage 1"
python -u tune.py oof 2>&1 | tee -a "$L"
step "stage 2 (frac=$BER_S2_FRAC)"
python -u stage2.py 2 2>&1 | tee -a "$L"
step "tune stage 2"
python -u tune.py oof2 2>&1 | tee -a "$L"
step "predict test"
python -u predict.py 2 2>&1 | tee -a "$L"
step "done"
