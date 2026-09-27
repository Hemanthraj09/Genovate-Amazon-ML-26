#!/usr/bin/env bash
# Candidate w<N>: v4b's settings trained on a different corrected world -- a fresh
# hash sample of which train entities are withheld and which decoys are kept.
# v2+v3 (two worlds) moved the leaderboard ~5x more than their local gain
# predicted, while same-world siblings transfer at 40-56%; this buys that kind of
# diversity at full v4b strength.
# Only the TRAIN side is rebuilt. Test candidates and features are v3's, so this
# model scores exactly the pairs v4/v4b score and averages with them pair for pair.
#     bash run_world.sh 1
set -euo pipefail
W="${1:?world number}"
export BER_VARIANT=fix
export BER_WORLD="$W"
export BER_MODEL_TAG="w$W"
export BER_TRAIN_FRAC=0.9
export BER_S2_FRAC=0.6
export BER_SEED=$((2027 + W))
export BER_LEAVES=383
export BER_LR=0.06
export BER_FEAT_FRAC=0.7
export PYTHONWARNINGS=ignore
export PYTHONIOENCODING=utf-8
R="$(cd ../../.. && pwd)"
export BER_OUT_DIR="$R/output_w$W"
L="$R/work/w${W}_build.log"

step () { echo "===== $1 ($(date +%H:%M:%S)) =====" | tee -a "$L"; }

step "blocking train (world $W)"
python -u blocking.py train 2>&1 | tee -a "$L"
step "features train"
python -u features.py train 2>&1 | tee -a "$L"
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
