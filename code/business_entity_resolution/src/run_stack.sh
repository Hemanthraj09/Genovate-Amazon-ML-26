#!/usr/bin/env bash
# Candidate v4s: stage 2 retrained on the average of v4's and v4b's stage-1
# probabilities. Both builds share v3's world and fold hash, so the averaged OOF
# is still out-of-fold and stage 2 is tuned end to end on it, unlike ensemble.py,
# which averages finished stage-2 outputs and borrows one build's decision rule.
set -euo pipefail
export BER_VARIANT=fix
export BER_MODEL_TAG=v4s
export BER_TRAIN_FRAC=0.9
export BER_S2_FRAC=0.6
export PYTHONWARNINGS=ignore
export PYTHONIOENCODING=utf-8
R="$(cd ../../.. && pwd)"
export BER_OUT_DIR="$R/output_stack"
L="$R/work/stack_build.log"

step () { echo "===== $1 ($(date +%H:%M:%S)) =====" | tee -a "$L"; }

step "average stage 1 (v4, v4b)"
python -u stack.py model_fix_v4 model_fix_v4b 2>&1 | tee -a "$L"
step "tune stage 1 (averaged)"
python -u tune.py oof 2>&1 | tee -a "$L"
step "stage 2 (frac=$BER_S2_FRAC)"
python -u stage2.py 2 2>&1 | tee -a "$L"
step "tune stage 2"
python -u tune.py oof2 2>&1 | tee -a "$L"
step "predict test"
BER_S1_FROM_FILE=1 python -u predict.py 2 2>&1 | tee -a "$L"
step "done"
