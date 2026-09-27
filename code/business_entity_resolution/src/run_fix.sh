#!/usr/bin/env bash
# Corrected-variant build (feedback6): assumes blocking + features for the
# "fix" variant are already in work/cand/train_fix.parquet and work/feat/train_fix/,
# and that work/feat/test/ is still the r2 test feature set (unchanged by these fixes).
# Writes models to work/model_fix/ and the submission to output_fix/.
set -euo pipefail
export BER_VARIANT=fix
export PYTHONWARNINGS=ignore
export PYTHONIOENCODING=utf-8
export BER_OUT_DIR="$(cd ../../.. && pwd)/output_fix"
L="$(cd ../../.. && pwd)/work/fix_build.log"

step () { echo "===== $1 ($(date +%H:%M:%S)) =====" | tee -a "$L"; }

step "stage 1"
python -u train.py 2>&1 | tee -a "$L"
step "tune stage 1"
python -u tune.py oof 2>&1 | tee -a "$L"
step "stage 2"
python -u stage2.py 2 2>&1 | tee -a "$L"
step "tune stage 2"
python -u tune.py oof2 2>&1 | tee -a "$L"
step "predict test"
python -u predict.py 2 2>&1 | tee -a "$L"
step "done"
