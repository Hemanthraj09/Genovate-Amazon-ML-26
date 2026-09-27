#!/usr/bin/env bash
# Candidate aw<N>: v4b's settings on an ANCHORED world -- kept entities, their true
# copies, and only the look-alike decoys that imitate them (no orphans, no
# look-alikes of removed entities; see blocking.select_anchored). Matches test's
# candidate structure (US 2.4 / India 3.1 pruned candidates per query). Train side
# only; test candidates and features are shared with v4/v4b/w1.
#     bash run_anch.sh 3
set -euo pipefail
W="${1:?world number}"
export BER_VARIANT=fix BER_WORLD="$W" BER_DECOYS=anchored BER_MODEL_TAG="aw$W"
export BER_TRAIN_FRAC=0.9 BER_S2_FRAC=0.6 BER_SEED=$((3030 + W)) BER_LEAVES=383 BER_LR=0.06 BER_FEAT_FRAC=0.7
export PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
R="$(cd ../../.. && pwd)"
export BER_OUT_DIR="$R/output_aw$W"
L="$R/work/aw${W}_build.log"
step () { echo "===== $1 ($(date +%H:%M:%S)) =====" | tee -a "$L"; }
step "blocking train (anchored world $W)"; python -u blocking.py train 2>&1 | tee -a "$L"
step "features train";                    python -u features.py train 2>&1 | tee -a "$L"
step "stage 1";                           python -u train.py 2>&1 | tee -a "$L"
step "tune stage 1";                      python -u tune.py oof 2>&1 | tee -a "$L"
step "stage 2";                           python -u stage2.py 2 2>&1 | tee -a "$L"
step "tune stage 2";                      python -u tune.py oof2 2>&1 | tee -a "$L"
step "predict test";                      python -u predict.py 2 2>&1 | tee -a "$L"
step "done"
