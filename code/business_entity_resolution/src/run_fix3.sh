#!/usr/bin/env bash
# Candidate v2 on the corrected variant. Two changes over model_fix:
#   * blocking: the whole compact name gets its own key type ("nc:") and so the
#     generous cap instead of the tight single-token one -- address-less records
#     often have no other selective key (recall 75.7% vs 99.5%).
#   * features: adk_* similarity on the country-common-token-stripped address.
#     France S1 carries a region where its S2/S3 records carry a department, and
#     it gets no learned maps, so this is the address signal it can actually use.
set -euo pipefail
export BER_VARIANT=fix
export BER_MODEL_TAG=v3
export PYTHONWARNINGS=ignore
export PYTHONIOENCODING=utf-8
R="$(cd ../../.. && pwd)"
export BER_OUT_DIR="$R/output_fix3"
L="$R/work/fix3_build.log"

step () { echo "===== $1 ($(date +%H:%M:%S)) =====" | tee -a "$L"; }

step "blocking"
python -u blocking.py train test 2>&1 | tee -a "$L"
step "features"
python -u features.py train test 2>&1 | tee -a "$L"
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
