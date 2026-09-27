#!/usr/bin/env bash
# A cheap ensemble member: a second stage-2 model on an honest build's stage-1
# predictions, with a different seed / tree shape. Stage 2 early-stops near 120
# rounds, so this costs ~25 min and no stage-1 work.
#     bash run_s2var.sh <base: v4bh|w1h|v4h> <variant name> <seed> <leaves> <lr> <feat_frac> <extra_trees 0|1>
set -uo pipefail
BASE="$1"; V="$2"
# a queued variant can be deferred by creating work/SKIP_s2var
[ -f "../../../work/SKIP_s2var" ] && { echo "skipping s2var $BASE $V"; exit 0; }
export BER_VARIANT=fix BER_TRAIN_FRAC=0.9 BER_S2_FRAC=0.6 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
export BER_SEED="$3" BER_LEAVES="$4" BER_LR="$5" BER_FEAT_FRAC="$6" BER_EXTRA_TREES="$7"
[ "$BASE" = "w1h" ] && export BER_WORLD=1
export BER_MODEL_TAG="${BASE}_${V}"
R="$(cd ../../.. && pwd)"
W="$R/work"
mkdir -p "$W/model_fix_${BER_MODEL_TAG}"
cp "$W/model_fix_${BASE}/oof.parquet" "$W/model_fix_${BASE}/test_pred.parquet" "$W/model_fix_${BER_MODEL_TAG}/"
export BER_OUT_DIR="$R/output_${BER_MODEL_TAG}"
L="$W/s2var_${BER_MODEL_TAG}.log"
run () {
  echo "===== ${BER_MODEL_TAG}: $* ($(date +%H:%M:%S)) =====" | tee -a "$L"
  "$@" 2>&1 | tee -a "$L"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "===== ${BER_MODEL_TAG} FAILED =====" | tee -a "$L"; exit 1; }
}
run python -u stage2.py 2
run python -u tune.py oof2
export BER_S1_FROM_FILE=1
run python -u predict.py 2
echo "===== ${BER_MODEL_TAG} done ($(date +%H:%M:%S)) =====" | tee -a "$L"
