#!/usr/bin/env bash
# Rebuild a model set's OOF honestly into a new folder model_fix_<build>h (the
# leaked build stays for comparison; copy lgb_fold*.txt and test_pred.parquet first).
# Rebuild (see train.predict_oof) from its saved
# stage-1 fold models, then retrain stage 2 on it, re-tune, and re-predict test
# from the saved stage-1 test predictions. No stage-1 retraining.
#     bash run_honest.sh v4b|v4|w1
set -uo pipefail
B="${1:?build: v4b | v4 | w1}"
# a queued rebuild can be deferred by creating work/SKIP_<build>
[ -f "../../../work/SKIP_$B" ] && { echo "skipping $B"; exit 0; }
export BER_VARIANT=fix BER_TRAIN_FRAC=0.9 BER_S2_FRAC=0.6 PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
case "$B" in
  v4)  export BER_MODEL_TAG=v4h ;;
  v4b) export BER_MODEL_TAG=v4bh BER_SEED=1337 BER_LEAVES=383 BER_LR=0.06 BER_FEAT_FRAC=0.7 ;;
  w1)  export BER_MODEL_TAG=w1h BER_WORLD=1 BER_SEED=2028 BER_LEAVES=383 BER_LR=0.06 BER_FEAT_FRAC=0.7 ;;
  *) echo "unknown build $B"; exit 2 ;;
esac
R="$(cd ../../.. && pwd)"
export BER_OUT_DIR="$R/output_${B}_h"
L="$R/work/honest_${B}.log"
run () {
  echo "===== $B: $* ($(date +%H:%M:%S)) =====" | tee -a "$L"
  "$@" 2>&1 | tee -a "$L"
  [ "${PIPESTATUS[0]}" -eq 0 ] || { echo "===== $B FAILED: $* =====" | tee -a "$L"; exit 1; }
}
run python -u -c "import train as T; T.predict_oof(T.load_models())"
run python -u tune.py oof
run python -u stage2.py 2
run python -u tune.py oof2
export BER_S1_FROM_FILE=1
run python -u predict.py 2
echo "===== $B done ($(date +%H:%M:%S)) =====" | tee -a "$L"
