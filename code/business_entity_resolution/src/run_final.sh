#!/usr/bin/env bash
# Regenerate the final submission end to end: raw TSVs -> output/*.tsv.
#
#   cd code/business_entity_resolution/src && bash run_final.sh
#
# Needs: requirements.txt installed; for the cross-encoder, a CUDA GPU, a Python
# with torch + transformers (CE_PYTHON, default: python) and the e5-small model
# files in E5_DIR. Roughly 12 h on a 16 GB / 24-thread laptop: model sets run
# one after another because two at once exhausts 16 GB. Every step writes its
# artefacts under BER_WORK_DIR, so a run can be resumed by commenting out steps.
#
# Final recipe (Documentation_template.md, section 5):
#   members   : five LightGBM model sets (stage 1 + stage 2), each trained on its
#               own validation world (see the build lines below)
#   ce blend  : e5-small cross-encoder on v4bh's uncertain pairs, blended with the
#               tree probability (tuned on held-out entities)
#   ensemble  : mean of the members' stage-2 test probabilities, blend applied
#   decision  : odds x ODDS (test carries ~2x the look-alike decoys of train),
#               best S1 per record, exact expected-F0.5 set per S1 entity
set -euo pipefail
export BER_VARIANT=fix PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
ODDS="${ODDS:-0.35}"
CE_PYTHON="${CE_PYTHON:-python}"
E5_DIR="${E5_DIR:?set E5_DIR to the folder with intfloat/multilingual-e5-small}"
R="$(cd ../../.. && pwd)"
W="${BER_WORK_DIR:-$R/work}"

step () { echo "===== $* ($(date +%H:%M:%S)) ====="; }

# ---- shared: parquet, learned maps, normalization, test candidates + features
step prepare;   python -u prepare.py
step maps;      python -u learn_maps.py
step normalize; python -u normalize_all.py
step "blocking test";  python -u blocking.py test
step "features test";  python -u features.py test
# the anchored world finds each decoy's anchor in full-train blocking
step "blocking full train (anchors)"; BER_VARIANT= python -u blocking.py train

# ---- one model set: world blocking + features (once per world), stage 1, stage 2, predict
build () {   # build <tag> <world> <decoys> <seed> <leaves> <lr> <feat_frac> <legacy_oof> [<reuse stage 1 from tag>]
  ( export BER_MODEL_TAG=$1 BER_WORLD=$2 BER_DECOYS=$3 BER_SEED=$4 BER_LEAVES=$5 BER_LR=$6 \
           BER_FEAT_FRAC=$7 BER_LEGACY_OOF=$8 BER_TRAIN_FRAC=0.9 BER_S2_FRAC=0.6
    local feat="$W/feat/train_fix$([ "$2" != 0 ] && echo "_w$2")$([ "$3" = anchored ] && echo _anch)"
    if [ ! -d "$feat" ]; then
      step "$1: blocking train"; python -u blocking.py train
      step "$1: features train"; python -u features.py train
    fi
    if [ -n "${9:-}" ]; then     # identical stage 1 (same world, seed, params): reuse it
      mkdir -p "$W/model_fix_$1"
      cp "$W/model_fix_$9"/lgb_fold*.txt "$W/model_fix_$9/test_pred.parquet" "$W/model_fix_$1/"
      step "$1: stage-1 OOF (reused models)"; python -u -c "import train as T; T.predict_oof(T.load_models())"
      export BER_S1_FROM_FILE=1
    else
      step "$1: stage 1";  python -u train.py
    fi
    step "$1: tune 1";   python -u tune.py oof
    step "$1: stage 2";  python -u stage2.py 2
    step "$1: tune 2";   python -u tune.py oof2
    step "$1: predict";  python -u predict.py 2 )
}
#     tag    world decoys    seed  leaves lr    ff   legacy-oof  reuse
build v4     0     ""        42    255    0.08  0.8  1
build v4b    0     ""        1337  383    0.06  0.7  1
build v4bh   0     ""        1337  383    0.06  0.7  0           v4b
build w1h    1     ""        2028  383    0.06  0.7  0
build aw3    3     anchored  3033  383    0.06  0.7  0

# ---- cross-encoder on v4bh's uncertain pairs (GPU), two seeds, then the blend
( export BER_MODEL_TAG=v4bh
  step "cross-encoder: prepare pairs";  python -u ce.py prep
  step "cross-encoder: run a";          "$CE_PYTHON" -u ce.py "$E5_DIR"
  step "cross-encoder: run b";          BER_CE_RUN=b "$CE_PYTHON" -u ce.py "$E5_DIR"
  step "cross-encoder: blend";          BER_CE_RUNS=",b" python -u ce_blend.py )

# ---- final ensemble + decision -> output/
MEMBERS="model_fix_v4bh model_fix_aw3 model_fix_w1h model_fix_v4b model_fix_v4"
step "ensemble ($MEMBERS, cross-encoder, odds x$ODDS)"
BER_SKIP_LOCAL=1 BER_CE=v4bh_ab BER_ODDS=$ODDS python -u ensemble.py output $MEMBERS
step done
