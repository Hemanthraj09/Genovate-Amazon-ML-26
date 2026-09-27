#!/usr/bin/env bash
# Regenerate the final submission end to end: raw TSVs -> output/*.tsv.
#
#   cd code/business_entity_resolution/src && bash run_final.sh
#
# Roughly 10-12 h on a 16 GB / 24-thread laptop (each model set ~2 h; they run one
# after another because two at once exhausts 16 GB). Every step writes its
# artefacts under BER_WORK_DIR, so an interrupted run can be resumed by commenting
# out the finished steps.
#
# Final recipe (see Documentation_template.md, section 5):
#   members  : the model sets listed in MEMBERS below, each = stage-1 LightGBM
#              (4 folds) + stage-2 LightGBM, trained on its own validation world
#   ensemble : mean of the members' stage-2 test probabilities (union of pairs)
#   decision : odds x ODDS (test carries ~2x the look-alike decoys of train),
#              best S1 per record, exact expected-F0.5 set per S1 entity
set -euo pipefail
export BER_VARIANT=fix PYTHONWARNINGS=ignore PYTHONIOENCODING=utf-8
ODDS="${ODDS:-0.35}"

step () { echo "===== $* ($(date +%H:%M:%S)) ====="; }

# ---- shared: parquet, learned maps, normalization, test candidates + features
step prepare;   python -u prepare.py
step maps;      python -u learn_maps.py
step normalize; python -u normalize_all.py
step "blocking test";  python -u blocking.py test
step "features test";  python -u features.py test
# the anchored world needs full-train blocking to find each decoy's anchor
step "blocking full train (anchors)"; BER_VARIANT= python -u blocking.py train

# ---- one model set: world blocking + features (once per world), stage 1, stage 2, predict
build () {   # build <tag> <world> <decoys> <seed> <leaves> <lr> <feat_frac> <legacy_oof>
  local tag=$1
  ( export BER_MODEL_TAG=$1 BER_WORLD=$2 BER_DECOYS=$3 BER_SEED=$4 BER_LEAVES=$5 BER_LR=$6 \
           BER_FEAT_FRAC=$7 BER_LEGACY_OOF=$8 BER_TRAIN_FRAC=0.9 BER_S2_FRAC=0.6
    local feat="../../../work/feat/train_fix$([ "$2" != 0 ] && echo "_w$2")$([ "$3" = anchored ] && echo _anch)"
    if [ ! -d "$feat" ]; then
      step "$tag: blocking train"; python -u blocking.py train
      step "$tag: features train"; python -u features.py train
    fi
    step "$tag: stage 1";  python -u train.py
    step "$tag: tune 1";   python -u tune.py oof
    step "$tag: stage 2";  python -u stage2.py 2
    step "$tag: tune 2";   python -u tune.py oof2
    step "$tag: predict";  python -u predict.py 2 )
}
#     tag    world decoys    seed  leaves lr    ff   legacy-oof
build v4     0     ""        42    255    0.08  0.8  1
build v4b    0     ""        1337  383    0.06  0.7  1
build v4bh   0     ""        1337  383    0.06  0.7  0
build w1h    1     ""        2028  383    0.06  0.7  0
build aw3    3     anchored  3033  383    0.06  0.7  0

# ---- final ensemble + decision -> output/
MEMBERS="model_fix_v4b model_fix_v4"          # TODO: final member list, set on 27 Sep
step "ensemble ($MEMBERS, odds x$ODDS)"
BER_ODDS=$ODDS BER_OUT_DIR="$(cd ../../.. && pwd)/output" python -u ensemble.py output $MEMBERS
step done
