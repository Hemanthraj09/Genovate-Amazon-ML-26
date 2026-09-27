"""Central configuration: paths, seeds and shared constants.

Paths can be overridden with environment variables so the pipeline runs from
any checkout:
    BER_DATA_DIR  folder containing train/ and test/ (the organizer TSVs)
    BER_WORK_DIR  scratch folder for intermediates (parquet, features, models)
    BER_OUT_DIR   folder where matching_results.tsv / candidate_pairs.tsv go
"""
import os
from pathlib import Path

SRC_DIR = Path(__file__).resolve().parent
PKG_DIR = SRC_DIR.parent                      # code/business_entity_resolution
REPO_DIR = PKG_DIR.parent.parent              # project root

DATA_DIR = Path(os.environ.get(
    "BER_DATA_DIR", REPO_DIR / "dataset" / "student_resource" / "dataset"))
WORK_DIR = Path(os.environ.get("BER_WORK_DIR", REPO_DIR / "work"))
OUT_DIR = Path(os.environ.get("BER_OUT_DIR", REPO_DIR / "output"))

# Training variant.
#   "tl"  legacy test-like: a flat DROP_FRAC of train S1 entities is removed
#         *after* blocking. This biases the candidate sets badly -- blocking has
#         already truncated each query to TOPK, so deleting 20% of S1 leaves
#         ~9.5 candidates per query with nothing to refill the slots, while test
#         keeps a full ~11.7. Every candidate-count and sibling-count feature is
#         then on a different scale at train and test time. Kept only to
#         reproduce v04/v05/r2; do not build new models with it.
#   "fix" corrected test-like: S1 entities are removed *before* blocking, each
#         country is cut to test's actual S1 size, and decoy queries are
#         subsampled to DECOY_SHARE. Blocking then produces a full TOPK drawn
#         from a test-sized index, so candidate statistics match test.
VARIANT = os.environ.get("BER_VARIANT", "")
DROP_FRAC = float(os.environ.get("BER_DROP_FRAC", "0.2")) if VARIANT == "tl" else 0.0
# test S1 rows / train S1 rows, per country (France exists only in test)
SIZE_MATCH = {"US": 663_106 / 1_323_633, "India": 809_986 / 883_188}
# Estimated share of test S2/S3 records that match nothing, per country.
# Train has 3.46 true matches per S1 entity in BOTH countries independently
# (India 3.464, US 3.458), so treating it as a generator constant is well
# supported. Combined with test's measured records per S1 (US 5.76, India 5.82,
# France 5.53) that gives: decoy = 1 - 3.46 / (records per S1).
# France (0.374) is listed for reference only -- it has no train data.
DECOY_SHARE = {"US": 0.399, "India": 0.405}
# India cannot reach its target: keeping 91.7% of its entities (to match test's
# S1 size) leaves only 1.33M unmatched records against the 1.91M needed, because
# train simply has fewer India records than test. We keep the S1 size exact --
# that is what drives the candidate statistics -- use the whole decoy pool, and
# let per-country calibration absorb the residual prior offset.
DROP_BEFORE_BLOCKING = VARIANT == "fix"
# Which sample of kept entities / decoys the corrected variant draws (0 = the
# original world). Models trained on different worlds ensemble far better than
# seed-only siblings, so extra worlds are a cheap source of diversity.
WORLD = int(os.environ.get("BER_WORLD", "0"))
# "anchored": decoys are only the look-alikes of kept entities -- no orphans and no
# look-alikes of removed entities, which test does not contain (see
# blocking.select_anchored).
ANCHORED = os.environ.get("BER_DECOYS", "") == "anchored"
TRAIN_TAG = ("train" + (f"_{VARIANT}" if VARIANT else "") + (f"_w{WORLD}" if WORLD else "")
             + ("_anch" if ANCHORED else ""))
MODEL_TAG = os.environ.get("BER_MODEL_TAG", "")   # optional suffix to keep model sets apart
MODEL_DIR = "model" + (f"_{VARIANT}" if VARIANT else "") + (f"_{MODEL_TAG}" if MODEL_TAG else "")
# Blocking-score artefacts: their scale depends on dataset size and key caps, so
# they shift between train and test (adversarial AUC 0.99 -> 0.73 without them).
# When ROBUST is on they are not used as model features (blocking still uses them
# to build and prune the candidate set).
ROBUST = os.environ.get("BER_ROBUST", "1") == "1"
BLOCKING_ARTEFACTS = ("score", "nk", "gap_best", "gap_second", "q_ncand_raw", "s_ncand",
                      "s_rank", "rank", "rel")

SEED = 42
N_THREADS = max(1, (os.cpu_count() or 4) - 2)

SPLITS = ("train", "test")
SOURCES = (1, 2, 3)


def raw_path(split: str, src: int) -> Path:
    """Path of an organizer TSV, e.g. train/train_source2.tsv."""
    return DATA_DIR / split / f"{split}_source{src}.tsv"


def gt_path() -> Path:
    """Path of the training ground-truth TSV."""
    return DATA_DIR / "train" / "train_ground_truth.tsv"


def work(*parts) -> Path:
    """Path inside the work dir (parent folders are created on demand)."""
    p = WORK_DIR.joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p
