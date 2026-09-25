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

# Training variant. "tl" = test-like: a fixed DROP_FRAC share of train S1
# entities is removed, so their S2/S3 records become unmatched decoys. This
# reproduces the higher decoy share of the test set (~41% of S2/S3 records
# vs 27% in train) so the model learns the right prior and feature context.
VARIANT = os.environ.get("BER_VARIANT", "")
DROP_FRAC = float(os.environ.get("BER_DROP_FRAC", "0.2")) if VARIANT == "tl" else 0.0
TRAIN_TAG = "train" + (f"_{VARIANT}" if VARIANT else "")
MODEL_DIR = "model" + (f"_{VARIANT}" if VARIANT else "")

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
