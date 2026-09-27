"""Build the final submission archive in the layout the problem statement requires.

    python make_submission_zip.py <output folder>      e.g.  output_final

Genovate_submission.zip
├── output/
│   ├── matching_results.tsv
│   └── candidate_pairs.tsv
├── code/business_entity_resolution/
│   ├── src/                 (all .py and .sh; no caches)
│   ├── README.md
│   └── requirements.txt
└── Documentation_template.md

It then extracts the two TSVs from the finished zip into a temp folder and runs the
organizer's validator on them with --check-ids, so what is checked is exactly
what is shipped.
"""
import subprocess
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PKG = ROOT / "code" / "business_entity_resolution"
ZIP = ROOT / "Genovate_submission.zip"
VALIDATOR = ROOT / "dataset" / "student_resource" / "utils" / "validate_submission.py"
TEST_DIR = ROOT / "dataset" / "student_resource" / "dataset" / "test"


def main(out_dir):
    out = ROOT / out_dir
    files = {out / "matching_results.tsv": "output/matching_results.tsv",
             out / "candidate_pairs.tsv": "output/candidate_pairs.tsv",
             PKG / "README.md": "code/business_entity_resolution/README.md",
             PKG / "requirements.txt": "code/business_entity_resolution/requirements.txt",
             ROOT / "Documentation_template.md": "Documentation_template.md"}
    for f in sorted((PKG / "src").iterdir()):
        if f.is_file() and f.suffix in (".py", ".sh"):
            files[f] = f"code/business_entity_resolution/src/{f.name}"
    missing = [str(f) for f in files if not f.exists()]
    if missing:
        sys.exit(f"missing: {missing}")
    with zipfile.ZipFile(ZIP, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for src, arc in files.items():
            z.write(src, arc)
    print(f"wrote {ZIP.name}: {len(files)} files, {ZIP.stat().st_size / 2**20:.1f} MB")
    with tempfile.TemporaryDirectory() as tmp:
        with zipfile.ZipFile(ZIP) as z:
            z.extractall(tmp)
        r = subprocess.run([sys.executable, str(VALIDATOR), "--check-ids", "--test-dir", str(TEST_DIR),
                            "--matching", str(Path(tmp) / "output" / "matching_results.tsv"),
                            "--candidate", str(Path(tmp) / "output" / "candidate_pairs.tsv")],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        last = r.stdout.strip().splitlines()[-1] if r.stdout.strip() else r.stderr
        print(last.encode("ascii", "replace").decode())
        if r.returncode != 0:
            sys.exit("VALIDATION FAILED")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "output")
