# Genovate: Amazon ML Challenge 2026, Business Entity Resolution

**Result: rank 677**, public leaderboard **macro F0.5 = 0.984083**. We were shortlisted in the Top 1,000 teams.

**Team Genovate:** Hemanth Raj, Kushal K V, Ayush Khanuja. The challenge ran for 72 hours, 25–27 September 2026, on Unstop.

## The task

Business records come from three sources that share no identifiers. Each record has a name, an address and a country (US, India, and France in test only). For every record in Source 1, the deduplicated reference list, the task is to find all matching records in Sources 2 and 3.

Scoring is macro F0.5 per Source 1 entity. It weights precision twice as much as recall. An entity with no true match scores 1.0 only if nothing is predicted for it.

## Our approach at a glance

1. **Blocking.** For each Source 2/3 record we retrieve its top 12 Source 1 candidates, ranked by IDF-weighted shared keys (name tokens, compact name, typo-robust prefixes, address words, house numbers and cross keys). This gives 30M test pairs, and a perfect classifier on them would score 0.9955.
2. **Two-stage LightGBM.**
   - Stage 1 scores each pair with 56 string-similarity features.
   - Stage 2 adds 28 context features: margin to competing candidates, and agreement among an entity's other candidates.
   - The final model averages five model sets, each trained on a different "test-shaped" validation world.
3. **Cross-encoder on the hard pairs.** We fine-tuned a small pretrained multilingual encoder (multilingual-e5-small, MIT, 118M parameters) on a laptop GPU. It reads the raw text of both records for every pair the trees are unsure about. Blending it in was our biggest single gain: **+0.003** on the leaderboard.
4. **Decision.** Each record goes to its best entity. Each entity's match set is chosen by maximising the *exact* expected F0.5. Probabilities are first shrunk to correct for test having about twice the look-alike decoys of train.

The full write-up is in **[SOLUTION_DETAILED.md](SOLUTION_DETAILED.md)**.

## Repository map

| Path | What |
|---|---|
| `code/business_entity_resolution/src/` | The pipeline (entry point `run_final.sh`) |
| `code/business_entity_resolution/README.md` | Setup and end-to-end reproduction |
| `Documentation_template.md` | Methodology document submitted to the organisers |
| `SOLUTION_DETAILED.md` | Detailed write-up: findings, results, what worked and what didn't |
| `PROJECT_STATUS.md`, `feedback6.md` | Working notes from the challenge |
| `submissions/SUBMISSIONS.md` | Every leaderboard upload with its commit and score |
| `make_submission_zip.py` | Builds and validates the organiser's submission archive |

## Run it

```bash
pip install -r code/business_entity_resolution/requirements.txt
cd code/business_entity_resolution/src
E5_DIR=/path/to/multilingual-e5-small bash run_final.sh
```

The organiser dataset is not included in this repository.
