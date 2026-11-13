# HMOG/FETA — frozen subject-disjoint splits (v1)

**Protocol:** one fixed 60/20/20 subject-level split, seed `20261112`, no cross-validation. Minimum **20 strictly multimodal valid gestures** in **each of the four chronologically earliest task-specific sessions**. Gestures use FETA's completed DOWN→UP, ≥3 touch samples and strictly >5 native-pixel maximum radial distance. HMOG qualifying gestures must have portrait orientation (`Phone_orientation=0`) at every touch point; strict motion is defined by the version 2.0 audit configuration.

## Output files

- `experiments/splits/hmog_feta_splits_seed_20261112.json` — the immutable split to commit to Git. Has per-scenario subject IDs, per-subject first-four session IDs, 60/20/20 quotas, preprocessing protocol and SHA-256 hashes of all five audit input files.
- `hmog_feta_splits_validation.json` — verification of the frozen split against the uploaded v2.0 audit outputs; also includes a secondary check using the older `user_device_mapping.csv` from the v1.2 audit. This is not independent re-extraction of the raw gestures.
- `make_hmog_feta_splits_v1.py` — reproduces the JSON from the **original** v2.0 audit output folder.
- `validate_hmog_feta_splits_v1.py` — verifies the JSON against the audit output; can also recompute gesture counts from local raw data using the same v2.0 parser.

## Fixed cohort sizes

| Dataset and scenario | Train | Validation | Test | Total |
| --- | ---: | ---: | ---: | ---: |
| FETA iPhone 7 (same users for scroll and swipe) | 38 | 12 | 12 | 62 |
| FETA iPhone X (same users for scroll and swipe) | 38 | 12 | 12 | 62 |
| FETA iPhone 6s (same users for scroll and swipe) | 36 | 12 | 12 | 60 |
| HMOG reading sitting | 33 | 11 | 12 | 56 |
| HMOG reading walking | 46 | 16 | 15 | 77 |
| HMOG reading combined (reserved for future) | 32 | 11 | 11 | 54 |

FETA includes 184 unique users; HMOG has 79 unique users in the union of its two reading tasks (54 both, 2 sitting-only, 23 walking-only). Split assignments are aligned across tasks. Future combined HMOG must use its *existing* shared subjects only, not form a new split.

## Reproducing and verifying locally

Run the two Python scripts from the **root** of the research project, which contains:

- `output_data/hmog_feta_feta_criterion_v2_0/summary.json`
- `output_data/hmog_feta_feta_criterion_v2_0/eligible_user_ids_by_threshold.csv`
- `output_data/hmog_feta_feta_criterion_v2_0/session_task_counts.csv`
- `output_data/hmog_feta_feta_criterion_v2_0/first_four_session_manifest.csv`
- `output_data/hmog_feta_feta_criterion_v2_0/task_intersection.csv`
- `input_data/tables/tables/userdata.csv`

### Validate a file committed to Git

```bash
python validate_hmog_feta_splits_v1.py --splits experiments/splits/hmog_feta_splits_seed_20261112.json --audit-dir output_data/hmog_feta_feta_criterion_v2_0 --userdata input_data/tables/tables/userdata.csv
```

### Recompute from the raw data (optional, slower)

```bash
python validate_hmog_feta_splits_v1.py \
  --splits experiments/splits/hmog_feta_splits_seed_20261112.json \
  --audit-dir output_data/hmog_feta_feta_criterion_v2_0 \
  --raw-check \
  --hmog-input input_data/hmog_unarchived \
  --feta-input input_data/data_files \
  --v2-script analyze_hmog_feta_feta_criterion_v2_0.py
```

Update `--hmog-input` if your HMOG folder is elsewhere. Raw-check uses the v2.0 extraction functions, so it is a reproducibility check, not a fully independent implementation of the gesture criterion. It takes considerably longer than the basic CSV check.

### Regeneration (only before the split is frozen for model development)

```bash
python make_hmog_feta_splits_v1.py --audit-dir output_data/hmog_feta_feta_criterion_v2_0 --output output_data/hmog_feta_splits_seed_20261112-1.json --seed 20261112 --candidates 1500
```

The deterministic generator chooses the best of 1,500 precommitted random subject allocations with fixed group sizes. The balancing objective considers **only the number of strictly eligible gestures in sessions 1 and 2** of each task (log-transformed); the later two sessions are not used to select the split. There is no model-based score selection or cross-validation.

Do not regenerate and replace the committed split after inspecting validation/test model results. The same frozen JSON must be shared by specialized and global encoder experiments. For a new unknown user, sessions 1–2 are for enrollment and sessions 3–4 for genuine verification. For globally trained encoders, combine *training* users only; keep validation/test user identities disjoint from training in every scenario.

## Verification limitation

The attached JSON was checked against the uploaded audit summaries and first-four per-session counts, and the FETA model mapping was cross-checked using the previously uploaded mapping CSV. The full FETA/HMOG raw archives and the original `userdata.csv` are not included in this answer, so we could **not** independently reprocess the raw sensor and orientation records in this environment. Run `--raw-check` and the default userdata check on your machine for additional assurance. If audit output files are regenerated, source SHA-256 hashes must match or the validator will deliberately fail.
