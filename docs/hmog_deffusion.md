# DeFFusion / HMOG reproduction (Python)

This project is a from-paper reconstruction of:

Y. Li, P. Tao, S. Deng, G. Zhou, **"DeFFusion: CNN-based Continuous Authentication Using Deep Feature Fusion"**, ACM Transactions on Sensor Networks, DOI 10.1145/3485060.

The instructions below assume:

- the repository/project directory is the current PowerShell working directory;
- the unarchived HMOG dataset is located at `../input_data/hmog_unarchived`;
- generated intermediate data and results are stored in `../deffusion_work`;
- commands are executed in **PowerShell**.

All commands below are intentionally written on one line so they can be copied directly into PowerShell.

## Pipeline implemented

The implementation follows the paper's reported pipeline:

1. Use accelerometer and gyroscope data.
2. Synchronize the two streams and resample to 100 Hz.
3. Take the first 100 minutes of usable motion data per selected user.
4. Create non-overlapping authentication windows; the principal experiment uses 5-second windows.
5. Construct a `6 x n` time-domain matrix.
6. Apply FFT to every sensor-axis row and concatenate its real and imaginary components, producing a `6 x 2n` frequency-domain matrix.
7. Standardize every row independently.
8. For a 5-second window, reshape the time representation to `20 x 150` and the frequency representation to `20 x 300`.
9. Train separate time-domain and frequency-domain CNNs with SGD and cross-entropy loss.
10. Extract deep features from each CNN.
11. Apply Factor Analysis separately to the time- and frequency-domain deep features.
12. For the 5-second experiment, fuse 28 time factors and 22 frequency factors into 50 features.
13. Train/evaluate an RBF One-Class SVM per legitimate user with repeated cross-validation.
14. Report EER, FAR, and FRR per user and in aggregate.

The script also includes training-size sweeps, time-window sweeps, the 90 hand-designed-feature baseline, and kNN / Random Forest / Decision Tree comparisons.

## Progress and timing logs

The script logs progress with `print()` and now measures wall-clock time for the main stages of the pipeline. Typical output includes entries such as:

```text
[stage] START  Train time-domain CNN
[train:time] epoch 001/150 DONE loss=... acc=... epoch_time=... total_time=...
[stage] DONE   Train time-domain CNN | elapsed=...
[stage] START  Extract time-domain deep features
[stage] DONE   Extract time-domain deep features | elapsed=...
[stage] START  Factor Analysis and time/frequency feature fusion
[stage] DONE   Factor Analysis and time/frequency feature fusion | elapsed=...
[stage] START  Per-user repeated OC-SVM cross-validation
[evaluate:100669] user total: ...
[stage] DONE   Per-user repeated OC-SVM cross-validation | elapsed=...
[total] Command 'evaluate' completed in ...
```

The `prepare` command also reports time per prepared user and total preprocessing time. If a stage fails, its elapsed time is printed before the exception is reported.

## Important reproducibility gaps in the paper

The publication does not specify enough implementation detail to guarantee bit-for-bit or metric-for-metric reproduction. This implementation makes the consequential ambiguities explicit.

- The identities of the 95 selected HMOG participants are not listed. By default, the code chooses the first 95 users sorted by ID that have at least 100 minutes of usable synchronized accelerometer/gyroscope data. `--users-file` can be used if an exact participant list becomes available.
- The exact accelerometer/gyroscope synchronization algorithm is not reported. This implementation linearly interpolates both streams onto a common 100 Hz grid over their overlapping timestamps, session by session.
- The CNN architecture table says Conv2 uses a `1x1` kernel but also reports a spatial height change that a `1x1` convolution cannot produce. The default is `--conv2-mode one_by_one`, following the explicitly stated `1x1` kernel. `--conv2-mode table_shape` remains available for testing the alternative interpretation.
- The paper states that the CNN-learned feature count is 95 but does not explicitly show the final 95-dimensional feature layer in its architecture table. The default uses the 95 class logits as deep features via `--feature-source logits`.
- The exact Factor Analysis implementation is not specified. This code uses `sklearn.decomposition.FactorAnalysis`.
- OC-SVM `nu` and `gamma` are not reported. Defaults are `nu=0.01` and `gamma=scale`.
- kNN, Random Forest, and Decision Tree hyperparameters and random seeds are not reported; the reconstruction uses explicit conventional scikit-learn settings.
- The learning-rate description is internally inconsistent: the displayed formula uses `step_size=30`, while the prose describes reduction every 50 epochs. The default follows the prose with `--lr-step 50`; `--lr-step 30` can be used to test the other interpretation.
- For 5-second windows, Table 5 reports mean EER 1.29%, median 1.00%, and SD 1.03%, while Table 7 reports EER 1.00%. The script reports mean, median, and SD separately.

## Installation

Python 3.10+ is recommended.

Create the virtual environment:

```powershell
python -m venv .venv
```

Activate it in PowerShell:

```powershell
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the appropriate CUDA-enabled PyTorch build for your system before running the full experiment.

## 1. Inspect the HMOG dataset

Run a recursive dataset scan and probe the first sensor pair:

```powershell
python hmog_deffusion.py inspect --data-root "./input_data/hmog_unarchived" --probe
```

The loader searches recursively for session folders containing `Accelerometer.csv` and `Gyroscope.csv`, case-insensitively. Headered and headerless HMOG CSVs are supported.

## 2. Prepare the first 100 minutes for up to 95 users

```powershell
python hmog_deffusion.py prepare --data-root "./input_data/hmog_unarchived" --work-dir "./deffusion_work" --minutes 100 --max-users 95
```

This creates memory-mappable `.npy` files containing synchronized six-axis sensor streams and writes `../deffusion_work/manifest.json`. Authentication windows never cross a session boundary.

You normally need to run `prepare` only once. Subsequent training runs reuse these prepared files.

## 3. Main 5-second CNN training and feature-fusion experiment

Run the complete time-CNN, frequency-CNN, deep-feature extraction, Factor Analysis, and feature-fusion pipeline:

```powershell
python hmog_deffusion.py train --work-dir "./deffusion_work" --window 5 --epochs 150 --batch-size 256 --lr 0.001 --lr-step 50 --time-factors 28 --freq-factors 22
```

If models or feature files already exist, they are reused. To deliberately retrain and overwrite cached training artifacts, add `--force`:

```powershell
python hmog_deffusion.py train --work-dir "./deffusion_work" --window 5 --epochs 150 --batch-size 256 --lr 0.001 --lr-step 50 --time-factors 28 --freq-factors 22 --force
```

During CNN training, every epoch prints loss, classification accuracy, current learning rate, that epoch's duration, and cumulative CNN training time. `--print-every` controls additional within-epoch batch progress messages.

## 4. Run the One-Class SVM authentication evaluation

```powershell
python hmog_deffusion.py evaluate --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20 --positive-limit 1000 --ocsvm-nu 0.01 --ocsvm-gamma scale
```

The evaluator logs every user's progress, each repetition's EER/FAR/FRR, the total evaluation time for that user, the duration of the complete OC-SVM stage, and the total command duration.

Results are written to `../deffusion_work/results` as CSV and JSON files in addition to being printed to the console.

## Paper targets for the 5-second experiment

Table 5 reports:

- EER mean: **1.29%**
- EER median: **1.00%**
- EER SD: **1.03%**

Table 7 reports DeFFusion:

- EER: **1.00%** (SD 1.03)
- FAR: **1.42%** (SD 1.26)
- FRR: **0.75%** (SD 0.99)

The discrepancy between Table 5 and Table 7 is in the paper itself.

## Fast smoke test before a full 150-epoch run

After `prepare`, train on a deliberately small number of windows for two epochs:

```powershell
python hmog_deffusion.py train --work-dir "./deffusion_work" --window 5 --epochs 2 --max-windows-per-user 20 --batch-size 64 --force
```

Then exercise the evaluation pipeline with a small protocol:

```powershell
python hmog_deffusion.py evaluate --work-dir "./deffusion_work" --window 5 --folds 2 --repeats 1 --positive-limit 20
```

This verifies that the pipeline executes; its EER is not scientifically meaningful.

**Important:** because `--force` overwrites the normal `w5s` model/features, rerun the full `train` command afterward before performing the full evaluation.

## Training-set-size sweep (Table 4 style)

This requires completed 5-second fused features:

```powershell
python hmog_deffusion.py training-size-sweep --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20
```

Paper EER targets:

| Positive samples | Mean | Median | SD |
|---:|---:|---:|---:|
| 100 | 2.13 | 1.25 | 2.35 |
| 200 | 2.34 | 1.75 | 2.19 |
| 300 | 1.87 | 1.75 | 1.46 |
| 400 | 2.45 | 2.43 | 1.75 |
| 500 | 2.20 | 2.20 | 1.42 |
| 600 | 1.89 | 1.71 | 1.27 |
| 700 | 2.08 | 1.75 | 1.46 |
| 800 | 2.03 | 1.72 | 1.44 |
| 900 | 2.01 | 1.78 | 1.29 |
| 1000 | 1.86 | 1.60 | 1.29 |

## Time-window sweep (Table 5 style)

This retrains two CNNs for each requested window size and is computationally expensive:

```powershell
python hmog_deffusion.py window-sweep --work-dir "./deffusion_work" --windows 1 2 3 4 5 6 7 8 9 10 --epochs 150 --folds 10 --repeats 20
```

Paper EER targets:

| Window | Mean | Median | SD |
|---:|---:|---:|---:|
| 1 s | 2.74 | 2.53 | 1.68 |
| 2 s | 1.85 | 1.67 | 1.18 |
| 3 s | 1.68 | 1.55 | 1.08 |
| 4 s | 1.64 | 1.47 | 1.05 |
| 5 s | 1.29 | 1.00 | 1.03 |
| 6 s | 1.35 | 1.23 | 0.87 |
| 7 s | 1.11 | 0.95 | 0.87 |
| 8 s | 1.23 | 1.18 | 0.77 |
| 9 s | 1.08 | 0.93 | 0.84 |
| 10 s | 1.11 | 1.00 | 0.79 |

For 2-second windows, the paper reports 29 time factors + 21 frequency factors. For 5-second windows it reports 28 + 22. It does not provide optimal factor balances for every other window size, so the implementation uses 28 + 22 unless overridden internally by the sweep logic.

## Classifier comparison using the same CNN features (Table 8 style)

```powershell
python hmog_deffusion.py classifier-baselines --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20
```

## Hand-designed 90-feature baseline (Table 6/7 style)

```powershell
python hmog_deffusion.py designed-baseline --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20
```

The exact definitions of spectral energy, entropy, and peak handling are not fully specified by the paper. The implementation uses documented standard FFT-power definitions.

## Laptop runtime notes

The full protocol is computationally heavy: approximately 95 users, roughly 1,200 five-second windows per user, two CNNs for 150 epochs, deep-feature extraction, Factor Analysis, and then many OC-SVM fits from 10-fold cross-validation repeated 20 times.

The script automatically chooses CUDA, Apple MPS, or CPU unless `--device` is supplied. On Windows, `--workers 0` is the safest DataLoader setting and is the default.

Intermediate files are cached under `../deffusion_work`, so completed stages can normally be reused instead of recomputed.
