# DeFFusion / HMOG reproduction (Python)

This project is a from-paper reconstruction of:

Y. Li, P. Tao, S. Deng, G. Zhou, **"DeFFusion: CNN-based Continuous Authentication Using Deep Feature Fusion"**, ACM Transactions on Sensor Networks, DOI 10.1145/3485060.

It is designed to run directly against an unarchived HMOG dataset and prints progress to the console with `print()`.

## What is reproduced

The implementation follows the paper's reported pipeline:

1. Use accelerometer + gyroscope only.
2. Synchronize the two streams and resample to 100 Hz.
3. Take the first 100 minutes of usable motion data per user.
4. Create non-overlapping windows (5 s for the main experiment; 1--10 s supported).
5. Construct a 6 x n time-domain matrix.
6. FFT each row; concatenate real and imaginary parts into a 12 x n frequency-domain matrix.
7. Standardize every row independently.
8. Reshape to 20 x 150 for a 5 s time window and 20 x 300 for a 5 s frequency window.
9. Train two CNNs (time and frequency domains) with batch size 256, SGD, cross entropy, initial LR 0.001, up to 150 epochs.
10. Extract 95-dimensional deep features from each CNN.
11. Apply Factor Analysis separately to the two 95-dimensional feature vectors.
12. For 5 s windows concatenate 28 time-domain factors + 22 frequency-domain factors = 50 features.
13. Train/evaluate a radial-basis One-Class SVM with 10-fold cross-validation and balanced genuine/impostor test samples, repeated 20 times.
14. Report per-user and aggregate EER, FAR, and FRR.

The script also implements:

- Table-4-style training-set-size sweep (100...1000 samples).
- Table-5-style time-window sweep (1...10 seconds; retrains CNNs for each window size, so it is expensive).
- The 90 hand-designed-feature baseline from Table 6.
- kNN / Random Forest / Decision Tree comparisons on CNN features.

## Important reproducibility gaps in the paper

The paper does not specify enough information to guarantee bit-for-bit or metric-for-metric reproduction. This implementation makes every material gap explicit instead of silently pretending an exact reproduction is possible.

- The identities of the 95 selected HMOG participants are not listed. By default this code chooses the first 95 users (sorted by ID) that have at least 100 minutes of usable synchronized accelerometer+gyroscope data. Use `--users-file` to supply a known 95-user list.
- The exact accelerometer/gyroscope synchronization algorithm is not reported. This code linearly interpolates both streams onto a common 100 Hz grid over their overlapping timestamps, session by session.
- The CNN architecture table reports Conv2 as a 1x1 convolution but simultaneously reports its height changing from 20 to 6, which a 1x1 convolution cannot do. The default `table_shape` reconstruction uses a 15x1 Conv2 so that the published tensor shape is reproduced. You can run `--conv2-mode one_by_one` to follow the stated 1x1 kernel instead.
- The paper says the CNN-learned feature number is 95, but the architecture table ends at FC2=512 and does not explicitly show the final 95-class layer. The default uses the 95 class logits as the reported 95 deep features. This is exposed as `--feature-source logits`.
- The exact Factor Analysis implementation is not specified. This code uses `sklearn.decomposition.FactorAnalysis` and treats its latent factors as the selected features.
- OC-SVM `nu` and `gamma` are not reported. Defaults are `nu=0.01`, `gamma=scale`; both are command-line options.
- kNN/RF/DT hyperparameters and random seeds are not reported; conventional sklearn defaults (made explicit in code) are used.
- The CNN learning-rate description is internally inconsistent: the displayed formula says `step_size=30`, while the prose says the LR is reduced by 90% every 50 epochs. The default follows the prose (`--lr-step 50`); use `--lr-step 30` to test the formula.
- The paper's abstract says "mean EER = 1.00%" at 5 s, but Table 5 reports **mean 1.29%, median 1.00%, SD 1.03%**. This program always prints mean, median, and SD separately.

Because of these gaps, expect a *faithful reconstruction*, not a guarantee of exactly 1.00%. The command-line switches let you test the plausible interpretations systematically.

## Installation

Python 3.10+ is recommended.

```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

If you have an NVIDIA GPU, install the CUDA-enabled PyTorch build suitable for your system before running the full 150-epoch experiments.

## Quick dataset inspection

```bash
python hmog_deffusion.py inspect --data-root "D:/HMOG"
```

The loader searches recursively for session folders containing `Accelerometer.csv` and `Gyroscope.csv` (case-insensitive). It supports both HMOG CSVs with headers and headerless CSVs.

## Prepare the first 100 minutes for 95 users

```bash
python hmog_deffusion.py prepare --data-root "./input_data/hmog_unarchived" --work-dir "./deffusion_work" --minutes 100 --max-users 95
```

This creates memory-mappable `.npy` files containing synchronized six-axis sensor streams. It never combines samples across a session boundary when forming authentication windows.

## Main 5-second experiment

First train the time- and frequency-domain CNNs and extract/fuse the deep features:

```bash
python hmog_deffusion.py train --work-dir "./deffusion_work" --window 5 --epochs 150 --batch-size 256 --lr 0.001 --lr-step 50 --time-factors 28 --freq-factors 22
```

Then run the OC-SVM protocol:

```bash
python hmog_deffusion.py evaluate --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20 --positive-limit 1000 --ocsvm-nu 0.01 --ocsvm-gamma scale
```

Results are written to `deffusion_work/results/` as CSV and JSON in addition to being printed.

### Paper target for the 5 s experiment

Table 5 reports:

- EER mean: **1.29%**
- EER median: **1.00%**
- EER SD: **1.03%**

Table 7 reports DeFFusion:

- EER: **1.00%** (SD 1.03)
- FAR: **1.42%** (SD 1.26)
- FRR: **0.75%** (SD 0.99)

The discrepancy between Table 5 and Table 7 is in the paper itself.

## Training-set-size sweep (Table 4 style)

Requires a completed 5-second `train` stage.

```bash
python hmog_deffusion.py training-size-sweep --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20
```

Paper EER targets (mean / median / SD):

| Positive samples | Mean | Median | SD |
|---:|---:|---:|---:|
|100|2.13|1.25|2.35|
|200|2.34|1.75|2.19|
|300|1.87|1.75|1.46|
|400|2.45|2.43|1.75|
|500|2.20|2.20|1.42|
|600|1.89|1.71|1.27|
|700|2.08|1.75|1.46|
|800|2.03|1.72|1.44|
|900|2.01|1.78|1.29|
|1000|1.86|1.60|1.29|

## Time-window sweep (Table 5 style)

This retrains two CNNs for every window size. On a laptop CPU it can take a very long time. Start with fewer epochs as a smoke test.

```bash
python hmog_deffusion.py window-sweep --work-dir "./deffusion_work" --windows 1 2 3 4 5 6 7 8 9 10 --epochs 150 --folds 10 --repeats 20
```

Paper EER targets (mean / median / SD):

| Window | Mean | Median | SD |
|---:|---:|---:|---:|
|1s|2.74|2.53|1.68|
|2s|1.85|1.67|1.18|
|3s|1.68|1.55|1.08|
|4s|1.64|1.47|1.05|
|5s|1.29|1.00|1.03|
|6s|1.35|1.23|0.87|
|7s|1.11|0.95|0.87|
|8s|1.23|1.18|0.77|
|9s|1.08|0.93|0.84|
|10s|1.11|1.00|0.79|

For 2 s the paper reports a balanced fusion of 29 time factors + 21 frequency factors. For 5 s it reports 28+22. It does not give optimal balances for all other window sizes, so this implementation uses 28+22 for them unless overridden.

## Classifier comparison using the same CNN features (Table 8 style)

```bash
python hmog_deffusion.py classifier-baselines --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20
```

## Hand-designed 90-feature baseline (Table 6/7 style)

```bash
python hmog_deffusion.py designed-baseline --work-dir "./deffusion_work" --window 5 --folds 10 --repeats 20
```

The exact definitions of spectral energy/entropy/peak handling are not fully specified in the paper. The implementation uses standard FFT-power definitions and documents them in code.

## Fast smoke test before a full run

After `prepare`, run:

```bash
python hmog_deffusion.py train --work-dir "./deffusion_work" --window 5 --epochs 2 --max-windows-per-user 20 --batch-size 64

python hmog_deffusion.py evaluate --work-dir "./deffusion_work" --window 5 --folds 2 --repeats 1 --positive-limit 20
```

This only verifies the pipeline. Its EER is not meaningful.

## Laptop runtime notes

The full paper protocol is computationally heavy: 95 users x roughly 1200 five-second windows, two CNNs, 150 epochs, then thousands of OC-SVM fits from 10-fold CV repeated 20 times. A discrete GPU is strongly recommended for CNN training. The code automatically selects CUDA, Apple MPS, or CPU unless `--device` is supplied.

The most important caches are stored under the work directory, so interrupted experiments can be resumed without rereading all HMOG CSVs.
