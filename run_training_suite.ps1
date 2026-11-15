# Sequential training. Skips FINISHED runs, resumes interrupted epochs.
param(
    [string[]]$Models = @("feta_x_scroll", "global"),
    [int]$Epochs = 30,
    [int]$StepsPerEpoch = 120
)
$ErrorActionPreference = "Stop"
foreach ($model in $Models) {
    $folder = Join-Path 'experiments/results' $model
    $complete = Join-Path $folder 'training_finished.json'
    $last = Join-Path $folder 'checkpoint_last.pt'
    if (Test-Path $complete) {
        Write-Host "[SKIP] $model already completed"
        continue
    }
    $arguments = @('train_gesture_encoder.py', '--scenario', $model,
                   '--epochs', "$Epochs", '--steps-per-epoch', "$StepsPerEpoch")
    if (Test-Path $last) {$arguments += '--resume'}
    Write-Host "[START] $model"
    & python @arguments
    if ($LASTEXITCODE -ne 0) {throw "Training failed: $model (exit $LASTEXITCODE)"}
}
