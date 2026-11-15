# Run from repository root: .\run_architecture_study.ps1
# Original experiments/results and experiments/tuning are NEVER written.
param(
    [string[]]$Architectures = @('baseline','tcn','attention_pool','crossmodal'),
    [int[]]$Seeds = @(20261112),
    [ValidateSet('supcon','prototypical')][string]$Objective = 'supcon',
    [int]$Epochs = 30,
    [string]$StudyRoot = 'experiments/architecture_study',
    [int]$EvalWorkers = 4
)
$ErrorActionPreference = 'Stop'
$windows = (1..20) -join ','
foreach ($arch in $Architectures) {
    foreach ($seed in $Seeds) {
        $run = "${arch}_${Objective}_seed${seed}"
        $root = Join-Path $StudyRoot $run
        # Do not reuse or overwrite ANY prior experiment output.
        if (Test-Path $root) { throw "Run directory already exists: $root. Change seeds/run root; do not overwrite." }
        Write-Host "`n===== TRAIN $run =====" -ForegroundColor Cyan
        python -u train_architecture_experiment.py `
            --scenario global --architecture $arch --objective $Objective `
            --seed $seed --epochs $Epochs --temperature 0.05 `
            --output-root $root
        if ($LASTEXITCODE -ne 0) { throw "Training failed: $run" }

        Write-Host "`n===== EMBEDDINGS $run =====" -ForegroundColor Cyan
        python -u extract_architecture_embeddings.py `
            --model-id global --role validation `
            --cache output_data/gesture_cache_validation_first_four `
            --results $root
        if ($LASTEXITCODE -ne 0) { throw "Embedding extraction failed: $run" }

        Write-Host "`n===== EVALUATION $run =====" -ForegroundColor Cyan
        python -u evaluate_architecture_run.py `
            --model-id global --results $root `
            --methods centroid,top5 --windows $windows --workers $EvalWorkers
        if ($LASTEXITCODE -ne 0) { throw "Validation evaluation failed: $run" }
    }
}
Write-Host "`n[DONE] Validation-only architecture study. Test not used." -ForegroundColor Green
