param(
    [ValidateSet('predict','reproduce','tests','report','evaluate-all')][string]$Action = 'predict',
    [string]$Center,
    [string]$Current,
    [ValidateSet('calibrated-runs','general')][string]$Mode = 'calibrated-runs',
    [string]$Output,
    [string]$Python = 'C:\Users\SSAFY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
)
$ErrorActionPreference = 'Stop'
$env:PYTHONPATH = Join-Path $PSScriptRoot 'deps'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($Action -eq 'predict') {
    if (!$Center -or !$Current) { throw 'Provide -Center and -Current paths.' }
    $cliArgs = @((Join-Path $PSScriptRoot 'infer.py'), '--center', $Center, '--current', $Current, '--mode', $Mode)
    if ($Output) { $cliArgs += @('--out', $Output) }
    & $Python @cliArgs
} elseif ($Action -eq 'evaluate-all') {
    & $Python (Join-Path $PSScriptRoot 'full280/evaluate_all.py')
} elseif ($Action -eq 'tests') {
    & $Python (Join-Path $PSScriptRoot 'verify_delivery.py')
} elseif ($Action -eq 'report') {
    & $Python (Join-Path $PSScriptRoot 'make_report.py')
} else {
    foreach ($file in @('reproduce_measurements.py','zero_experiment.py','validate_calibrated_runs.py','modeling/unified.py','package_reference.py','verify_delivery.py','make_report.py')) {
        & $Python (Join-Path $PSScriptRoot $file)
        if ($LASTEXITCODE -ne 0) { throw "Experiment failed: $file" }
        if ($file -eq 'zero_experiment.py') {
            & $Python (Join-Path $PSScriptRoot $file) --measurement gn17_spread14
            if ($LASTEXITCODE -ne 0) { throw 'QR205 zero experiment failed.' }
        }
    }
}
if ($LASTEXITCODE -ne 0) { throw "Python exited with $LASTEXITCODE" }
