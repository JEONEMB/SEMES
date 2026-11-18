param([ValidateSet('all','prepare','run','verify','report','predict')][string]$Action='all',[string]$Center,[string]$Current)
$ErrorActionPreference='Stop'
$env:PYTHONPATH=Join-Path (Split-Path $PSScriptRoot -Parent) 'deps'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
$taskPython='C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if ($Action -eq 'all' -or $Action -eq 'prepare') {
 & $taskPython (Join-Path $PSScriptRoot 'experiment.py') prepare
 if ($LASTEXITCODE -ne 0) { throw 'Prepare failed.' }
}
if ($Action -eq 'all' -or $Action -eq 'run') {
 & $taskPython (Join-Path $PSScriptRoot 'experiment.py') run
 if ($LASTEXITCODE -ne 0) { throw 'Experiment failed.' }
}
if ($Action -eq 'all' -or $Action -eq 'verify') {
 & $taskPython (Join-Path $PSScriptRoot 'verify.py')
 if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
}
if ($Action -eq 'all' -or $Action -eq 'report') {
 & $taskPython (Join-Path $PSScriptRoot 'report.py')
 if ($LASTEXITCODE -ne 0) { throw 'Report failed.' }
}
if ($Action -eq 'predict') {
 if (!$Center -or !$Current) { throw 'Center and Current paths are required.' }
 & $taskPython (Join-Path $PSScriptRoot 'predict.py') --center $Center --current $Current
 if ($LASTEXITCODE -ne 0) { throw 'Prediction failed.' }
}
