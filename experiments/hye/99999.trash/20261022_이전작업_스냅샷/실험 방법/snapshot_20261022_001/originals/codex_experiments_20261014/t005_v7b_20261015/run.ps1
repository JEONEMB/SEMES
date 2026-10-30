param([ValidateSet('all','experiment','verify','report')][string]$Action='all')
$ErrorActionPreference='Stop'
$env:PYTHONPATH=Join-Path (Split-Path $PSScriptRoot -Parent) 'deps'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
$taskPython='C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if ($Action -eq 'all' -or $Action -eq 'experiment') {
 & $taskPython (Join-Path $PSScriptRoot 'experiment.py')
 if ($LASTEXITCODE -ne 0) { throw 'Comparison failed.' }
}
if ($Action -eq 'all' -or $Action -eq 'verify') {
 & $taskPython (Join-Path $PSScriptRoot 'verify.py')
 if ($LASTEXITCODE -ne 0) { throw 'Verification failed.' }
}
if ($Action -eq 'all' -or $Action -eq 'report') {
 & $taskPython (Join-Path $PSScriptRoot 'analyze.py')
 if ($LASTEXITCODE -ne 0) { throw 'Analysis failed.' }
}
