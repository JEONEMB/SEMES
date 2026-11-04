param(
    [ValidateSet('compare','report','predict')][string]$Action='compare',
    [string]$Center,[string]$Current,[string]$Method='spatial_gls_lw'
)
$ErrorActionPreference='Stop'
$env:PYTHONPATH=Join-Path (Split-Path $PSScriptRoot -Parent) 'deps'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
$env:PYTHONIOENCODING='utf-8'
$taskPython='C:/Users/SSAFY/.cache/codex-runtimes/codex-primary-runtime/dependencies/python/python.exe'
if ($Action -eq 'compare') {
    & $taskPython (Join-Path $PSScriptRoot 'compare.py')
    if ($LASTEXITCODE -ne 0) { throw 'Primary experiment failed.' }
    & $taskPython (Join-Path $PSScriptRoot 'direct410/run_direct410.py')
    if ($LASTEXITCODE -ne 0) { throw 'Direct experiment failed.' }
    & $taskPython (Join-Path $PSScriptRoot 'build_report.py')
    if ($LASTEXITCODE -ne 0) { throw 'Report failed.' }
    & $taskPython (Join-Path $PSScriptRoot 'figures/build_figures.py')
    if ($LASTEXITCODE -ne 0) { throw 'Figures failed.' }
    & $taskPython (Join-Path $PSScriptRoot 'verify_raw_inference.py')
} elseif ($Action -eq 'report') {
    & $taskPython (Join-Path $PSScriptRoot 'build_report.py')
} else {
    if (!$Center -or !$Current) { throw 'Center and Current paths are required.' }
    & $taskPython (Join-Path $PSScriptRoot 'predict.py') --center $Center --current $Current --method $Method
}
if ($LASTEXITCODE -ne 0) { throw 'Python command failed.' }
