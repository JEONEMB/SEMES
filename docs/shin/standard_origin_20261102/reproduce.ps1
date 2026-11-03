$ErrorActionPreference = 'Stop'
$originWorkspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$originPython = 'C:\Users\SSAFY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = Join-Path $originWorkspace '.analysis_packages'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
foreach ($originScript in @('fit_origin.py','validate_origin.py','make_report.py')) {
    & $originPython (Join-Path $PSScriptRoot $originScript)
    if ($LASTEXITCODE -ne 0) { throw ('Failed: ' + $originScript) }
}
