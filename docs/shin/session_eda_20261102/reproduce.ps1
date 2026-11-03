param([switch]$Remeasure)
$ErrorActionPreference = 'Stop'
$taskAnalysisDir = $PSScriptRoot
$taskWorkspaceDir = Split-Path (Split-Path $taskAnalysisDir -Parent) -Parent
$taskPython = 'C:\Users\SSAFY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH = Join-Path $taskWorkspaceDir '.analysis_packages'
if ($Remeasure) {
    $taskRepo = Join-Path $taskWorkspaceDir 'qrwithbar-lanczos-src'
    $taskExperiment = Join-Path $taskRepo 'experiments/hye/20261102_173432_t_six_methods'
    & (Join-Path $taskRepo 'build-t-only/experiments/hye/20261102_173432_t_six_methods/Release/six_extract.exe') `
        'C:\Users\SSAFY\Documents\GitHub\S15P31A507\data\SSAFY Test Image Set' `
        (Join-Path $taskRepo 'modules/qrwithbar_lanczos/assets') `
        (Join-Path $taskExperiment 'results/metadata.csv') `
        (Join-Path $taskRepo 'modules/t_estimation/results-raw575/observations.csv') `
        (Join-Path $taskAnalysisDir 'remeasured')
    if ($LASTEXITCODE -ne 0) { throw 'Remeasurement failed' }
}
& $taskPython (Join-Path $taskAnalysisDir 'analyze.py')
if ($LASTEXITCODE -ne 0) { throw 'EDA failed' }
& $taskPython (Join-Path $taskAnalysisDir 'build_report.py')
if ($LASTEXITCODE -ne 0) { throw 'Report failed' }
