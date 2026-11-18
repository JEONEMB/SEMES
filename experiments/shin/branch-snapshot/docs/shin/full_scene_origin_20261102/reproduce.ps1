$ErrorActionPreference = 'Stop'
$sceneWorkspace = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
$scenePython = 'C:\Users\SSAFY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$sceneCmake = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$env:PYTHONPATH = Join-Path $sceneWorkspace '.analysis_packages'
$env:OPENBLAS_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
& $sceneCmake -S $PSScriptRoot -B (Join-Path $PSScriptRoot 'build') -G 'Visual Studio 18 2026' -A x64 -T v141
if ($LASTEXITCODE -ne 0) { throw 'CMake configure failed' }
& $sceneCmake --build (Join-Path $PSScriptRoot 'build') --config Release --parallel 1
if ($LASTEXITCODE -ne 0) { throw 'CMake build failed' }
foreach ($sceneScript in @('extract_scene.py','quality_scene.py','model_scene.py','report_quality.py','annotate_features.py')) {
    & $scenePython (Join-Path $PSScriptRoot $sceneScript)
    if ($LASTEXITCODE -ne 0) { throw ('Failed: ' + $sceneScript) }
}
