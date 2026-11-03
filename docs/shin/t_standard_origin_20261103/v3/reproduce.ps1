$ErrorActionPreference='Stop'
$v3Workspace=Split-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) -Parent
$v3Python='C:\Users\SSAFY\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\python.exe'
$env:PYTHONPATH=Join-Path $v3Workspace '.analysis_packages'
$env:OPENBLAS_NUM_THREADS='1'
$env:OMP_NUM_THREADS='1'
& $v3Python (Join-Path $PSScriptRoot 'prepare_v3.py')
if($LASTEXITCODE -ne 0){throw 'Origin preparation failed'}
& $v3Python (Join-Path $PSScriptRoot 'build_sources.py')
cmake -S $PSScriptRoot -B (Join-Path $PSScriptRoot 'build') -G 'Visual Studio 18 2026' -A x64 -T v141 -DOpenCV_DIR=C:/opencv/build
cmake --build (Join-Path $PSScriptRoot 'build') --config Release --parallel 1
if($LASTEXITCODE -ne 0){throw 'Build failed'}
$v3Legacy=Join-Path $v3Workspace 'qrwithbar-lanczos-src/experiments/hye/20261102_173432_t_six_methods/results/models'
$v3Variants=@('original','corrected','fixed_v3','crossfit','prior_half','prior_double','bound_double')
foreach($v3Name in $v3Variants){
 & (Join-Path $PSScriptRoot 'build/Release/v3_train.exe') (Join-Path $PSScriptRoot $v3Name) $v3Legacy
 if($LASTEXITCODE -ne 0){throw ('Training failed: '+$v3Name)}
}
& $v3Python (Join-Path $PSScriptRoot 'scene_estimator.py') --train $v3Variants
if($LASTEXITCODE -ne 0){throw 'Scene model training failed'}
& $v3Python (Join-Path $PSScriptRoot 'verify_and_package.py')
if($LASTEXITCODE -ne 0){throw 'Verification failed'}
& (Join-Path $PSScriptRoot 'build/Release/v3_calibration_control.exe') $PSScriptRoot
if($LASTEXITCODE -ne 0){throw 'Calibration control failed'}
& $v3Python (Join-Path $PSScriptRoot 'report.py')
if($LASTEXITCODE -ne 0){throw 'Report failed'}
