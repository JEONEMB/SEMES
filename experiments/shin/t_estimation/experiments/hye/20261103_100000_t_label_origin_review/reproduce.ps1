$ErrorActionPreference='Stop'
$experiment=$PSScriptRoot
$repository=Split-Path (Split-Path (Split-Path $experiment -Parent) -Parent) -Parent
$oldExperiment=Join-Path (Split-Path $experiment -Parent) '20261102_173432_t_six_methods'
$exe=Join-Path $repository 'build-t-only/experiments/hye/20261103_100000_t_label_origin_review/Release/t_label_review.exe'
if(!(Test-Path -LiteralPath $exe)){throw '먼저 README에 따라 t_label_review Release 대상을 빌드하세요.'}
foreach($mode in @('raw','standard','group','all')){
    & $exe $oldExperiment $experiment (Join-Path $experiment 'original_audit_coordinates.csv') $mode
    if($LASTEXITCODE -ne 0){throw "평가 실패: $mode"}
}
