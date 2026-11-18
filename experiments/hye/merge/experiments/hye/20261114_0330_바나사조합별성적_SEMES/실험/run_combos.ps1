# 바·나사 조합별 성적: final00_tool eval / range 를 조합마다 돌리고 결과 줄을 표로 모은다 (계산은 모두 C++ final00_tool).
# 사용: powershell -File run_combos.ps1 <final00_tool.exe> <저장소 .batch_build 폴더> <출력 폴더(ASCII)>
param([string]$Tool, [string]$Batch, [string]$Out)
$ErrorActionPreference = 'Stop'
$labels = "$Batch\semes_new\in\labels.tsv"; $splits = "$Batch\semes_new\in\splits.tsv"
$features = [ordered]@{ ncc = "$Batch\b2\semes\features7.tsv"; track536 = "$Batch\healthcpp\semes536_features7.tsv" }
# 이름, 값 번호(0 QR, 1 위 바, 2 아래 바, 3~6 나사 1~4)
$combos = [ordered]@{
  'bar0_pin4'='0,3,4,5,6'; 'barU_pin4'='0,1,3,4,5,6'; 'barL_pin4'='0,2,3,4,5,6'; 'bar2_pin4'='0,1,2,3,4,5,6'
  'pin3_no1'='0,4,5,6'; 'pin3_no2'='0,3,5,6'; 'pin3_no3'='0,3,4,6'; 'pin3_no4'='0,3,4,5'
  'pin2_left13'='0,3,5'; 'pin2_right24'='0,4,6'; 'pin2_top12'='0,3,4'; 'pin2_bottom34'='0,5,6'; 'pin2_diag14'='0,3,6'; 'pin2_diag23'='0,4,5'
  'pin1_1'='0,3'; 'pin1_2'='0,4'; 'pin1_3'='0,5'; 'pin1_4'='0,6'
  'qr_only'='0'; 'bar2_pin0'='0,1,2'
}
New-Item -ItemType Directory -Force $Out | Out-Null
$rows = foreach ($m in $features.Keys) { foreach ($c in $combos.Keys) {
  $d = "$Out\$m\$c"; if (Test-Path $d) { Remove-Item -Recurse -Force $d }
  & $Tool eval $features[$m] $labels $splits "$d\eval" "use=$($combos[$c])" | Out-Null
  & $Tool range $features[$m] $labels $splits "$d\range" "use=$($combos[$c])" | Out-Null
  $e = Get-Content "$d\eval\eval.txt"; $r = Get-Content "$d\range\range.txt"
  $main = [regex]::Match(($e | Select-String '^main_cv_fails').Line, 'mean ([\d.]+) worst (\d+) rmsY ([\d.]+) rmsT ([\d.]+)')
  $half = [regex]::Match(($e | Select-String '^half_fails').Line, 'mean ([\d.]+)')
  $leave = [regex]::Match(($e | Select-String '^leave_one_session').Line, 'all ([\d.]+)%')
  $fin = [regex]::Match(($e | Select-String '^final_check').Line, 'fails (\d+)/(\d+)')
  $outer = [regex]::Match(($r | Select-String '^outer n').Line, 'fails (\d+) rmsY ([\d.]+) rmsT ([\d.]+)')
  [pscustomobject]@{ qr = $m; combo = $c; use = $combos[$c]; main = $main.Groups[1].Value; rmsY = $main.Groups[3].Value; rmsT = $main.Groups[4].Value
    half = $half.Groups[1].Value; unseen = $leave.Groups[1].Value; final = $fin.Groups[1].Value; outer_fail = $outer.Groups[1].Value; outer_rmsY = $outer.Groups[2].Value; outer_rmsT = $outer.Groups[3].Value }
} }
$rows | Export-Csv -Delimiter "`t" -NoTypeInformation -Encoding UTF8 "$Out\combo_scores.tsv"
$rows | Format-Table -AutoSize | Out-String -Width 220
