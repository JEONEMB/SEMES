param([Parameter(Mandatory=$true)][string]$ResultDirectory)
$predictions=Import-Csv (Join-Path $ResultDirectory 'development_predictions.csv')
$summary=@()
foreach($block in ($predictions | Group-Object mode,cohort)){
 $rows=@($block.Group);$errors=@($rows|ForEach-Object {[double]$_.error_ticks});$absSum=0.;$squareSum=0.;$maximum=0.;$passes=0
 foreach($e in $errors){$a=[math]::Abs($e);$absSum+=$a;$squareSum+=$e*$e;$maximum=[math]::Max($maximum,$a);if($a -le 1){$passes++}}
 $summary+=[pscustomobject]@{mode=$rows[0].mode;cohort=$rows[0].cohort;count=$rows.Count;MAE_ticks=$absSum/$rows.Count;RMSE_ticks=[math]::Sqrt($squareSum/$rows.Count);max_ticks=$maximum;pass_1tick=$passes}
}
$summary | Export-Csv (Join-Path $ResultDirectory 'development_cohort_metrics.csv') -NoTypeInformation -Encoding utf8
$evalPath=Join-Path $ResultDirectory 'evaluation_predictions.csv'
if(Test-Path -LiteralPath $evalPath){Import-Csv $evalPath | Where-Object pass_1tick -eq '0' | Sort-Object {[math]::Abs([double]$_.error_ticks)} -Descending | Export-Csv (Join-Path $ResultDirectory 'evaluation_failures.csv') -NoTypeInformation -Encoding utf8}
$chosenModeLine=Get-Content (Join-Path $ResultDirectory 'freeze.txt') | Where-Object {$_ -like 'selected_mode=*'} | Select-Object -First 1
if($chosenModeLine){$chosenMode=$chosenModeLine.Split('=')[1];$failures=@($predictions | Where-Object {$_.mode -eq $chosenMode -and [math]::Abs([double]$_.error_ticks) -gt 1});if($failures.Count){$failures|Sort-Object {[math]::Abs([double]$_.error_ticks)} -Descending|Export-Csv (Join-Path $ResultDirectory 'development_failures.csv') -NoTypeInformation -Encoding utf8}}
if(Test-Path -LiteralPath $evalPath){
 $evaluation=Import-Csv $evalPath
 $bins=@(foreach($block in ($evaluation|Group-Object {[math]::Floor([math]::Round([double]$_.true_T*100)/10)})){
  [pscustomobject]@{T_bin_start=[double]$block.Name*.1;count=$block.Count;mean_error_ticks=($block.Group|ForEach-Object {[double]$_.error_ticks}|Measure-Object -Average).Average;max_abs_error_ticks=($block.Group|ForEach-Object {[math]::Abs([double]$_.error_ticks)}|Measure-Object -Maximum).Maximum}
 })
 $bins|Sort-Object T_bin_start|Export-Csv (Join-Path $ResultDirectory 'evaluation_bias_by_T.csv') -NoTypeInformation -Encoding utf8
}
