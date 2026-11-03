param([Parameter(Mandatory=$true)][string]$ImageRoot,[Parameter(Mandatory=$true)][string]$ResultDirectory,[Parameter(Mandatory=$true)][string]$HistoricalInventory)
$rows=Import-Csv (Join-Path $ResultDirectory 'audit.csv')
if($rows.Count -ne 575){throw 'Extraction is incomplete'}
$historical=@{}
Import-Csv $HistoricalInventory | ForEach-Object {$historical[$_.path]=$_.sha256}
$changed=0;$historicalMismatch=0;$matched=0;$crossSplitDuplicates=0
$cohortByPath=@{};foreach($row in $rows){$cohortByPath[$row.path]=$row.cohort}
foreach($row in $rows){
 $actual=(Get-FileHash -LiteralPath (Join-Path $ImageRoot $row.path) -Algorithm SHA256).Hash.ToLowerInvariant()
 if($actual -ne $row.sha256){$changed++}
 if($historical.ContainsKey($row.path)){$matched++;if($actual -ne $historical[$row.path]){$historicalMismatch++}}
 if($row.duplicate_of -and (($row.cohort -eq 'new_evaluation') -ne ($cohortByPath[$row.duplicate_of] -eq 'new_evaluation'))){$crossSplitDuplicates++}
}
[pscustomobject]@{images=$rows.Count;changed_since_measurement=$changed;historical_images_checked=$matched;historical_hash_mismatches=$historicalMismatch;duplicate_files=@($rows|Where-Object duplicate_of -ne '').Count;cross_split_duplicate_files=$crossSplitDuplicates;measurement_failures=@($rows|Where-Object status -ne '0').Count} | ConvertTo-Json | Set-Content -Encoding utf8 (Join-Path $ResultDirectory 'integrity.json')
if($changed -or $historicalMismatch -or $crossSplitDuplicates){throw 'Data integrity check failed; independent evaluation is blocked'}
