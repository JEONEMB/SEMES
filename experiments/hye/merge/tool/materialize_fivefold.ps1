param([string]$SourceClean,[string]$Destination,[string]$Splitter)
$ErrorActionPreference='Stop'
$culture=[Globalization.CultureInfo]::InvariantCulture
$dest=[IO.Path]::GetFullPath($Destination)
$records=@(); $centers=@()
foreach($date in @('20261014','20261020','20261029')) {
  $base=Join-Path $SourceClean "clean_$date"
  foreach($r in Import-Csv -Encoding UTF8 (Join-Path $base '_metadata/manifest.csv')) {
    $y=[double]::Parse($r.filename_Y_command,$culture); $t=[double]::Parse($r.filename_T_command,$culture)
    $src=Join-Path $base $r.relative_path
    if(!(Test-Path -LiteralPath $src)){throw "Missing $src"}
    $hash=(Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLowerInvariant()
    if($hash -ne $r.sha256){throw "Hash mismatch $src"}
    $id=$date+'_Y'+$y.ToString('+0.00;-0.00;+0.00',$culture)+'_T'+$t.ToString('+0.00;-0.00;+0.00',$culture)
    $stratum=if($date -eq '20261014'){if($t -eq 0){'1014_Y'}else{'1014_T'}}else{$date.Substring(4)}
    $entry=[pscustomobject]@{date=$date;source=$src;relative_source=$r.relative_path;sha256=$hash;Y=$y;T=$t;group=$id;stratum=$stratum;name=[IO.Path]::GetFileName($src);repeat=$r.capture_repeat;test_fold=0}
    if($y -eq 0 -and $t -eq 0){$centers+=$entry}else{$records+=$entry}
  }
}
if($records.Count -ne 571 -or $centers.Count -ne 4){throw 'Unexpected total'}
if(@(($records+$centers)|Group-Object sha256|Where-Object Count -gt 1).Count){throw 'Duplicate images: review before split'}
Write-Output 'Source hashes verified: 575 BMP'
$meta=Join-Path $dest '_metadata'; New-Item -ItemType Directory -Force $meta|Out-Null
$groups=$records|Group-Object group
$inputPath=Join-Path $meta 'groups.tsv'
$groupLines=@($groups|ForEach-Object {$_.Group[0].stratum+"`t"+$_.Name})
[IO.File]::WriteAllLines($inputPath,$groupLines,[Text.UTF8Encoding]::new($false))
$assigned=& $Splitter $inputPath
if($LASTEXITCODE -ne 0){throw 'Splitter failed'}
$assignmentPath=Join-Path $meta 'assignments.tsv'
[IO.File]::WriteAllLines($assignmentPath,[string[]]$assigned,[Text.UTF8Encoding]::new($false))
$map=@{}; foreach($a in ($assigned|ConvertFrom-Csv -Delimiter "`t")){$map[$a.group_id]=[int]$a.test_fold}
foreach($r in $records){$r.test_fold=$map[$r.group];if($r.test_fold -lt 1 -or $r.test_fold -gt 5){throw 'Invalid fold'}}
$records|Export-Csv -Encoding UTF8 -NoTypeInformation (Join-Path $meta 'images.csv')
$centers|Export-Csv -Encoding UTF8 -NoTypeInformation (Join-Path $meta 'centers.csv')
$summary=@();$verified=0
for($fold=1;$fold -le 5;$fold++) {
  foreach($split in @('train','test')) {
    foreach($date in @('20261014','20261020','20261029')) {
      $dir=Join-Path $dest ('fold_{0:00}/{1}/{2}' -f $fold,$split,$date)
      New-Item -ItemType Directory -Force $dir|Out-Null
      if($date -eq '20261014'){foreach($axis in @('Y','T')){New-Item -ItemType Directory -Force (Join-Path $dir $axis)|Out-Null}}
      $selected=@($records|Where-Object {$_.date -eq $date -and (($_.test_fold -eq $fold) -eq ($split -eq 'test'))})
      foreach($r in $selected) {
        $targetDir=$dir
        if($date -eq '20261014'){$axis=if($r.T -eq 0){'Y'}else{'T'};$targetDir=Join-Path $dir $axis}
        $target=Join-Path $targetDir $r.name
        if((Test-Path -LiteralPath $target) -and (Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant() -ne $r.sha256){
          $target=Join-Path $dir $r.relative_source
          New-Item -ItemType Directory -Force ([IO.Path]::GetDirectoryName($target))|Out-Null
        }
        if(!(Test-Path -LiteralPath $target)){Copy-Item -LiteralPath $r.source -Destination $target}
        if((Get-FileHash -LiteralPath $target -Algorithm SHA256).Hash.ToLowerInvariant() -ne $r.sha256){throw "Copied hash mismatch $target"}
        $verified++
      }
      $summary+=[pscustomobject]@{fold=$fold;split=$split;date=$date;count=$selected.Count}
    }
  }
  Write-Output "Completed fold $fold : copied and hash-verified"
}
foreach($r in $centers){$dir=Join-Path $dest ('center/'+$r.date);New-Item -ItemType Directory -Force $dir|Out-Null;$target=Join-Path $dir $r.name;if(!(Test-Path -LiteralPath $target)){Copy-Item -LiteralPath $r.source -Destination $target};if((Get-FileHash -LiteralPath $target).Hash.ToLowerInvariant() -ne $r.sha256){throw 'Center copy mismatch'}}
$summary|Export-Csv -Encoding UTF8 -NoTypeInformation (Join-Path $meta 'counts.csv')
$actual=@(Get-ChildItem -LiteralPath $dest -Recurse -Filter '*.bmp' -File).Count
if($verified -ne 2855 -or $actual -ne 2859){throw "Unexpected copied count $verified / $actual"}
@{status='complete';source_images=575;noncenter_images=571;center_images=4;group_count=$groups.Count;copied_fold_images=$verified;total_bmp=$actual;source_hashes_verified=$true;destination_hashes_verified=$true;duplicate_source_hashes=0;test_appearances_per_image=1;train_appearances_per_image=4;seed=20261104;timestamp=(Get-Date -Format o)}|ConvertTo-Json|Set-Content -Encoding UTF8 (Join-Path $meta 'integrity_checks.json')
Write-Output 'COMPLETE: 2859 BMP files copied and hash-verified'
