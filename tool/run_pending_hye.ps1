param()
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$cmake = 'C:\Program Files (x86)\Microsoft Visual Studio\18\BuildTools\Common7\IDE\CommonExtensions\Microsoft\CMake\CMake\bin\cmake.exe'
$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$run = Join-Path $repo "batch_runs\$stamp"
New-Item -ItemType Directory -Path $run -Force | Out-Null
$env:PATH = "C:\opencv\build\x64\vc15\bin;$env:PATH"
function Invoke-Step($Arguments, $Log) {
    & $cmake @Arguments >> $Log 2>&1
    if ($LASTEXITCODE -ne 0) { throw "CMake failed: $LASTEXITCODE" }
}
function Snapshot($Path) {
    $map = @{}
    if (Test-Path -LiteralPath $Path) {
        Get-ChildItem -LiteralPath $Path -Recurse -File | ForEach-Object {
            $map[$_.FullName.Substring($Path.Length)] = (Get-FileHash -LiteralPath $_.FullName -Algorithm SHA256).Hash
        }
    }
    return $map
}
$jobs = @()
Get-ChildItem -LiteralPath "$repo\experiments\hye" -Directory | Sort-Object Name | ForEach-Object {
    $src = Join-Path $_.FullName ([string][char]0xC2E4 + [char]0xD5D8)
    $file = Join-Path $src 'CMakeLists.txt'
    if (Test-Path -LiteralPath $file) {
        $text = Get-Content -LiteralPath $file -Raw -Encoding UTF8
        if ($text -match 'hye_experiment\(\s*([A-Za-z0-9_]+)') {
            $target = $Matches[1]
            $tracked = @(& git -C $repo ls-files -- $file)
            if ($tracked.Count -eq 0) {
                $jobs += [pscustomobject]@{ Name=$_.Name; Source=$src; Target=$target; Folder=$_.FullName }
            }
        }
    }
}
$jobs | Export-Csv "$run\plan.csv" -NoTypeInformation -Encoding UTF8
Write-Host "Pending experiments: $($jobs.Count). Logs/backups: $run"
$rows = @()
$i = 0
foreach ($job in $jobs) {
    $i++
    $log = Join-Path $run ($job.Target + '.log')
    $build = Join-Path $repo ('.batch_build\' + $job.Target)
    $result = Join-Path $job.Folder ([string][char]0xACB0 + [char]0xACFC)
    $status = 'FAILED'
    $detail = ''
    Write-Host "[$i/$($jobs.Count)] $($job.Target)"
    try {
        if (Test-Path -LiteralPath $result) {
            $backup = Join-Path $run ('before\' + $job.Target)
            New-Item -ItemType Directory -Path $backup -Force | Out-Null
            Copy-Item -LiteralPath $result -Destination $backup -Recurse
        }
        Invoke-Step @('-S', $job.Source, '-B', $build, '-G', 'Visual Studio 18 2026', '-A', 'x64', '-T', 'v141', '-DOpenCV_DIR=C:/opencv/build') $log
        Invoke-Step @('--build', $build, '--config', 'Release', '--parallel', '2') $log
        $exe = Join-Path $build ('Release\' + $job.Target + '.exe')
        & $exe >> $log 2>&1
        if ($LASTEXITCODE -ne 0) { throw "First run failed: $LASTEXITCODE" }
        $first = Snapshot $result
        if ($first.Count -eq 0) { throw 'No result files generated' }
        & $exe >> $log 2>&1
        if ($LASTEXITCODE -ne 0) { throw "Second run failed: $LASTEXITCODE" }
        $second = Snapshot $result
        $changed = @($first.Keys | Where-Object { -not $second.ContainsKey($_) -or $first[$_] -ne $second[$_] })
        if ($changed.Count -gt 0 -or $first.Count -ne $second.Count) {
            $status = 'RESULTS_DIFFER'
        } else { $status = 'REPRODUCIBLE' }
    } catch { $detail = $_.Exception.Message; Add-Content -LiteralPath $log -Value $detail }
    $rows += [pscustomobject]@{ Experiment=$job.Name; Target=$job.Target; Status=$status; Detail=$detail; Log=$log }
    $rows | Export-Csv "$run\summary.csv" -NoTypeInformation -Encoding UTF8
    Write-Host "  $status $detail"
}
Write-Host "Finished. Summary: $run\summary.csv"
Write-Host 'REPRODUCIBLE means identical files across two runs; original-result parity must be reviewed separately.'
