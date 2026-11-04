param([string]$OutputPath)
$ErrorActionPreference = 'Stop'
$repo = Split-Path $PSScriptRoot -Parent
$v2 = Get-ChildItem -LiteralPath (Join-Path $repo 'docs/hye') -Directory |
    Where-Object { $_.Name -like '*_v2' }
if (@($v2).Count -ne 1) { throw 'Expected exactly one pipeline v2 directory' }
$selection = Join-Path $v2.FullName '_워크플로/선정'
$rows = @()
foreach ($file in Get-ChildItem -LiteralPath $selection -Filter '*.json' | Sort-Object Name) {
    $selected = Get-Content -LiteralPath $file.FullName -Raw -Encoding UTF8 | ConvertFrom-Json
    $missing = @()
    $stagePath = $null
    foreach ($doc in $selected.docs) {
        $relative = $doc.folder.Replace('\', '/')
        $marker = $relative.IndexOf('docs/hye/')
        if ($marker -lt 0) { throw "Unexpected document path: $relative" }
        $relative = $relative.Substring($marker)
        $folder = [IO.Path]::GetFullPath((Join-Path $repo $relative))
        $boundary = [IO.Path]::GetFullPath($repo).TrimEnd('\') + '\'
        if (-not $folder.StartsWith($boundary, [StringComparison]::OrdinalIgnoreCase)) {
            throw "Document path outside repository: $relative"
        }
        $stagePath = Join-Path (Split-Path (Split-Path $folder -Parent) -Parent) '00_단계.md'
        foreach ($name in @('번역.md', '요약.md')) {
            $target = Join-Path $folder $name
            if (-not (Test-Path -LiteralPath $target -PathType Leaf) -or
                (Get-Item -LiteralPath $target).Length -eq 0) {
                $missing += "$relative/$name"
            }
        }
    }
    if (-not $stagePath) { throw "No selected documents: $($file.Name)" }
    $stageExists = (Test-Path -LiteralPath $stagePath -PathType Leaf)
    if ($stageExists) { $stageExists = (Get-Item -LiteralPath $stagePath).Length -gt 0 }
    $rows += [pscustomobject]@{
        unit = $file.BaseName
        stage = $stagePath.Substring($repo.Length + 1)
        stage_exists = $stageExists
        missing_documents = @($missing)
    }
}
$result = [pscustomobject]@{
    scope = 'Nonempty file existence only; content quality and source verification are separate.'
    units = $rows.Count
    stage_pages = @($rows | Where-Object stage_exists).Count
    missing_document_files = @($rows | ForEach-Object { $_.missing_documents }).Count
    pending = @($rows | Where-Object { -not $_.stage_exists -or $_.missing_documents.Count -gt 0 })
}
$json = $result | ConvertTo-Json -Depth 8
if ($OutputPath) {
    [IO.File]::WriteAllText([IO.Path]::GetFullPath($OutputPath), $json, [Text.UTF8Encoding]::new($false))
}
Write-Output $json
if ($result.pending.Count -gt 0) { exit 1 }
