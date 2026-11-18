param([string]$Path = (Split-Path $PSScriptRoot -Parent), [switch]$Fix)
$ErrorActionPreference = 'Stop'
$root = [IO.Path]::GetFullPath($Path)
$extensions = @('.md','.txt','.cpp','.hpp','.h','.c','.cmake','.ps1','.bat','.cmd','.json','.canvas','.html','.htm','.svg','.xml','.manifest','.yaml','.yml','.tsv','.csv','.py','.js','.css','.ini')
$pattern = '[\u00B7\u2013\u2014\u2022\u318D\u2600-\u27BF]|[\uD83C-\uD83E][\uDC00-\uDFFF]|\uFE0F'
$regex = [regex]::new($pattern)
$files = if (Test-Path -LiteralPath $root -PathType Leaf) { @(Get-Item -LiteralPath $root) } else {
    @(Get-ChildItem -LiteralPath $root -Recurse -Force -File | Where-Object {
        $relative = $_.FullName.Substring($root.Length).TrimStart('\','/')
        $relative -notmatch '(^|[\\/])(\.git|\.claude|build|node_modules|\.venv)([\\/]|$)' -and
        $relative -notmatch '^data[\\/]results[\\/]' -and
        ($extensions -contains $_.Extension.ToLowerInvariant() -or $_.Name -eq 'CMakeLists.txt')
    })
}
$found = 0; $changed = 0
foreach ($file in $files) {
    $bytes = [IO.File]::ReadAllBytes($file.FullName)
    $text = [IO.File]::ReadAllText($file.FullName)
    # Check visible HTML entities without decoding unrelated markup.
    if ($file.Extension -in @('.html','.htm','.svg','.xml','.manifest','.canvas')) {
        $text = [regex]::Replace($text, '&(?:#[xX][0-9a-fA-F]+|#[0-9]+|[A-Za-z]+);', [Text.RegularExpressions.MatchEvaluator]{ param($entity)
            $decoded = [Net.WebUtility]::HtmlDecode($entity.Value)
            if ($regex.IsMatch($decoded)) { return $decoded }
            return $entity.Value
        })
    }
    $matches = $regex.Matches($text)
    if (!$matches.Count) { continue }
    $found += $matches.Count
    Write-Output ($file.FullName + ': ' + $matches.Count + ' prohibited characters')
    if (!$Fix) { continue }
    $text = $regex.Replace($text, [Text.RegularExpressions.MatchEvaluator]{ param($match)
        $code = [char]::ConvertToUtf32($match.Value, 0)
        switch ($code) {
            0x00B7 { ', '; break }
            0x318D { ', '; break }
            0x2013 { '-'; break }
            0x2014 { '-'; break }
            0x2022 { '-'; break }
            0x2610 { '[ ]'; break }
            0x2611 { '[x]'; break }
            0x2717 { '[no]'; break }
            0x26A0 { '[warning]'; break }
            0x2753 { '[question]'; break }
            0x270E { '[edit]'; break }
            0x2605 { '*'; break }
            0xFE0F { ''; break }
            default { '[note]' }
        }
    })
    $bom = $bytes.Length -ge 3 -and $bytes[0] -eq 239 -and $bytes[1] -eq 187 -and $bytes[2] -eq 191
    [IO.File]::WriteAllText($file.FullName, $text, [Text.UTF8Encoding]::new($bom))
    ++$changed
}
Write-Output ('Checked {0} text files; found {1}; changed {2}.' -f $files.Count,$found,$changed)
if ($found -gt 0 -and !$Fix) { exit 1 }
exit 0
