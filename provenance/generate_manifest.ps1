$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$tsvPath = Join-Path $PSScriptRoot "artifact_manifest.tsv"
$shaPath = Join-Path $PSScriptRoot "artifact_manifest.sha256"

$excludedNames = @(
    "artifact_manifest.tsv",
    "artifact_manifest.sha256"
)

$files = Get-ChildItem -LiteralPath $repoRoot -File -Recurse |
    Where-Object {
        $_.FullName -notmatch "[\\/]\.git[\\/]" -and
        $_.FullName -notmatch "[\\/]\.venv[\\/]" -and
        $_.FullName -notmatch "[\\/]__pycache__[\\/]" -and
        $_.FullName -notmatch "[\\/]provenance[\\/]logs[\\/]" -and
        $_.Extension -notin @(".pyc", ".aux", ".bbl", ".blg", ".fdb_latexmk", ".fls", ".log", ".out", ".toc") -and
        $_.Name -notin $excludedNames
    } |
    Sort-Object FullName

$rows = foreach ($file in $files) {
    $relative = $file.FullName.Substring($repoRoot.Length + 1).Replace("\", "/")
    $hash = (Get-FileHash -LiteralPath $file.FullName -Algorithm SHA256).Hash.ToLowerInvariant()
    [PSCustomObject]@{
        path = $relative
        bytes = $file.Length
        sha256 = $hash
    }
}

$rows | Export-Csv -LiteralPath $tsvPath -Delimiter "`t" -NoTypeInformation
$shaLines = $rows | ForEach-Object { "$($_.sha256)  $($_.path)" }
Set-Content -LiteralPath $shaPath -Value $shaLines -Encoding utf8

Write-Host "Wrote $($rows.Count) artifact hashes."
