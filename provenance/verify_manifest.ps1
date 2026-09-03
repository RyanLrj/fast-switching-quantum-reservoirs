$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$manifestPath = Join-Path $PSScriptRoot "artifact_manifest.tsv"

if (-not (Test-Path -LiteralPath $manifestPath)) {
    throw "Missing manifest: $manifestPath"
}

$rows = Import-Csv -LiteralPath $manifestPath -Delimiter "`t"
$failures = @()

foreach ($row in $rows) {
    $path = Join-Path $repoRoot $row.path
    if (-not (Test-Path -LiteralPath $path)) {
        $failures += "Missing: $($row.path)"
        continue
    }

    $file = Get-Item -LiteralPath $path
    $hash = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash.ToLowerInvariant()
    if ($file.Length -ne [long]$row.bytes -or $hash -ne $row.sha256) {
        $failures += "Changed: $($row.path)"
    }
}

if ($failures.Count -gt 0) {
    $failures | ForEach-Object { Write-Error $_ }
    throw "Artifact verification failed."
}

Write-Host "Verified $($rows.Count) artifacts successfully."
