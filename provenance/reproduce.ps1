param(
    [string]$Python = "python"
)

$ErrorActionPreference = "Stop"
Set-StrictMode -Version Latest

$repoRoot = Split-Path -Parent $PSScriptRoot
$logDir = Join-Path $PSScriptRoot "logs"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stamp = Get-Date -Format "yyyyMMdd-HHmmss"
$logPath = Join-Path $logDir "reproduce-$stamp.log"

$programs = @(
    "prereearch_switched_baths.py",
    "second_round_driven_qubit.py",
    "third_round_inference.py",
    "fourth_round_detectability.py",
    "fifth_round_trajectory_kl.py",
    "sixth_round_first_order.py",
    "plot_manuscript_figures.py"
)

Push-Location $repoRoot
try {
    foreach ($program in $programs) {
        "Running $program" | Tee-Object -FilePath $logPath -Append
        & $Python $program 2>&1 | Tee-Object -FilePath $logPath -Append
        if ($LASTEXITCODE -ne 0) {
            throw "$program failed with exit code $LASTEXITCODE"
        }
    }

    $generatedFigures = Join-Path $repoRoot "manuscript_figures"
    $releaseFigures = Join-Path $repoRoot "figures"
    Copy-Item -Path (Join-Path $generatedFigures "*") `
        -Destination $releaseFigures -Force

    "Reproduction completed successfully." |
        Tee-Object -FilePath $logPath -Append
}
finally {
    Pop-Location
}
