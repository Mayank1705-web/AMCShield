$ErrorActionPreference = "Stop"
$root = (Get-Location).Path
$baseline = Join-Path $root "src\train_baseline.py"
$robust = Join-Path $root "src\train_robust.py"
$hook = Join-Path $root "src\training_dashboard_hooks.py"

foreach ($file in @($baseline, $robust)) {
    if (-not (Test-Path -LiteralPath $file -PathType Leaf)) {
        throw "Missing required file: $file. No files changed."
    }
}

$baselineText = [System.IO.File]::ReadAllText($baseline)
$robustText = [System.IO.File]::ReadAllText($robust)
if (Test-Path -LiteralPath $hook) {
    throw "src\training_dashboard_hooks.py already exists. Inspect it before applying this patch."
}

$baselineImport = "from src.train import train_model, load_config"
$robustImport = "from src.train import train_model, load_config"
$baselineCall = @'
        checkpoint_path=checkpoint_path,
    )
'@
$baselineCallNew = @'
        checkpoint_path=checkpoint_path,
        should_stop=should_stop_training,
    )
'@
$robustCall = @'
        checkpoint_path=checkpoint_path,
        max_train_batches=args.max_train_batches,
    )
'@
$robustCallNew = @'
        checkpoint_path=checkpoint_path,
        max_train_batches=args.max_train_batches,
        should_stop=should_stop_training,
    )
'@

if (-not $baselineText.Contains($baselineImport) -or
    -not $baselineText.Contains($baselineCall)) {
    throw "Baseline runner anchors did not match. No files changed."
}
if (-not $robustText.Contains($robustImport) -or
    -not $robustText.Contains($robustCall)) {
    throw "Robust runner anchors did not match. No files changed."
}

$baselineText = $baselineText.Replace(
    $baselineImport,
    $baselineImport + "`r`nfrom src.training_dashboard_hooks import should_stop_training"
)
$baselineText = $baselineText.Replace($baselineCall, $baselineCallNew)

$robustText = $robustText.Replace(
    $robustImport,
    $robustImport + "`r`nfrom src.training_dashboard_hooks import should_stop_training"
)
$robustText = $robustText.Replace($robustCall, $robustCallNew)

$hookText = @'
"""Optional stop-file integration for dashboard-launched training.

When AMC_TRAINING_CONTROL_DIR is not set, standalone training is unaffected.
"""
import os
from pathlib import Path


def should_stop_training() -> bool:
    control_dir = os.environ.get("AMC_TRAINING_CONTROL_DIR")
    if not control_dir:
        return False
    return (Path(control_dir) / "stop.request").is_file()
'@

$backupDir = Join-Path $root ("backups\dashboard_hook_patch_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item -LiteralPath $baseline -Destination (Join-Path $backupDir "train_baseline.py")
Copy-Item -LiteralPath $robust -Destination (Join-Path $backupDir "train_robust.py")

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($baseline, $baselineText, $utf8NoBom)
[System.IO.File]::WriteAllText($robust, $robustText, $utf8NoBom)
[System.IO.File]::WriteAllText($hook, $hookText, $utf8NoBom)

Write-Output "PATCH APPLIED: dashboard stop-file hook"
Write-Output "Updated: src\train_baseline.py"
Write-Output "Updated: src\train_robust.py"
Write-Output "Created: src\training_dashboard_hooks.py"
Write-Output "Backup: $backupDir"
Write-Output "Standalone runners ignore dashboard stop requests when AMC_TRAINING_CONTROL_DIR is unset."
Write-Output "No training process was started."
