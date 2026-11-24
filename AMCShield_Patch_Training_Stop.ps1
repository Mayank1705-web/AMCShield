$ErrorActionPreference = "Stop"

$projectRoot = (Get-Location).Path
$target = Join-Path $projectRoot "src\train.py"

if (-not (Test-Path -LiteralPath $target -PathType Leaf)) {
    throw "Run this script from the AMCShield project root. Missing: $target"
}

$content = [System.IO.File]::ReadAllText($target)
$original = $content

# 1) Add an optional dashboard stop callback without changing existing callers.
$oldSignature = @'
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
) -> torch.nn.Module:
'@
$newSignature = @'
    max_train_batches: Optional[int] = None,
    max_val_batches: Optional[int] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> torch.nn.Module:
'@

if (-not $content.Contains($oldSignature)) {
    throw "Expected train_model signature was not found. No changes made."
}
$content = $content.Replace($oldSignature, $newSignature)

# 2) Track manual interruption independently from early stopping.
$epochLoopAnchor = @'
    # ========================================================
    # Epoch loop
    # ========================================================

    if early_stopped or start_epoch > epochs:
'@
$epochLoopReplacement = @'
    # ========================================================
    # Epoch loop
    # ========================================================

    manual_stop_requested = False

    if early_stopped or start_epoch > epochs:
'@

if (-not $content.Contains($epochLoopAnchor)) {
    throw "Expected epoch-loop anchor was not found. No changes made."
}
$content = $content.Replace($epochLoopAnchor, $epochLoopReplacement)

# 3) After each completed batch, poll the dashboard stop callback.
#    Save the next batch index so resume does not repeat the completed batch.
$batchSaveAnchor = @'
            if (
                save_every_batches > 0
                and (batch_index + 1) % save_every_batches == 0
            ):
                save_resume_state(
                    epoch,
                    batch_index + 1,
                    running_loss,
                    correct,
                    total,
                )
'@
$batchSaveReplacement = @'
            if (
                save_every_batches > 0
                and (batch_index + 1) % save_every_batches == 0
            ):
                save_resume_state(
                    epoch,
                    batch_index + 1,
                    running_loss,
                    correct,
                    total,
                )

            # Dashboard Stop is cooperative: finish this batch, then
            # persist the exact next-batch position before leaving.
            if should_stop is not None and should_stop():
                save_resume_state(
                    epoch,
                    batch_index + 1,
                    running_loss,
                    correct,
                    total,
                )
                manual_stop_requested = True
                print("\nDashboard stop requested. Resume state saved.")
                break
'@

if (-not $content.Contains($batchSaveAnchor)) {
    throw "Expected periodic resume-save block was not found. No changes made."
}
$content = $content.Replace($batchSaveAnchor, $batchSaveReplacement)

# 4) Skip validation and epoch completion when manually paused.
$closeBatchAnchor = @'
        # Close batch progress bar
        # ----------------------------------------------------

        batch_progress.close()

        # ----------------------------------------------------
        # Epoch training metrics
'@
$closeBatchReplacement = @'
        # Close batch progress bar
        # ----------------------------------------------------

        batch_progress.close()

        if manual_stop_requested:
            break

        # ----------------------------------------------------
        # Epoch training metrics
'@

if (-not $content.Contains($closeBatchAnchor)) {
    throw "Expected post-batch-loop anchor was not found. No changes made."
}
$content = $content.Replace($closeBatchAnchor, $closeBatchReplacement)

# 5) Report a manual pause accurately; preserve the normal completion message.
$completionAnchor = @'
    print()
    print("=" * 70)
    print("TRAINING COMPLETED")
    print("=" * 70)
'@
$completionReplacement = @'
    print()
    print("=" * 70)
    if manual_stop_requested:
        print("TRAINING PAUSED (manual stop)")
    else:
        print("TRAINING COMPLETED")
    print("=" * 70)
'@

if (-not $content.Contains($completionAnchor)) {
    throw "Expected completion-message block was not found. No changes made."
}
$content = $content.Replace($completionAnchor, $completionReplacement)

if ($content -eq $original) {
    throw "Patch produced no changes. Nothing written."
}

# Preserve a separate timestamped copy immediately before modifying the source.
$backupDir = Join-Path $projectRoot ("backups\training_stop_patch_" + (Get-Date -Format "yyyyMMdd_HHmmss"))
New-Item -ItemType Directory -Path $backupDir -Force | Out-Null
Copy-Item -LiteralPath $target -Destination (Join-Path $backupDir "train.py") -ErrorAction Stop

$utf8NoBom = New-Object System.Text.UTF8Encoding($false)
[System.IO.File]::WriteAllText($target, $content, $utf8NoBom)

Write-Output "PATCH APPLIED: src\train.py"
Write-Output "Backup: $backupDir\train.py"
Write-Output "Added optional should_stop callback."
Write-Output "Existing callers remain compatible because the new argument is optional."
Write-Output "No training process was started."
