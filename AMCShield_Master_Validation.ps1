}
    }
    $TrainingFiles = $TrainingFiles |
        Where-Object { $_.Name -match "train|model|config|checkpoint|experiment|robust|baseline" } |
        Sort-Object FullName -Unique

    $TrainingFiles |
        Select-Object FullName, Length, LastWriteTime |
        Format-Table -AutoSize

    # 11. Training entry points and config keys
    Section "11. TRAINING ENTRY POINTS AND CONFIG KEYS"
    $TrainingPatterns = @(
        "__main__", "ArgumentParser", "add_argument", "def train",
        "def main", "class .*Trainer", "DataLoader", "RobustCNN",
        "Baseline", "epochs:", "batch_size:", "learning_rate:",
        "lr:", "optimizer:", "scheduler:", "checkpoint", "resume",
        "save", "load"
    )
    foreach ($File in $TrainingFiles) {
        if ($File.Extension -eq ".py") {
            $Hits = Select-String -Path $File.FullName -Pattern $TrainingPatterns
            if ($Hits) {
                Write-Host "`n--- $($File.FullName) ---" -ForegroundColor Green
                $Hits | Select-Object -First 35 | ForEach-Object {
                    "{0}: {1}" -f $_.LineNumber, $_.Line.Trim()
                }
            }
        }
    }

    # 12. Configuration files
    Section "12. CONFIGURATION FILES"
    $ConfigFiles = Get-ChildItem $Root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object {
            $_.Extension -in @(".yaml", ".yml", ".toml") -and
            $_.FullName -notmatch "\\(\.venv|venv|node_modules|site-packages|data|backups|\.git)\\"
        } |
        Where-Object { $_.FullName -match "config|train|model" } |
        Select-Object -First 20

    foreach ($File in $ConfigFiles) {
        Show-File $File.FullName 100
    }

    # 13. Checkpoint inventory — names and sizes only
    Section "13. CHECKPOINT INVENTORY"
    foreach ($Folder in @(".\checkpoints", ".\results", ".\artifacts", ".\outputs")) {
        if (Test-Path $Folder) {
            Write-Host "`nFolder: $Folder" -ForegroundColor Green
            Get-ChildItem $Folder -Recurse -File -ErrorAction SilentlyContinue |
                Where-Object { $_.Extension -in @(".pt", ".pth", ".ckpt", ".safetensors") } |
                Select-Object FullName, Length, LastWriteTime |
                Format-Table -AutoSize
        }
    }

    # 14. Python dependencies and hardware
    Section "14. TRAINING DEPENDENCIES"
    if (Test-Path ".\requirements.txt") {
        Select-String -Path ".\requirements.txt" -Pattern @(
            "torch", "torchvision", "torchaudio", "numpy", "h5py",
            "scikit-learn", "psutil", "fastapi", "uvicorn",
            "pyyaml", "matplotlib", "tensorboard"
        )
    }

    Write-Host "`nPython version:"
    python --version 2>&1

    Write-Host "`nPyTorch availability:"
    python -c "import torch; print('Torch:',torch.__version__); print('CUDA available:',torch.cuda.is_available()); print('CUDA version:',torch.version.cuda); print('Device:',torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')" 2>&1

    # 15. Registered API routes
    Section "15. REGISTERED API ROUTES"
    @'
try:
    from backend.main import app
    schema = app.openapi()
    for path, methods in sorted(schema.get("paths", {}).items()):
        if any(word in path.lower() for word in
               ("train", "model", "checkpoint", "telemetry", "history")):
            print(path, "=>", ", ".join(methods.keys()))
    print("OpenAPI route inspection completed.")
except Exception as exc:
    print("Could not inspect OpenAPI:", type(exc).__name__, str(exc))
'@ | python - 2>&1

    # 16. Security references
    Section "16. TRAINING SECURITY REFERENCES"
    foreach ($File in @(".\backend\dependencies.py", ".\backend\security.py")) {
        if (Test-Path $File) {
            Show-File $File 150
        }
    }

    if (Test-Path ".\backend\routers") {
        Get-ChildItem ".\backend\routers" -File -Filter "*.py" |
            Select-String -Pattern @(
                "require_authenticated", "require_admin", "Depends\(",
                "HTTPException", "Path\(", "resolve\(", "subprocess"
            ) |
            Select-Object -First 100 |
            ForEach-Object {
                "{0}:{1}: {2}" -f $_.Path, $_.LineNumber, $_.Line.Trim()
            }
    }

    Section "17. AUDIT COMPLETE"
    Write-Host "Read-only inspection finished." -ForegroundColor Green
    Write-Host "No training process was started."
    Write-Host "No checkpoint was opened or modified."
    Write-Host "No dataset contents were loaded."
    Write-Host "No authentication database or secret values were intentionally printed."
    Write-Host "Report file: $ReportPath" -ForegroundColor Yellow
}
catch {
    Write-Host "`nAUDIT ERROR: $($_.Exception.Message)" -ForegroundColor Red
    Write-Host $_.ScriptStackTrace
}
finally {
    Stop-Transcript | Out-Null
    Write-Host "`nAudit report saved to:" -ForegroundColor Green
    Write-Host $ReportPath -ForegroundColor Yellow
}