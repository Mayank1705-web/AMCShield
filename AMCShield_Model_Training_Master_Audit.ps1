[CmdletBinding()]
param(
    [string]$ProjectRoot = $PSScriptRoot,
    [switch]$RunPytest
)

$ErrorActionPreference = "Stop"
$timestamp = Get-Date -Format "yyyyMMdd_HHmmss"
$reportDir = Join-Path $ProjectRoot "reports\model_training_sprint_$timestamp"
$null = New-Item -ItemType Directory -Path $reportDir -Force
$reportFile = Join-Path $reportDir "AMCShield_Model_Training_Audit.txt"
$transcriptFile = Join-Path $reportDir "execution_transcript.txt"

function Write-Section([string]$Title) {
    "`r`n========== $Title ==========" | Tee-Object -FilePath $reportFile -Append
}
function Write-Result([string]$Text) {
    $Text | Tee-Object -FilePath $reportFile -Append
}
function Invoke-SafeCheck([string]$Name, [scriptblock]$Action) {
    Write-Section $Name
    try {
        & $Action
    } catch {
        Write-Result "CHECK ERROR: $($_.Exception.Message)"
    }
}

if (-not (Test-Path -LiteralPath $ProjectRoot -PathType Container)) {
    throw "Project directory not found: $ProjectRoot"
}
Set-Location -LiteralPath $ProjectRoot
Start-Transcript -Path $transcriptFile -Force | Out-Null

try {
    "AMCShield Model Training Dashboard - Safe Audit" | Set-Content -LiteralPath $reportFile
    "Timestamp: $(Get-Date -Format o)" | Add-Content -LiteralPath $reportFile
    "Project: $ProjectRoot" | Add-Content -LiteralPath $reportFile
    "Mode: READ-ONLY AUDIT + VALIDATION. No training will be started." | Add-Content -LiteralPath $reportFile

    Invoke-SafeCheck "1. Environment" {
        Write-Result "PowerShell: $($PSVersionTable.PSVersion)"
        if (Test-Path ".venv\Scripts\python.exe") {
            $python = (Resolve-Path ".venv\Scripts\python.exe").Path
        } else {
            $cmd = Get-Command python -ErrorAction SilentlyContinue
            if (-not $cmd) { throw "Python was not found. Activate/install the project environment." }
            $python = $cmd.Source
        }
        Write-Result "Python executable: $python"
        & $python --version 2>&1 | ForEach-Object { Write-Result "$_" }
        $script:PythonExe = $python
        $git = Get-Command git -ErrorAction SilentlyContinue
        Write-Result "Git available: $([bool]$git)"
        $node = Get-Command node -ErrorAction SilentlyContinue
        Write-Result "Node available: $([bool]$node)"
    }

    Invoke-SafeCheck "2. Git safety baseline" {
        if (Get-Command git -ErrorAction SilentlyContinue) {
            Write-Result "-- git status --short"
            git status --short 2>&1 | ForEach-Object { Write-Result "$_" }
            Write-Result "-- git diff --check"
            git diff --check 2>&1 | ForEach-Object { Write-Result "$_" }
            Write-Result "-- tracked diff summary"
            git diff --stat 2>&1 | ForEach-Object { Write-Result "$_" }
            Write-Result "No Git reset/checkout/clean/stash commands are executed by this script."
        } else {
            Write-Result "Git not installed or not on PATH; Git checks skipped."
        }
    }

    Invoke-SafeCheck "3. Important file inventory" {
        $files = @(
            "backend\main.py",
            "backend\config.py",
            "backend\dependencies.py",
            "backend\services\security_service.py",
            "backend\routers\training.py",
            "backend\services\training_service.py",
            "src\train.py",
            "src\train_baseline.py",
            "src\train_robust.py",
            "configs\config.yaml",
            "frontend\AMCShield_All_In_One\model-training.html",
            "frontend\AMCShield_All_In_One\model-training.js",
            "frontend\AMCShield_All_In_One\security.js",
            "frontend\AMCShield_All_In_One\auth-guard.js"
        )
        foreach ($file in $files) {
            if (Test-Path -LiteralPath $file -PathType Leaf) {
                $item = Get-Item -LiteralPath $file
                Write-Result ("FOUND  {0} ({1} bytes; modified {2})" -f $file, $item.Length, $item.LastWriteTime)
            } else {
                Write-Result "MISSING $file"
            }
        }
    }

    Invoke-SafeCheck "4. Python syntax validation (AST only; no imports/training)" {
        $pythonCheck = @'
import ast
from pathlib import Path
roots = [Path("backend"), Path("src")]
files = []
for root in roots:
    if root.exists():
        files.extend(p for p in root.rglob("*.py") if ".venv" not in p.parts)
errors = []
for path in files:
    try:
        ast.parse(path.read_text(encoding="utf-8-sig"), filename=str(path))
    except Exception as exc:
        errors.append(f"{path}: {type(exc).__name__}: {exc}")
print(f"Python files checked: {len(files)}")
print(f"Syntax errors: {len(errors)}")
for error in errors:
    print(error)
raise SystemExit(1 if errors else 0)
'@
        $pythonCheck | & $script:PythonExe - 2>&1 | ForEach-Object { Write-Result "$_" }
        if ($LASTEXITCODE -ne 0) { Write-Result "RESULT: Python syntax check FAILED." }
        else { Write-Result "RESULT: Python syntax check PASSED." }
    }

    Invoke-SafeCheck "5. Frontend JavaScript syntax validation" {
        $js = "frontend\AMCShield_All_In_One\model-training.js"
        if (-not (Test-Path -LiteralPath $js)) {
            Write-Result "Model Training JavaScript not found; skipped."
        } elseif (Get-Command node -ErrorAction SilentlyContinue) {
            node --check $js 2>&1 | ForEach-Object { Write-Result "$_" }
            if ($LASTEXITCODE -ne 0) { Write-Result "RESULT: JavaScript syntax check FAILED." }
            else { Write-Result "RESULT: JavaScript syntax check PASSED." }
        } else {
            Write-Result "Node.js unavailable; JavaScript syntax check skipped."
        }
    }

    Invoke-SafeCheck "6. Training API and frontend security indicators" {
        $router = "backend\routers\training.py"
        $service = "backend\services\training_service.py"
        $frontend = "frontend\AMCShield_All_In_One\model-training.js"
        $security = "frontend\AMCShield_All_In_One\security.js"

        if (Test-Path $router) {
            $content = Get-Content -LiteralPath $router -Raw
            foreach ($endpoint in @("overview","status","config","progress","history","models","checkpoints","logs","start","stop","resume")) {
                $found = $content -match [regex]::Escape('"' + $endpoint + '"') -or
                         $content -match [regex]::Escape("'" + $endpoint + "'") -or
                         $content -match [regex]::Escape("/" + $endpoint)
                Write-Result ("Backend route token {0}: {1}" -f $endpoint, $(if ($found) {"FOUND"} else {"NOT FOUND - inspect manually"}))
            }
        } else { Write-Result "Training router missing." }

        if (Test-Path $frontend) {
            $jsContent = Get-Content -LiteralPath $frontend -Raw
            Write-Result "Frontend uses AMCShieldSecurity.fetch: $($jsContent.Contains('AMCShieldSecurity.fetch'))"
            Write-Result "Frontend uses relative /api path: $($jsContent.Contains('/api'))"
        } else { Write-Result "Model Training frontend JS missing." }

        if (Test-Path $security) {
            $secContent = Get-Content -LiteralPath $security -Raw
            Write-Result "Security wrapper defines AMCShieldSecurity: $($secContent.Contains('AMCShieldSecurity'))"
            Write-Result "Security wrapper includes X-CSRF-Token: $($secContent.Contains('X-CSRF-Token'))"
        } else { Write-Result "security.js missing." }

        if (Test-Path $service) {
            $svcContent = Get-Content -LiteralPath $service -Raw
            Write-Result "Training service contains process/lifecycle indicators: $($svcContent -match 'Popen|subprocess|start_training|stop_training')"
            Write-Result "Training service contains progress indicators: $($svcContent -match 'progress|current_epoch|current_batch')"
        }
    }

    Invoke-SafeCheck "7. Dataset and checkpoint safety checks" {
        $dataset = "data\raw\GOLD_XYZ_OSC.0001_1024.hdf5"
        if (Test-Path -LiteralPath $dataset) {
            $size = (Get-Item -LiteralPath $dataset).Length
            Write-Result ("HDF5 dataset present; size: {0:N0} bytes. It was NOT opened." -f $size)
        } else {
            Write-Result "HDF5 dataset not found at expected path; it was not accessed."
        }
        foreach ($dir in @("checkpoints","results","logs","data")) {
            if (Test-Path -LiteralPath $dir -PathType Container) {
                $count = (Get-ChildItem -LiteralPath $dir -File -Recurse -ErrorAction SilentlyContinue | Measure-Object).Count
                Write-Result "$dir directory present; file count: $count"
            } else {
                Write-Result "$dir directory not found."
            }
        }
        Write-Result "No dataset, checkpoint, result, log, or database files were modified."
    }

    Invoke-SafeCheck "8. Optional tests" {
        if ($RunPytest) {
            $testFiles = Get-ChildItem -Path . -Filter "test_*.py" -File -Recurse -ErrorAction SilentlyContinue |
                Where-Object { $_.FullName -notmatch '\\.venv\\|\\site-packages\\|\\node_modules\\' }
            Write-Result "Discovered test files: $($testFiles.Count)"
            if ($testFiles.Count -eq 0) {
                Write-Result "No test_*.py files found; pytest not run."
            } else {
                Write-Result "Running pytest because -RunPytest was explicitly supplied."
                & $script:PythonExe -m pytest -q 2>&1 | ForEach-Object { Write-Result "$_" }
                Write-Result "Pytest exit code: $LASTEXITCODE"
            }
        } else {
            Write-Result "Pytest skipped. Re-run with -RunPytest only after reviewing test contents for training/data side effects."
        }
    }

    Write-Section "9. Summary and next action"
    Write-Result "Audit completed: $(Get-Date -Format o)"
    Write-Result "Report: $reportFile"
    Write-Result "Transcript: $transcriptFile"
    Write-Result "This script does not implement missing application features; it establishes a safe, evidence-based baseline."
    Write-Result "No model training was started."
}
finally {
    Stop-Transcript | Out-Null
}

Write-Host ""
Write-Host "AMCShield safe audit completed." -ForegroundColor Green
Write-Host "Report: $reportFile"
Write-Host "Transcript: $transcriptFile"
Write-Host "No training job was launched."
