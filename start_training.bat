@echo off

cd /d "D:\MediCaps University\7 SEM\Project 1\AMCShield"

call ".venv\Scripts\activate.bat"

rem --- Refuse to train on CPU (would take weeks) -------------------
python -c "import sys, torch; sys.exit(0 if torch.cuda.is_available() else 1)"
if errorlevel 1 (
    echo.
    echo ERROR: PyTorch cannot see your GPU ^(CUDA not available^).
    echo Training was NOT started. See instructions to install the CUDA build of PyTorch.
    echo.
    pause
    exit /b 1
)

rem --- Auto resume --------------------------------------------------
rem First run  : no resume file exists, so training starts at epoch 1.
rem After crash: just run this file again. It continues from the exact
rem              epoch and batch saved in checkpoints\baseline_resume.pt.
rem To force a brand-new run from epoch 1, delete
rem checkpoints\baseline_resume.pt  (do NOT set AMC_FRESH here).

set PYTHONUTF8=1
set PYTHONIOENCODING=utf-8
python -m src.train_baseline >> "training.log" 2>&1