@echo off
setlocal EnableDelayedExpansion
title FlightTracker CV - Setup

echo.
echo  ================================================
echo   FlightTracker CV - Windows Setup
echo  ================================================
echo.

:: ── Find a compatible Python (3.10, 3.11, or 3.12 only) ───────────────────────
set PYTHON_CMD=
set PY_OK=0

:: Try py launcher first (preferred on Windows)
for %%v in (3.12 3.11 3.10) do (
    if "!PY_OK!"=="0" (
        py -%%v --version >nul 2>&1
        if not errorlevel 1 (
            set PYTHON_CMD=py -%%v
            set PY_VER=%%v
            set PY_OK=1
        )
    )
)

:: Fallback: check if plain 'python' is an acceptable version
if "!PY_OK!"=="0" (
    for /f "tokens=2" %%v in ('python --version 2^>^&1') do set RAW_VER=%%v
    for /f "tokens=1,2 delims=." %%a in ("!RAW_VER!") do (
        if "%%a"=="3" (
            if "%%b"=="10" ( set PYTHON_CMD=python & set PY_VER=!RAW_VER! & set PY_OK=1 )
            if "%%b"=="11" ( set PYTHON_CMD=python & set PY_VER=!RAW_VER! & set PY_OK=1 )
            if "%%b"=="12" ( set PYTHON_CMD=python & set PY_VER=!RAW_VER! & set PY_OK=1 )
        )
    )
)

if "!PY_OK!"=="0" (
    echo.
    echo [ERROR] No compatible Python found.
    echo.
    echo  PyTorch requires Python 3.10, 3.11, or 3.12.
    echo  Your current Python is too new ^(3.13+ not yet supported by PyTorch^).
    echo.
    echo  WHAT TO DO:
    echo  1. Download Python 3.11 from:
    echo     https://www.python.org/downloads/release/python-3119/
    echo     ^(choose "Windows installer 64-bit"^)
    echo.
    echo  2. During install, tick "Add python.exe to PATH"
    echo     and select "Customize" to ensure pip is included.
    echo.
    echo  3. After install, run this setup.bat again.
    echo     It will auto-detect py -3.11.
    echo.
    pause & exit /b 1
)

echo [OK] Using Python %PY_VER% via: %PYTHON_CMD%
echo.

:: ── Check pip ─────────────────────────────────────────────────────────────────
%PYTHON_CMD% -m pip --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] pip not found for Python %PY_VER%.
    echo         Run: %PYTHON_CMD% -m ensurepip
    pause & exit /b 1
)
echo [OK] pip found

:: ── Create virtual environment ────────────────────────────────────────────────
if exist venv (
    echo [SKIP] venv already exists - delete the venv folder and re-run to rebuild
) else (
    echo [....] Creating virtual environment with Python %PY_VER%...
    %PYTHON_CMD% -m venv venv
    if errorlevel 1 ( echo [ERROR] Failed to create venv & pause & exit /b 1 )
    echo [OK] venv created
)

:: ── Activate venv ─────────────────────────────────────────────────────────────
call venv\Scripts\activate.bat
echo [OK] venv activated

:: ── Upgrade pip ───────────────────────────────────────────────────────────────
echo [....] Upgrading pip...
python -m pip install --upgrade pip --quiet

:: ── Detect CUDA ───────────────────────────────────────────────────────────────
set CUDA_VER=
set TORCH_INDEX=https://download.pytorch.org/whl/cpu

nvidia-smi >nul 2>&1
if not errorlevel 1 (
    echo [OK] NVIDIA GPU detected, checking CUDA version...
    for /f "tokens=*" %%i in ('nvidia-smi --query-gpu=driver_version --format=csv^,noheader 2^>nul') do set DRIVER=%%i

    :: Detect CUDA from nvidia-smi output line
    for /f "tokens=9" %%c in ('nvidia-smi 2^>nul ^| findstr /i "CUDA Version"') do set CUDA_VER=%%c

    echo     Driver: !DRIVER!   CUDA: !CUDA_VER!

    :: Pick wheel index based on CUDA version
    :: CUDA 12.x -> cu121 index
    :: CUDA 11.x -> cu118 index
    for /f "tokens=1 delims=." %%m in ("!CUDA_VER!") do (
        if "%%m"=="12" set TORCH_INDEX=https://download.pytorch.org/whl/cu121
        if "%%m"=="11" set TORCH_INDEX=https://download.pytorch.org/whl/cu118
    )
    if "!TORCH_INDEX!"=="https://download.pytorch.org/whl/cpu" (
        echo [WARN] Could not read CUDA version, defaulting to cu121 index
        set TORCH_INDEX=https://download.pytorch.org/whl/cu121
    )
    echo [OK] Using PyTorch index: !TORCH_INDEX!
) else (
    echo [WARN] No NVIDIA GPU detected - using CPU-only PyTorch
    echo        Inference will work but training will be slow.
)

:: ── Install PyTorch ───────────────────────────────────────────────────────────
echo.
echo [....] Installing PyTorch (may take 3-10 minutes on first run)...
pip install torch torchvision torchaudio --index-url %TORCH_INDEX%
if errorlevel 1 (
    echo [WARN] Primary PyTorch install failed, trying CPU fallback...
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    if errorlevel 1 (
        echo [ERROR] PyTorch install failed entirely.
        echo         Check your internet connection and try running:
        echo         pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
        pause & exit /b 1
    )
    echo [OK] PyTorch installed ^(CPU mode^)
) else (
    echo [OK] PyTorch installed
)

:: ── Verify PyTorch ────────────────────────────────────────────────────────────
python -c "import torch; print('[OK] torch', torch.__version__, '- CUDA available:', torch.cuda.is_available())"

:: ── Install remaining requirements ───────────────────────────────────────────
echo.
echo [....] Installing project dependencies...
pip install ultralytics --quiet
pip install opencv-python pillow timm albumentations --quiet
pip install FlightRadarAPI --quiet
pip install numpy pandas scikit-learn matplotlib seaborn --quiet
pip install flask flask-socketio --quiet
pip install pygame pyttsx3 --quiet
pip install pyyaml requests yt-dlp tqdm python-dotenv --quiet
pip install sqlalchemy geopy rich reportlab --quiet
echo [OK] All dependencies installed

:: ── Create directories ────────────────────────────────────────────────────────
echo.
echo [....] Creating project directories...
for %%d in (
    weights
    data\aircraft_detection\images\train
    data\aircraft_detection\images\val
    data\aircraft_detection\images\test
    data\aircraft_detection\labels\train
    data\aircraft_detection\labels\val
    data\aircraft_detection\labels\test
    data\classifier\train
    data\classifier\val
    screenshots
    logs
    runs\detect
    runs\classify
    reports
    test_videos
) do (
    if not exist %%d mkdir %%d
)
echo [OK] Directories created

:: ── Download YOLOv8m weights ──────────────────────────────────────────────────
echo.
if exist weights\yolov8m.pt (
    echo [SKIP] YOLOv8m weights already present
) else (
    echo [....] Downloading YOLOv8m pretrained weights...
    python -c "import os; from ultralytics import YOLO; m=YOLO('yolov8m.pt'); print('[OK] Weights ready')"
    if exist yolov8m.pt (
        move yolov8m.pt weights\yolov8m.pt >nul
    )
    if exist weights\yolov8m.pt (
        echo [OK] Weights saved to weights\yolov8m.pt
    ) else (
        echo [WARN] Weights download may have stored in ultralytics cache ^(that is fine^)
    )
)

:: ── Check webcam ──────────────────────────────────────────────────────────────
echo.
echo [....] Checking webcam on device 0...
python -c "import cv2; cap=cv2.VideoCapture(0); ok,_=cap.read(); cap.release(); print('[OK] Webcam found and working' if ok else '[WARN] No webcam on device 0 - plug in USB camera before running main.py')"

:: ── Done ──────────────────────────────────────────────────────────────────────
echo.
echo  ================================================
echo   Setup Complete!
echo  ================================================
echo.
echo  NEXT STEPS (run in order, venv must be active):
echo.
echo    venv\Scripts\activate
echo.
echo    python training\prepare_dataset.py     ^(download + prep data^)
echo    python training\train_detector.py      ^(~2-4 hrs GPU^)
echo    python training\train_classifier.py    ^(~1 hr GPU^)
echo.
echo    python main.py --dashboard             ^(live + web UI^)
echo    python main.py --test-video            ^(offline test^)
echo    python main.py --generate-report       ^(accuracy report^)
echo.
echo  Web dashboard: http://localhost:5000
echo.
pause
