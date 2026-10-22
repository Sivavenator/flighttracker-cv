#!/usr/bin/env bash
# =============================================================================
# FlightTracker CV — Setup Script
# Sets up virtual environment, installs dependencies, downloads base models,
# and scaffolds necessary directories.
# Usage: bash setup.sh [--skip-dataset] [--cpu-only]
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV_DIR="${SCRIPT_DIR}/venv"
PYTHON_BIN="python3"
PIP_BIN="${VENV_DIR}/bin/pip"
PYTHON_VENV="${VENV_DIR}/bin/python"

SKIP_DATASET=false
CPU_ONLY=false

for arg in "$@"; do
  case $arg in
    --skip-dataset) SKIP_DATASET=true ;;
    --cpu-only) CPU_ONLY=true ;;
    *) echo "Unknown argument: $arg"; exit 1 ;;
  esac
done

# ---------------------------------------------------------------------------
# Colour helpers
# ---------------------------------------------------------------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info()    { echo -e "${CYAN}[INFO]${RESET}  $*"; }
success() { echo -e "${GREEN}[OK]${RESET}    $*"; }
warn()    { echo -e "${YELLOW}[WARN]${RESET}  $*"; }
error()   { echo -e "${RED}[ERROR]${RESET} $*" >&2; exit 1; }

echo -e "${BOLD}${CYAN}"
echo "  ███████╗██╗     ██╗ ██████╗ ██╗  ██╗████████╗"
echo "  ██╔════╝██║     ██║██╔════╝ ██║  ██║╚══██╔══╝"
echo "  █████╗  ██║     ██║██║  ███╗███████║   ██║   "
echo "  ██╔══╝  ██║     ██║██║   ██║██╔══██║   ██║   "
echo "  ██║     ███████╗██║╚██████╔╝██║  ██║   ██║   "
echo "  ╚═╝     ╚══════╝╚═╝ ╚═════╝ ╚═╝  ╚═╝   ╚═╝   TRACKER CV"
echo -e "${RESET}"

# ---------------------------------------------------------------------------
# Check Python version
# ---------------------------------------------------------------------------
info "Checking Python version..."
PYTHON_VERSION=$("${PYTHON_BIN}" -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null) \
  || error "python3 not found. Please install Python 3.9 or newer."

MAJOR=$(echo "${PYTHON_VERSION}" | cut -d. -f1)
MINOR=$(echo "${PYTHON_VERSION}" | cut -d. -f2)
if [[ "${MAJOR}" -lt 3 ]] || ([[ "${MAJOR}" -eq 3 ]] && [[ "${MINOR}" -lt 9 ]]); then
  error "Python 3.9+ required. Found ${PYTHON_VERSION}."
fi
success "Python ${PYTHON_VERSION} found."

# ---------------------------------------------------------------------------
# Create virtual environment
# ---------------------------------------------------------------------------
if [[ ! -d "${VENV_DIR}" ]]; then
  info "Creating virtual environment at ${VENV_DIR}..."
  "${PYTHON_BIN}" -m venv "${VENV_DIR}"
  success "Virtual environment created."
else
  info "Virtual environment already exists, reusing."
fi

# ---------------------------------------------------------------------------
# Upgrade pip / setuptools
# ---------------------------------------------------------------------------
info "Upgrading pip and setuptools..."
"${PIP_BIN}" install --quiet --upgrade pip setuptools wheel

# ---------------------------------------------------------------------------
# Install PyTorch (CPU or CUDA)
# ---------------------------------------------------------------------------
if "${PYTHON_VENV}" -c "import torch" 2>/dev/null; then
  success "PyTorch already installed, skipping."
else
  if [[ "${CPU_ONLY}" == "true" ]]; then
    info "Installing PyTorch (CPU-only)..."
    "${PIP_BIN}" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
  else
    # Detect CUDA version
    CUDA_VERSION=""
    if command -v nvcc &>/dev/null; then
      CUDA_VERSION=$(nvcc --version 2>/dev/null | grep -oP 'release \K[\d.]+' | head -1)
    elif command -v nvidia-smi &>/dev/null; then
      CUDA_VERSION=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[\d.]+' | head -1)
    fi

    if [[ -n "${CUDA_VERSION}" ]]; then
      CUDA_MAJOR=$(echo "${CUDA_VERSION}" | cut -d. -f1)
      CUDA_MINOR=$(echo "${CUDA_VERSION}" | cut -d. -f2)
      if [[ "${CUDA_MAJOR}" -ge 12 ]]; then
        info "Installing PyTorch with CUDA 12.x support..."
        "${PIP_BIN}" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
      elif [[ "${CUDA_MAJOR}" -eq 11 && "${CUDA_MINOR}" -ge 8 ]]; then
        info "Installing PyTorch with CUDA 11.8 support..."
        "${PIP_BIN}" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
      else
        warn "CUDA ${CUDA_VERSION} detected but may not have PyTorch wheel. Installing CPU version."
        "${PIP_BIN}" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
      fi
    else
      warn "No CUDA detected. Installing CPU PyTorch."
      "${PIP_BIN}" install --quiet torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
    fi
  fi
  success "PyTorch installed."
fi

# ---------------------------------------------------------------------------
# Install remaining requirements
# ---------------------------------------------------------------------------
info "Installing project requirements from requirements.txt..."
"${PIP_BIN}" install --quiet -r "${SCRIPT_DIR}/requirements.txt"
success "Requirements installed."

# ---------------------------------------------------------------------------
# Create directory structure
# ---------------------------------------------------------------------------
info "Creating directory structure..."
DIRS=(
  "${SCRIPT_DIR}/data/fgvc_yolo/images/train"
  "${SCRIPT_DIR}/data/fgvc_yolo/images/val"
  "${SCRIPT_DIR}/data/fgvc_yolo/images/test"
  "${SCRIPT_DIR}/data/fgvc_yolo/labels/train"
  "${SCRIPT_DIR}/data/fgvc_yolo/labels/val"
  "${SCRIPT_DIR}/data/fgvc_yolo/labels/test"
  "${SCRIPT_DIR}/data/classifier"
  "${SCRIPT_DIR}/models/weights"
  "${SCRIPT_DIR}/logs"
  "${SCRIPT_DIR}/screenshots"
  "${SCRIPT_DIR}/testing/videos"
  "${SCRIPT_DIR}/testing/results"
  "${SCRIPT_DIR}/reports"
)
for d in "${DIRS[@]}"; do
  mkdir -p "${d}"
done
success "Directories created."

# ---------------------------------------------------------------------------
# Download YOLOv8 base weights
# ---------------------------------------------------------------------------
YOLO_WEIGHTS="${SCRIPT_DIR}/models/weights/yolov8m.pt"
if [[ ! -f "${YOLO_WEIGHTS}" ]]; then
  info "Downloading YOLOv8m base weights (this may take a moment)..."
  "${PYTHON_VENV}" -c "
from ultralytics import YOLO
import shutil, os
model = YOLO('yolov8m.pt')
src = 'yolov8m.pt'
dst = '${YOLO_WEIGHTS}'
if os.path.exists(src) and src != dst:
    shutil.move(src, dst)
    print(f'Moved to {dst}')
else:
    print(f'Weights at {dst}')
" && success "YOLOv8m weights ready at ${YOLO_WEIGHTS}." \
  || warn "Could not auto-download YOLOv8m weights. Run: yolo export model=yolov8m.pt"
else
  info "YOLOv8m weights already present."
fi

# ---------------------------------------------------------------------------
# Download/prepare FGVC-Aircraft dataset (optional)
# ---------------------------------------------------------------------------
if [[ "${SKIP_DATASET}" == "false" ]]; then
  FGVC_ROOT="${SCRIPT_DIR}/data/fgvc-aircraft-2013b"
  if [[ -d "${FGVC_ROOT}" ]]; then
    info "FGVC-Aircraft dataset already downloaded."
  else
    info "Downloading FGVC-Aircraft dataset (~2.75 GB)..."
    "${PYTHON_VENV}" -I "${SCRIPT_DIR}/training/prepare_dataset.py" \
      --fgvc-dest "${SCRIPT_DIR}/data" \
      --output "${SCRIPT_DIR}/data/fgvc_yolo" \
      && success "Dataset prepared." \
      || warn "Dataset preparation encountered an error. Check logs."
  fi
else
  info "Skipping dataset download (--skip-dataset)."
fi

# ---------------------------------------------------------------------------
# Install yt-dlp (for test video download)
# ---------------------------------------------------------------------------
if ! command -v yt-dlp &>/dev/null; then
  info "Installing yt-dlp..."
  "${PIP_BIN}" install --quiet yt-dlp
  success "yt-dlp installed."
else
  success "yt-dlp already available."
fi

# ---------------------------------------------------------------------------
# Check camera access
# ---------------------------------------------------------------------------
info "Checking camera device 0..."
"${PYTHON_VENV}" -c "
import cv2, sys
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    cap.release()
    if ret:
        print('Camera OK')
    else:
        print('Camera opened but no frame read')
        sys.exit(1)
else:
    print('Camera not found at device 0')
    sys.exit(1)
" && success "Camera device 0 accessible." \
  || warn "Camera not found or not accessible. Use --source flag to point to a video file."

# ---------------------------------------------------------------------------
# Verify GPU availability
# ---------------------------------------------------------------------------
info "Checking GPU availability..."
"${PYTHON_VENV}" -c "
import torch
if torch.cuda.is_available():
    print(f'CUDA available: {torch.cuda.get_device_name(0)}, VRAM: {torch.cuda.get_device_properties(0).total_memory // (1024**3)} GB')
elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
    print('Apple Silicon MPS available')
else:
    print('No GPU found — running on CPU (slower)')
"

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo -e "${BOLD}  FlightTracker CV setup complete!${RESET}"
echo -e "${BOLD}${GREEN}========================================${RESET}"
echo ""
echo -e "  Start live tracking:  ${CYAN}${VENV_DIR}/bin/python main.py${RESET}"
echo -e "  Run web dashboard:    ${CYAN}${VENV_DIR}/bin/python main.py --web${RESET}"
echo -e "  Test with video:      ${CYAN}${VENV_DIR}/bin/python main.py --source path/to/video.mp4${RESET}"
echo -e "  Train detector:       ${CYAN}${VENV_DIR}/bin/python training/train_detector.py${RESET}"
echo -e "  Train classifier:     ${CYAN}${VENV_DIR}/bin/python training/train_classifier.py${RESET}"
echo -e "  Run test suite:       ${CYAN}${VENV_DIR}/bin/python main.py --test${RESET}"
echo ""
echo -e "  Activate venv:        ${YELLOW}source ${VENV_DIR}/bin/activate${RESET}"
echo ""
