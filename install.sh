#!/bin/bash

# Dual-Arm Franka Manipulation — Conda-Only Installation Script
#
# Works on local and remote (headless) systems without sudo.
# Prerequisites: conda (Miniconda or Anaconda), git, NVIDIA GPU + driver.
#
# Usage:
#   git clone --recursive <repo-url>
#   cd Bimanual-Manipulation
#   bash install.sh

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

ENV_NAME="dual_arm_manipulation"
PYTHON_VERSION="3.10"
CUDA_VERSION="12.1"
PYTORCH_VERSION="2.1.0"
TORCHVISION_VERSION="0.16.0"

echo "============================================================"
echo "  Dual-Arm Franka Manipulation — Installer"
echo "  (conda-only, no sudo required)"
echo "============================================================"
echo ""

# ── Pre-flight checks ──────────────────────────────────────────
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    echo "  https://docs.conda.io/en/latest/miniconda.html"
    exit 1
fi

if ! command -v git &> /dev/null; then
    echo "Error: git is not available."
    exit 1
fi

# Check for NVIDIA GPU (optional — CPU-only will still work but slower)
if command -v nvidia-smi &> /dev/null; then
    echo "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
else
    echo "Warning: nvidia-smi not found. GPU acceleration may not be available."
    echo "         PyTorch will still be installed with CUDA ${CUDA_VERSION} support."
fi

# ── Step 1: Git submodules ─────────────────────────────────────
echo ""
echo "[1/9] Initializing git submodules (GraspGen, YCB_Dataset)..."
git submodule update --init deps/GraspGen deps/YCB_Dataset

# ── Step 2: Clone GraspGen pre-trained models ──────────────────
echo ""
echo "[2/9] Downloading GraspGen pre-trained models..."
if [ -d "deps/GraspGenModels/checkpoints" ]; then
    echo "  GraspGenModels already present — skipping."
else
    git clone https://huggingface.co/adithyamurali/GraspGenModels deps/GraspGenModels
fi

# ── Step 3: Create conda environment ──────────────────────────
echo ""
echo "[3/9] Creating conda environment '${ENV_NAME}' (Python ${PYTHON_VERSION})..."
eval "$(conda shell.bash hook)"

if conda env list | grep -qw "^${ENV_NAME} "; then
    echo "  Environment '${ENV_NAME}' already exists — skipping creation."
else
    conda create -n "${ENV_NAME}" python="${PYTHON_VERSION}" -y
fi
conda activate "${ENV_NAME}"

# ── Step 4: System-level deps via conda (no sudo) ─────────────
echo ""
echo "[4/9] Installing system libraries via conda (CUDA toolkit, Mesa, Eigen, cmake)..."
conda install -y -c "nvidia/label/cuda-${CUDA_VERSION}.0" cuda-toolkit 2>/dev/null || \
    conda install -y -c "nvidia/label/cuda-${CUDA_VERSION}.1" cuda-toolkit
conda install -y -c conda-forge mesalib glew cmake eigen

# Persist environment variables so they survive conda activate/deactivate
# Auto-detect display: use GLX if a display is available, EGL otherwise
if [[ -n "${DISPLAY:-}" ]] || [[ -n "${WAYLAND_DISPLAY:-}" ]]; then
    MUJOCO_GL_MODE="glx"
    echo "  Display detected (DISPLAY=${DISPLAY:-}${WAYLAND_DISPLAY:+, WAYLAND=$WAYLAND_DISPLAY}) → MUJOCO_GL=glx"
else
    MUJOCO_GL_MODE="egl"
    echo "  No display detected (headless/remote) → MUJOCO_GL=egl"
fi
conda env config vars set MUJOCO_GL="${MUJOCO_GL_MODE}" -n "${ENV_NAME}"
conda env config vars set CPLUS_INCLUDE_PATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3" -n "${ENV_NAME}"
conda env config vars set C_INCLUDE_PATH="${CONDA_PREFIX}/include:${CONDA_PREFIX}/include/eigen3" -n "${ENV_NAME}"

# Re-activate to pick up the new env vars
conda deactivate
conda activate "${ENV_NAME}"

# ── Step 5: PyTorch + CUDA extensions ─────────────────────────
echo ""
echo "[5/9] Installing PyTorch ${PYTORCH_VERSION} + CUDA ${CUDA_VERSION}..."
pip install torch==${PYTORCH_VERSION} torchvision==${TORCHVISION_VERSION} \
    --index-url "https://download.pytorch.org/whl/cu${CUDA_VERSION//./}"

pip install "setuptools<70.0.0"

echo "  Installing torch-scatter and torch-cluster..."
pip install torch-scatter torch-cluster \
    -f "https://data.pyg.org/whl/torch-${PYTORCH_VERSION}+cu${CUDA_VERSION//./}.html"

# Pin numpy <2.0 for PyTorch ${PYTORCH_VERSION} ABI compatibility
pip install "numpy>=1.21.0,<2.0.0"

# ── Step 6: MuJoCo ────────────────────────────────────────────
echo ""
echo "[6/9] Installing MuJoCo..."
pip install mujoco==3.9

# ── Step 7: GraspGen + PointNet2 CUDA ops ─────────────────────
echo ""
echo "[7/9] Installing GraspGen..."
cd deps/GraspGen
pip install -e .

echo "  Building PointNet2 CUDA operations..."
cd pointnet2_ops
pip install --no-build-isolation .
cd "$SCRIPT_DIR"

# Pin PyOpenGL to 3.1.5 (required by GraspGen)
pip install PyOpenGL==3.1.5

# ── Step 8: Main package ──────────────────────────────────────
echo ""
echo "[8/9] Installing bimanual-manipulation (editable)..."
pip install -e .

# Install supplementary packages not in pyproject.toml
pip install pyrender scene-synthesizer
pip install PyOpenGL==3.1.5 PyOpenGL-accelerate

# ── Step 9: Verify ────────────────────────────────────────────
echo ""
echo "[9/9] Verifying installation..."

python -c "
import sys
errors = []

def check(name, test):
    try:
        test()
        print(f'  ✓ {name}')
    except Exception as e:
        errors.append((name, str(e)))
        print(f'  ✗ {name}: {e}')

check('bimanual_manipulation', lambda: __import__('bimanual_manipulation'))
check('torch + CUDA',          lambda: (__import__('torch'), print(f'    torch {__import__(\"torch\").__version__}, CUDA available: {__import__(\"torch\").cuda.is_available()}'))[0])
check('mujoco',                lambda: __import__('mujoco'))
check('robosuite',             lambda: __import__('robosuite'))
check('open3d',                lambda: __import__('open3d'))
check('eaik',                  lambda: __import__('eaik'))
check('mink',                  lambda: __import__('mink'))
check('grasp_gen',             lambda: __import__('grasp_gen'))
check('pointnet2_ops',         lambda: __import__('pointnet2_ops'))
check('torch_scatter',         lambda: __import__('torch_scatter'))
check('torch_cluster',         lambda: __import__('torch_cluster'))
check('pyrender',              lambda: __import__('pyrender'))
check('proxsuite',             lambda: __import__('proxsuite'))

if errors:
    print(f'\n  {len(errors)} check(s) failed — review above.')
    sys.exit(1)
else:
    print('\n  All checks passed!')
"

echo ""
echo "============================================================"
echo "  Installation Complete!"
echo "============================================================"
echo ""
echo "  Environment:  conda activate ${ENV_NAME}"
echo "  Rendering:    MUJOCO_GL=egl (GPU-accelerated, headless-safe)"
echo ""
echo "  Quick test:"
echo "    conda activate ${ENV_NAME}"
echo "    python -c \"import bimanual_manipulation; print('OK')\""
echo ""