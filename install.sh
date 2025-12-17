#!/bin/bash

# Dual-Arm Franka Manipulation Installation Script

set -e

echo "=== Dual-Arm Franka Manipulation Installer ==="
echo ""

# Check if conda is available
if ! command -v conda &> /dev/null; then
    echo "Error: conda is not installed. Please install Miniconda or Anaconda first."
    exit 1
fi

# Create conda environment
echo "Creating conda environment 'dual_arm_manipulation'..."
conda create -n dual_arm_manipulation python=3.10 -y

# Activate environment
echo "Activating environment..."
eval "$(conda shell.bash hook)"
conda activate dual_arm_manipulation

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install GraspGen
echo "Installing GraspGen..."
cd deps/GraspGen
pip install -e .
cd ../..

# Install the main package
echo "Installing dual-arm-franka-manipulation..."
pip install -e .

# Note about models
echo ""
echo "Note: GraspGenModels (pre-trained models) need to be downloaded separately:"
echo "  git clone https://huggingface.co/adithyamurali/GraspGenModels deps/GraspGenModels"
echo "  (or use your own fork if you have custom models)"
echo "Verifying installation..."
python -c "import robosuite_franka_manipulation; print('✓ Package imported successfully')"

echo ""
echo "=== Installation Complete ==="
echo ""
echo "To activate the environment in new shells:"
echo "  conda activate dual_arm_manipulation"
echo ""
echo "To run the manipulation demo:"
echo "  python -m robosuite_franka_manipulation.scripts.manipulation.dual_arm_manipulation --task handoff"
echo ""
echo "To start the prompt service:"
echo "  python -m robosuite_franka_manipulation.services.prompt_service"