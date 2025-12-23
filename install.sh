#!/bin/bash

# Dual-Arm Franka Manipulation Installation Script

set -e

echo "=== Dual-Arm Franka Manipulation Installer ==="
echo ""

# Install system dependencies for rendering
echo "Installing system dependencies for offscreen rendering..."
if command -v apt-get &> /dev/null; then
    sudo apt-get update
    sudo apt-get install -y libegl1-mesa-dev libgles2-mesa-dev libosmesa6-dev libglu1-mesa-dev libgl1-mesa-dev freeglut3-dev build-essential cmake
elif command -v yum &> /dev/null; then
    sudo yum install -y mesa-libEGL-devel mesa-libGLES-devel mesa-libOSMesa-devel mesa-libGLU-devel freeglut-devel gcc gcc-c++ make cmake
else
    echo "Warning: Could not detect package manager. Please manually install OpenGL development libraries and build tools."
fi

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

# Set environment variables for rendering
echo "Setting environment variables for offscreen rendering..."
export MUJOCO_GL=osmesa
echo 'export MUJOCO_GL=osmesa' >> ~/.bashrc

# Install PyTorch (adjust CUDA version as needed)
echo "Installing PyTorch..."
pip install torch==2.1.0 torchvision==0.16.0 torch-cluster -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install torch-scatter (CUDA version matching PyTorch)
echo "Installing torch-scatter..."
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.1.0+cu121.html

# Install GraspGen
echo "Installing GraspGen..."
cd deps/GraspGen
pip install -e .

# Install PointNet dependency
echo "Installing PointNet++ operations..."
cd pointnet2_ops
pip install --no-build-isolation .
cd ..

# Install additional GraspGen dependencies
echo "Installing additional GraspGen dependencies..."
pip install pyrender PyOpenGL==3.1.5 transformers tensordict diffusers==0.11.1 timm huggingface-hub==0.25.2 scene-synthesizer[recommend] hydra-core trimesh addict spconv-cu120 torch-geometric yourdfpy

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