#!/bin/bash
# Load required modules for PIV bubble prediction on nibi cluster
#
# Usage: source scripts/nibi/setup_modules.sh
# Or: bash scripts/nibi/setup_modules.sh (to see what modules will be loaded)

echo "Setting up modules for PIV bubble prediction on nibi cluster..."

# Purge existing modules to ensure clean environment
module purge

# Load standard environment (StdEnv/2023 includes GCC 12.3, CUDA 12, etc.)
echo "Loading standard environment (StdEnv/2023)..."
module load StdEnv/2023

# Load Python 3.11
echo "Loading Python 3.11..."
module load python/3.11

# Load CUDA (version will be determined by StdEnv/2023, typically 12.x)
echo "Loading CUDA..."
module load cuda

# Note: PyTorch is NOT available as a module
# It must be installed via pip in a virtual environment
# See setup_venv.sh for installation instructions

# Verify module loading
echo ""
echo "Modules loaded successfully!"
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'NVCC not found')"
echo ""
echo "Note: PyTorch must be installed in a virtual environment via pip."
echo "Run: bash scripts/nibi/setup_venv.sh to create venv and install PyTorch."
