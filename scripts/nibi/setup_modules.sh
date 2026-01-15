#!/bin/bash
# Load required modules for PIV bubble prediction on nibi cluster
#
# Usage: source scripts/nibi/setup_modules.sh
# Or: bash scripts/nibi/setup_modules.sh (to see what modules will be loaded)

echo "Setting up modules for PIV bubble prediction on nibi cluster..."

# Purge existing modules to ensure clean environment
module purge

# Load Python 3.11
echo "Loading Python 3.11..."
module load python/3.11

# Load CUDA 12.1 (or latest available for H100 GPUs)
echo "Loading CUDA 12.1..."
module load cuda/12.1

# Load PyTorch 2.1.0 (or latest available)
echo "Loading PyTorch 2.1.0..."
module load pytorch/2.1.0

# Verify module loading
echo ""
echo "Modules loaded successfully!"
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'NVCC not found')"
if python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())" 2>/dev/null; then
    echo "PyTorch and CUDA are ready!"
else
    echo "Warning: PyTorch import failed. Check module availability."
fi
