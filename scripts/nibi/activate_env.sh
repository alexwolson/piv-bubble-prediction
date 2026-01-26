#!/bin/bash
# Activate environment for PIV bubble prediction on nibi cluster
#
# Usage: source scripts/nibi/activate_env.sh
# Note: This script must be sourced (not executed) to modify the current shell environment

PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
VENV_PATH=${PIV_VENV_PATH:-$HOME/.venv/piv-bubble-prediction}

# Load required modules
if [ -f "$PROJECT_ROOT/scripts/nibi/setup_modules.sh" ]; then
    source "$PROJECT_ROOT/scripts/nibi/setup_modules.sh"
else
    echo "Loading modules manually..."
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load cuda
fi

# Activate virtual environment
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated: $VENV_PATH"
else
    echo "Warning: Virtual environment not found at $VENV_PATH"
    echo "Run: bash scripts/nibi/setup_venv.sh $VENV_PATH"
    return 1 2>/dev/null || exit 1
fi

# Set default environment variables if not already set
export PIV_DATA_PATH=${PIV_DATA_PATH:-/project/<group>/data/raw/all_experiments.zarr/}
export SLURM_ACCOUNT=${SLURM_ACCOUNT:-def-<your-account>}

echo "Environment activated successfully!"
echo "Data path: $PIV_DATA_PATH"
echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'not available')"
