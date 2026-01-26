#!/bin/bash
# Create virtual environment for PIV bubble prediction on nibi cluster
#
# Usage: bash scripts/nibi/setup_venv.sh [VENV_PATH]
# If VENV_PATH is not specified, defaults to $HOME/.venv/piv-bubble-prediction

set -e  # Exit on error

VENV_PATH=${1:-$HOME/.venv/piv-bubble-prediction}
PROJECT_ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)

echo "Setting up virtual environment for PIV bubble prediction..."
echo "Virtual environment path: $VENV_PATH"
echo "Project root: $PROJECT_ROOT"

# Load required modules first
if [ -f "$PROJECT_ROOT/scripts/nibi/setup_modules.sh" ]; then
    echo "Loading required modules..."
    source "$PROJECT_ROOT/scripts/nibi/setup_modules.sh"
else
    echo "Warning: setup_modules.sh not found. Make sure to load modules manually."
    module purge
    module load StdEnv/2023
    module load python/3.11
    module load cuda
fi

# Check if uv is available
if command -v uv &> /dev/null; then
    echo "Using uv to create virtual environment..."
    uv venv "$VENV_PATH" --python python3.11
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Install dependencies using uv
    echo "Installing dependencies with uv..."
    cd "$PROJECT_ROOT"
    uv pip install -e .
    
    echo "Virtual environment created successfully with uv!"
else
    echo "uv not available, using standard venv + pip..."
    
    # Create virtual environment
    python -m venv "$VENV_PATH"
    
    # Activate virtual environment
    source "$VENV_PATH/bin/activate"
    
    # Upgrade pip
    pip install --upgrade pip
    
    # Install dependencies from pyproject.toml
    echo "Installing dependencies from pyproject.toml..."
    cd "$PROJECT_ROOT"
    pip install -e .
    
    echo "Virtual environment created successfully with venv!"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "import torch; import zarr; import optuna; import wandb; print('All dependencies installed successfully!')" || {
    echo "Warning: Some dependencies may not be installed correctly."
}

echo ""
echo "Virtual environment is ready at: $VENV_PATH"
echo "To activate it, run: source $VENV_PATH/bin/activate"
