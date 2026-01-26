#!/bin/bash
#SBATCH --account=def-bussmann            # DRAC account
#SBATCH --time=48:00:00                   # Time limit (48 hours - longer for hyperparameter tuning)
#SBATCH --gres=gpu:1                      # Request 1 H100 GPU
#SBATCH --cpus-per-task=16                # CPU cores per task
#SBATCH --mem=64G                         # Memory per node
#SBATCH --job-name=piv-tune
#SBATCH --output=logs/tune_%j.out
#SBATCH --error=logs/tune_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Load required modules
module purge
module load StdEnv/2023
module load python/3.11
module load cuda

# Print module information
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'NVCC not found')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())')"

# Activate virtual environment
# Update this path to match your virtual environment location
if [ -d "$HOME/.venv/piv-bubble-prediction" ]; then
    source $HOME/.venv/piv-bubble-prediction/bin/activate
elif [ -d "$HOME/venv/piv-bubble-prediction" ]; then
    source $HOME/venv/piv-bubble-prediction/bin/activate
elif [ -d "$PROJECT/venv/piv-bubble-prediction" ]; then
    source $PROJECT/venv/piv-bubble-prediction/bin/activate
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Set environment variables
# Default: /scratch/$USER/data/raw/all_experiments.zarr/
export PIV_DATA_PATH=${PIV_DATA_PATH:-/scratch/$USER/data/raw/all_experiments.zarr/}
export WANDB_API_KEY=${WANDB_API_KEY:-}  # Set in ~/.bashrc or job script

# Create output directories
mkdir -p logs optuna_studies

# Print configuration
echo "Data path: $PIV_DATA_PATH"
echo "Output directory: optuna_studies"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU info not available')"

# Run hyperparameter tuning
# Adjust arguments as needed for your specific use case
python -m src.tune \
    --zarr-path "$PIV_DATA_PATH" \
    --sequence-length 20 \
    --n-trials 50 \
    --study-name cnn_lstm_hyperopt \
    --storage sqlite:///optuna_study.db \
    --pruning \
    --direction minimize \
    --objective validation_loss \
    --epochs 30 \
    --patience 10 \
    --output-dir optuna_studies/ \
    --use-wandb \
    --wandb-project piv-bubble-prediction \
    || { echo "Hyperparameter tuning failed with exit code $?"; exit 1; }

echo "Job completed at: $(date)"
