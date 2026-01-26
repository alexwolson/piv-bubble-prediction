#!/bin/bash
#SBATCH --account=def-<your-account>      # Replace with your DRAC account
#SBATCH --time=24:00:00                   # Time limit (24 hours)
#SBATCH --gres=gpu:1                      # Request 1 H100 GPU
#SBATCH --cpus-per-task=16                # CPU cores per task
#SBATCH --mem=64G                         # Memory per node
#SBATCH --job-name=piv-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

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
# Update PIV_DATA_PATH to match your data location on the cluster
# Default: /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw/all_experiments.zarr/
export PIV_DATA_PATH=${PIV_DATA_PATH:-/home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw/all_experiments.zarr/}
export WANDB_API_KEY=${WANDB_API_KEY:-}  # Set in ~/.bashrc or job script

# Create output directories
mkdir -p logs models checkpoints

# Print configuration
echo "Data path: $PIV_DATA_PATH"
echo "Output directory: models"
echo "Using GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU info not available')"

# Run training
# Adjust arguments as needed for your specific use case
python -m src.train \
    --zarr-path "$PIV_DATA_PATH" \
    --sequence-length 20 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --output-dir models/ \
    --use-wandb \
    --wandb-project piv-bubble-prediction \
    --split-method experiment \
    --patience 15 \
    || { echo "Training failed with exit code $?"; exit 1; }

echo "Job completed at: $(date)"
