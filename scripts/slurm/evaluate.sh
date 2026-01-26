#!/bin/bash
#SBATCH --account=def-bussmann            # DRAC account
#SBATCH --time=04:00:00                   # Time limit (4 hours - evaluation is typically faster)
#SBATCH --cpus-per-task=8                 # CPU cores (GPU not required for evaluation)
#SBATCH --mem=32G                         # Memory per node
#SBATCH --job-name=piv-eval
#SBATCH --output=logs/evaluate_%j.out
#SBATCH --error=logs/evaluate_%j.err

# Print job information
echo "Job ID: $SLURM_JOB_ID"
echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Working directory: $(pwd)"

# Load required modules
module purge
module load StdEnv/2023
module load python/3.11
# Note: CUDA module may still be needed if evaluating on GPU
# Uncomment if you want to evaluate on GPU instead of CPU
# module load cuda

# Print module information
echo "Python version: $(python --version)"

# Activate virtual environment
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

# Create output directories
mkdir -p logs evaluation_results

# Print configuration
echo "Data path: $PIV_DATA_PATH"
echo "Model checkpoint: ${MODEL_CHECKPOINT:-models/best_model.pt}"
echo "Output directory: evaluation_results"

# Run evaluation
# Update MODEL_CHECKPOINT to point to your trained model
MODEL_CHECKPOINT=${MODEL_CHECKPOINT:-models/best_model.pt}

python -m src.evaluate \
    --model-path "$MODEL_CHECKPOINT" \
    --zarr-path "$PIV_DATA_PATH" \
    --output-dir evaluation_results/ \
    --split test \
    || { echo "Evaluation failed with exit code $?"; exit 1; }

echo "Job completed at: $(date)"
