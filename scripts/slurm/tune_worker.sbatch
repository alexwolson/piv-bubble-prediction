#!/bin/bash
#SBATCH --account=<ACCOUNT>          # Set by submit script via --account
#SBATCH --time=00:15:00              # Default wall time (15 minutes, override in submit script)
#SBATCH --gres=gpu:1                 # Require a single GPU
#SBATCH --cpus-per-task=16           # CPU cores per worker
#SBATCH --mem=64G                    # Memory per worker
#SBATCH --job-name=optuna-worker
#SBATCH --output=logs/optuna_worker_%A_%a.out
#SBATCH --error=logs/optuna_worker_%A_%a.err

# SLURM batch script that runs an Optuna worker.
# Expected environment variables (set by submit_tuning.sh):
#   STUDY_NAME, STORAGE_URL, TRIALS_PER_WORKER, PROJECT_ROOT
#   ZARR_PATH (optional, defaults to environment)
# Optional:
#   VENV_PATH, PYTHON_MODULE, MAX_EPOCHS, PATIENCE, PRUNING
#   DISABLE_WANDB, WANDB_MODE_OVERRIDE, WANDB_PROJECT, WANDB_ENTITY
#   SQLITE_TIMEOUT, OPTUNA_WORKER_VERBOSITY, LOG_DIR

set -euo pipefail

echo "=========================================="
echo "Optuna Worker Information"
echo "=========================================="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID:-N/A}"
echo "Node: ${SLURM_NODELIST}"
echo "Start time: $(date '+%Y-%m-%d %H:%M:%S')"
echo "Study: ${STUDY_NAME}"
echo "Storage: ${STORAGE_URL}"
echo "Trials per worker: ${TRIALS_PER_WORKER}"
echo "=========================================="

# Load required modules
module purge
module load StdEnv/2023
module load python/3.11
module load cuda

# Print module information
echo "Python version: $(python --version)"
echo "CUDA version: $(nvcc --version 2>/dev/null | grep 'release' || echo 'NVCC not found')"
echo "PyTorch version: $(python -c 'import torch; print(torch.__version__)' 2>/dev/null || echo 'PyTorch not available')"
echo "CUDA available: $(python -c 'import torch; print(torch.cuda.is_available())' 2>/dev/null || echo 'PyTorch not available')"

# Activate virtual environment
# Update this path to match your virtual environment location
VENV_PATH=${VENV_PATH:-$HOME/.venv/piv-bubble-prediction}
if [ -d "$VENV_PATH" ]; then
    source "$VENV_PATH/bin/activate"
    echo "Virtual environment activated: $VENV_PATH"
elif [ -d "$HOME/venv/piv-bubble-prediction" ]; then
    source "$HOME/venv/piv-bubble-prediction/bin/activate"
    echo "Virtual environment activated: $HOME/venv/piv-bubble-prediction"
else
    echo "Warning: Virtual environment not found. Using system Python."
fi

# Verify GPU is available
if command -v nvidia-smi &> /dev/null; then
    echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'GPU info not available')"
else
    echo "Warning: nvidia-smi not found. GPU may not be available."
fi

# Set project root if not set
PROJECT_ROOT=${PROJECT_ROOT:-$(pwd)}
export PYTHONPATH="${PROJECT_ROOT}:${PYTHONPATH:-}"

# Set data path
ZARR_PATH=${ZARR_PATH:-${PIV_DATA_PATH:-}}
if [ -z "$ZARR_PATH" ]; then
    # Default cluster path
    ZARR_PATH="/scratch/$USER/data/raw/all_experiments.zarr/"
fi

# Create output directories
mkdir -p logs

# Build command
CMD=(
    python -m src.optuna_worker
    --study-name "${STUDY_NAME}"
    --storage-url "${STORAGE_URL}"
    --trials-per-worker "${TRIALS_PER_WORKER}"
    --zarr-path "${ZARR_PATH}"
)

# Add optional arguments
if [ -n "${MAX_EPOCHS:-}" ]; then
    CMD+=(--epochs "${MAX_EPOCHS}")
fi

if [ -n "${PATIENCE:-}" ]; then
    CMD+=(--patience "${PATIENCE}")
fi

if [ "${PRUNING:-}" = "true" ]; then
    CMD+=(--pruning)
fi

if [ "${AUGMENT:-}" = "true" ]; then
    CMD+=(--augment)
fi

if [ "${USE_WANDB:-}" = "true" ]; then
    CMD+=(--use-wandb)
    if [ -n "${WANDB_PROJECT:-}" ]; then
        CMD+=(--wandb-project "${WANDB_PROJECT}")
    fi
    if [ -n "${WANDB_ENTITY:-}" ]; then
        CMD+=(--wandb-entity "${WANDB_ENTITY}")
    fi
    if [ -n "${WANDB_MODE_OVERRIDE:-}" ]; then
        CMD+=(--wandb-mode "${WANDB_MODE_OVERRIDE}")
    fi
fi

if [ "${DISABLE_WANDB:-}" = "true" ]; then
    export DISABLE_WANDB=true
fi

if [ -n "${SQLITE_TIMEOUT:-}" ]; then
    CMD+=(--sqlite-timeout "${SQLITE_TIMEOUT}")
fi

if [ -n "${OPTUNA_WORKER_VERBOSITY:-}" ]; then
    CMD+=(--verbosity "${OPTUNA_WORKER_VERBOSITY}")
fi

# Print configuration
echo ""
echo "Configuration:"
echo "  Data path: ${ZARR_PATH}"
echo "  Project root: ${PROJECT_ROOT}"
echo "  Worker ID: ${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID:-N/A}"
echo ""
echo "Executing command:"
printf ' %q' "${CMD[@]}"
echo ""
echo ""

# Run worker
"${CMD[@]}"
EXIT_CODE=$?

echo ""
echo "Worker finished with exit code ${EXIT_CODE} at $(date '+%Y-%m-%d %H:%M:%S')"

exit "${EXIT_CODE}"
