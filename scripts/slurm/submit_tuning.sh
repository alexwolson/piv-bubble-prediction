#!/bin/bash
# Submit a SLURM job array that runs Optuna tuning workers.
#
# Usage:
#   ./submit_tuning.sh --account <ACCOUNT> [options]
#
# Options:
#   --account <ACCOUNT>         SLURM account name (required or use SLURM_ACCOUNT)
#   --study-name <NAME>         Optuna study name (default: cnn_lstm_hyperopt)
#   --storage-path <PATH>       Path to SQLite database (default: optuna_studies/optuna.db)
#   --num-workers <N>           Number of workers / job array size (default: 8)
#   --max-concurrent <N>        Maximum concurrent workers (default: unlimited)
#   --trials-per-worker <N>     Number of sequential trials per worker (default: 1)
#   --zarr-path <PATH>          Path to Zarr archive (default: from environment)
#   --time <HH:MM:SS>           Wall clock limit per worker (default: 00:15:00)
#   --mem <MEM>                 Memory per worker (default: 64G)
#   --cpus <N>                  CPU count per worker (default: 16)
#   --epochs <N>                Maximum epochs per trial (default: 30)
#   --patience <N>              Early stopping patience (default: 10)
#   --pruning                   Enable Optuna pruning
#   --augment                   Enable data augmentation
#   --venv-path <PATH>          Virtual environment path (default: ~/.venv/piv-bubble-prediction)
#   --use-wandb                 Enable Weights & Biases logging
#   --wandb-project <NAME>      WandB project name (default: piv-bubble-prediction)
#   --wandb-entity <NAME>       WandB entity (username or team)
#   --wandb-mode <MODE>         WandB mode override (online/offline)
#   --sqlite-timeout <SECONDS>  SQLite connection timeout (default: 60)
#   --verbosity <LEVEL>         Worker logging level (default: 20 / INFO)
#   --help                      Show this message
#
# Examples:
#   ./submit_tuning.sh --account def-bussmann --study-name cnn_lstm_hyperopt
#   ./submit_tuning.sh --account def-bussmann --num-workers 16 --trials-per-worker 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Defaults
ACCOUNT="${SLURM_ACCOUNT:-}"
STUDY_NAME="cnn_lstm_hyperopt"
STORAGE_PATH=""
NUM_WORKERS=8
MAX_CONCURRENT=""
TRIALS_PER_WORKER=1
ZARR_PATH=""
TIME_LIMIT="00:15:00"
MEM="64G"
CPUS="16"
EPOCHS=""
PATIENCE=""
PRUNING=false
AUGMENT=false
VENV_PATH="${HOME}/.venv/piv-bubble-prediction"
USE_WANDB=false
WANDB_PROJECT="piv-bubble-prediction"
WANDB_ENTITY=""
WANDB_MODE=""
SQLITE_TIMEOUT="60"
VERBOSITY="20"

show_usage() {
    grep "^# " "${0}" | sed 's/^# //' | head -n 50
    exit 0
}

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --account) ACCOUNT="$2"; shift 2 ;;
        --study-name) STUDY_NAME="$2"; shift 2 ;;
        --storage-path) STORAGE_PATH="$2"; shift 2 ;;
        --num-workers) NUM_WORKERS="$2"; shift 2 ;;
        --max-concurrent)
            if [ -z "$2" ] || [ "$2" = "unlimited" ]; then
                MAX_CONCURRENT=""
            else
                MAX_CONCURRENT="$2"
            fi
            shift 2
            ;;
        --trials-per-worker) TRIALS_PER_WORKER="$2"; shift 2 ;;
        --zarr-path) ZARR_PATH="$2"; shift 2 ;;
        --time) TIME_LIMIT="$2"; shift 2 ;;
        --mem) MEM="$2"; shift 2 ;;
        --cpus) CPUS="$2"; shift 2 ;;
        --epochs) EPOCHS="$2"; shift 2 ;;
        --patience) PATIENCE="$2"; shift 2 ;;
        --pruning) PRUNING=true; shift ;;
        --augment) AUGMENT=true; shift ;;
        --venv-path) VENV_PATH="$2"; shift 2 ;;
        --use-wandb) USE_WANDB=true; shift ;;
        --wandb-project) WANDB_PROJECT="$2"; shift 2 ;;
        --wandb-entity) WANDB_ENTITY="$2"; shift 2 ;;
        --wandb-mode) WANDB_MODE="$2"; shift 2 ;;
        --sqlite-timeout) SQLITE_TIMEOUT="$2"; shift 2 ;;
        --verbosity) VERBOSITY="$2"; shift 2 ;;
        --help) show_usage ;;
        *) echo "Unknown option: $1"; show_usage ;;
    esac
done

# Validate required arguments
if [ -z "${ACCOUNT}" ]; then
    echo "Error: --account is required (or set SLURM_ACCOUNT environment variable)"
    exit 1
fi

if ! [[ "${NUM_WORKERS}" =~ ^[0-9]+$ ]] || [ "${NUM_WORKERS}" -le 0 ]; then
    echo "Error: --num-workers must be a positive integer"
    exit 1
fi

if ! [[ "${TRIALS_PER_WORKER}" =~ ^[0-9]+$ ]] || [ "${TRIALS_PER_WORKER}" -le 0 ]; then
    echo "Error: --trials-per-worker must be a positive integer"
    exit 1
fi

# Determine storage path
if [ -z "${STORAGE_PATH}" ]; then
    STORAGE_PATH="${PROJECT_ROOT}/optuna_studies/optuna.db"
fi

# Convert to absolute path
STORAGE_PATH="$(cd "$(dirname "${STORAGE_PATH}")" && pwd)/$(basename "${STORAGE_PATH}")"
STORAGE_DIR="$(dirname "${STORAGE_PATH}")"
STORAGE_URL="sqlite:///${STORAGE_PATH}"

# Create storage directory if needed
mkdir -p "${STORAGE_DIR}"

# Check if study already exists
if [ -f "${STORAGE_PATH}" ]; then
    echo "Warning: Existing Optuna database detected at ${STORAGE_PATH}"
    echo "Warning: New trials will append to this study"
    echo "Warning: If you intended to start fresh, delete or move the file manually"
    echo ""
fi

# Set default ZARR path if not provided
if [ -z "${ZARR_PATH}" ]; then
    ZARR_PATH="${PIV_DATA_PATH:-}"
    if [ -z "${ZARR_PATH}" ]; then
        # Default cluster path
        ZARR_PATH="/home/${USER}/projects/def-bussmann/${USER}/piv-bubble-prediction/data/raw/all_experiments.zarr/"
    fi
fi

# Verify ZARR path exists
if [ ! -d "${ZARR_PATH}" ]; then
    echo "Warning: Zarr path does not exist: ${ZARR_PATH}"
    echo "Warning: Worker may fail if path is incorrect"
    echo ""
fi

# Build array specification
ARRAY_SPEC="1-${NUM_WORKERS}"
if [ -n "${MAX_CONCURRENT}" ]; then
    ARRAY_SPEC="${ARRAY_SPEC}%${MAX_CONCURRENT}"
fi

# Build export variables
EXPORT_VARS="ALL"
EXPORT_VARS="${EXPORT_VARS},PROJECT_ROOT=${PROJECT_ROOT}"
EXPORT_VARS="${EXPORT_VARS},STUDY_NAME=${STUDY_NAME}"
EXPORT_VARS="${EXPORT_VARS},STORAGE_URL=${STORAGE_URL}"
EXPORT_VARS="${EXPORT_VARS},TRIALS_PER_WORKER=${TRIALS_PER_WORKER}"
EXPORT_VARS="${EXPORT_VARS},ZARR_PATH=${ZARR_PATH}"
EXPORT_VARS="${EXPORT_VARS},VENV_PATH=${VENV_PATH}"
EXPORT_VARS="${EXPORT_VARS},SQLITE_TIMEOUT=${SQLITE_TIMEOUT}"
EXPORT_VARS="${EXPORT_VARS},OPTUNA_WORKER_VERBOSITY=${VERBOSITY}"

if [ -n "${EPOCHS:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},MAX_EPOCHS=${EPOCHS}"
fi

if [ -n "${PATIENCE:-}" ]; then
    EXPORT_VARS="${EXPORT_VARS},PATIENCE=${PATIENCE}"
fi

if [ "${PRUNING}" = true ]; then
    EXPORT_VARS="${EXPORT_VARS},PRUNING=true"
fi

if [ "${AUGMENT}" = true ]; then
    EXPORT_VARS="${EXPORT_VARS},AUGMENT=true"
fi

if [ "${USE_WANDB}" = true ]; then
    EXPORT_VARS="${EXPORT_VARS},USE_WANDB=true"
    if [ -n "${WANDB_PROJECT:-}" ]; then
        EXPORT_VARS="${EXPORT_VARS},WANDB_PROJECT=${WANDB_PROJECT}"
    fi
    if [ -n "${WANDB_ENTITY:-}" ]; then
        EXPORT_VARS="${EXPORT_VARS},WANDB_ENTITY=${WANDB_ENTITY}"
    fi
    if [ -n "${WANDB_MODE:-}" ]; then
        EXPORT_VARS="${EXPORT_VARS},WANDB_MODE_OVERRIDE=${WANDB_MODE}"
    fi
fi

# Create logs directory
mkdir -p "${PROJECT_ROOT}/logs"

# Submit job
SBATCH_SCRIPT="${SCRIPT_DIR}/tune_worker.sbatch"

echo "Submitting Optuna tuning job array..."
echo ""

SUBMIT_OUTPUT=$(sbatch \
    --account="${ACCOUNT}" \
    --time="${TIME_LIMIT}" \
    --mem="${MEM}" \
    --cpus-per-task="${CPUS}" \
    --job-name="optuna_tuning" \
    --array="${ARRAY_SPEC}" \
    --output="${PROJECT_ROOT}/logs/optuna_worker_%A_%a.out" \
    --error="${PROJECT_ROOT}/logs/optuna_worker_%A_%a.err" \
    --export="${EXPORT_VARS}" \
    "${SBATCH_SCRIPT}" 2>&1)

if echo "${SUBMIT_OUTPUT}" | grep -q "Submitted batch job"; then
    # Extract job ID
    JOB_ID=$(echo "${SUBMIT_OUTPUT}" | sed -n 's/.*Submitted batch job \([0-9]\+\).*/\1/p')
    
    if [ -z "${JOB_ID}" ]; then
        JOB_ID=$(echo "${SUBMIT_OUTPUT}" | grep -oE '[0-9]+' | head -n 1)
    fi
    
    echo "=========================================="
    echo "Optuna Tuning Submission Summary"
    echo "=========================================="
    echo "Job array base ID: ${JOB_ID}"
    echo "Array specification: ${ARRAY_SPEC}"
    echo "Study name: ${STUDY_NAME}"
    echo "Storage URL: ${STORAGE_URL}"
    echo "Zarr path: ${ZARR_PATH}"
    echo "Trials per worker: ${TRIALS_PER_WORKER}"
    echo "Workers requested: ${NUM_WORKERS}"
    echo "Max concurrent: ${MAX_CONCURRENT:-unlimited}"
    echo "Time limit: ${TIME_LIMIT}"
    echo "Memory: ${MEM}"
    echo "CPUs: ${CPUS}"
    if [ -n "${EPOCHS:-}" ]; then
        echo "Max epochs per trial: ${EPOCHS}"
    fi
    if [ -n "${PATIENCE:-}" ]; then
        echo "Early stopping patience: ${PATIENCE}"
    fi
    echo "Pruning: ${PRUNING}"
    echo "Augmentation: ${AUGMENT}"
    echo "WandB enabled: ${USE_WANDB}"
    if [ "${USE_WANDB}" = true ] && [ -n "${WANDB_PROJECT:-}" ]; then
        echo "WandB project: ${WANDB_PROJECT}"
    fi
    echo "Virtual environment: ${VENV_PATH}"
    echo ""
    echo "Monitor job status:"
    echo "  squeue -j ${JOB_ID}              # All array tasks"
    echo "  squeue -u \$(whoami)             # All your jobs"
    echo ""
    echo "View logs:"
    echo "  tail -f ${PROJECT_ROOT}/logs/optuna_worker_${JOB_ID}_*.out"
    echo "  tail -f ${PROJECT_ROOT}/logs/optuna_worker_${JOB_ID}_*.err"
    echo ""
    echo "Cancel all workers:"
    echo "  scancel ${JOB_ID}"
    echo ""
    echo "View study results:"
    echo "  python -c \"import optuna; study = optuna.load_study(study_name='${STUDY_NAME}', storage='${STORAGE_URL}'); print('Best params:', study.best_params); print('Best value:', study.best_value)\""
    echo ""
    echo "Note: In squeue, array jobs appear as ${JOB_ID}_<index> (e.g., ${JOB_ID}_1, ${JOB_ID}_2)"
    echo "=========================================="
    exit 0
else
    echo "Error: Job submission failed"
    echo "${SUBMIT_OUTPUT}"
    exit 1
fi
