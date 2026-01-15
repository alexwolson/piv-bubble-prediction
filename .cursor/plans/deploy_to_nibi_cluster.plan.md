---
name: Deploy PIV Bubble Prediction to DRAC Nibi Compute Cluster
overview: Configure the PIV bubble prediction project to run on DRAC's nibi compute cluster, including SLURM job scripts, environment setup, and data transfer strategies.
todos: []
---

# Deploy PIV Bubble Prediction to DRAC Nibi Compute Cluster

## Overview

This plan covers deploying and running the PIV bubble prediction project on DRAC's nibi compute cluster. The nibi cluster is a general-purpose HPC system with 134,400 CPU cores and 288 H100 NVIDIA GPUs, making it ideal for training deep learning models.

## Current Project Status

The project is currently configured for local development on Apple Silicon (M4 Pro) with:
- MPS device support for Apple Silicon
- Memory configurations optimized for 24GB RAM
- Local data paths (`data/raw/all_experiments.zarr/`)
- Python environment managed via `uv`

## Nibi Cluster Details

Based on alliance-docs documentation:
- **SSH login node**: `nibi.alliancecan.ca`
- **Web interface**: `ondemand.sharcnet.ca` (Open OnDemand)
- **Portal**: `portal.nibi.sharcnet.ca`
- **GPU nodes**: 36 nodes with 8× NVIDIA H100 SXM (80 GB) each
- **CPU nodes**: 700 nodes with 192 cores, 748GB RAM each
- **Storage**: 25 PB parallel storage (VAST Data SSD)
  - `/home`: User home directories
  - `/project`: Project space (user directories not created by default)
  - `/scratch`: Scratch space (1TB soft quota, 60-day grace period)
- **Job scheduler**: SLURM Workload Manager
- **Module system**: Lmod (Nix-based)

## Implementation Tasks

### Task 1: Create SLURM Job Scripts

**Files to Create**: `scripts/slurm/`

1. **Training job script** (`scripts/slurm/train.sh`)
   - Request GPU resources (H100)
   - Set appropriate time limits
   - Configure memory requirements
   - Load required modules (Python, PyTorch, CUDA)
   - Activate virtual environment
   - Run training script

2. **Hyperparameter tuning job script** (`scripts/slurm/tune.sh`)
   - Similar to training but for Optuna hyperparameter search
   - May need longer time limits or array jobs

3. **Evaluation job script** (`scripts/slurm/evaluate.sh`)
   - For running evaluation on trained models
   - May use CPU nodes if GPU not required

**Key SLURM Directives**:
```bash
#!/bin/bash
#SBATCH --account=def-<account-name>        # Required: your DRAC account
#SBATCH --time=04:00:00                     # Time limit (4 hours)
#SBATCH --gres=gpu:1                        # Request 1 GPU (H100)
#SBATCH --cpus-per-task=16                  # CPU cores per task
#SBATCH --mem=64G                           # Memory per node
#SBATCH --job-name=piv-train                # Job name
#SBATCH --output=logs/train_%j.out          # Output file
#SBATCH --error=logs/train_%j.err           # Error file
```

### Task 2: Update Device Configuration

**File to Modify**: `src/config.py`

**Changes**:
1. Update `get_device()` to prioritize CUDA over MPS
   - On nibi: CUDA > CPU (no MPS available)
   - Device priority should be: CUDA > CPU

2. Update memory recommendations for GPU nodes
   - GPU nodes have different memory characteristics
   - Adjust batch size recommendations for H100 GPUs

3. Add cluster detection logic
   - Detect if running on nibi cluster (e.g., check `$SLURM_JOB_NODELIST` or hostname)
   - Apply cluster-specific configurations automatically

**Key Changes**:
```python
def get_device(force_cpu: bool = False) -> torch.device:
    """Get best available device, prioritizing CUDA on HPC clusters."""
    if force_cpu:
        return torch.device("cpu")
    
    # On HPC clusters (like nibi), prioritize CUDA
    if torch.cuda.is_available():
        logger.info(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        # Only on Apple Silicon (local development)
        logger.info("Using MPS (Metal Performance Shaders) device")
        return torch.device("mps")
    else:
        logger.info("Using CPU device")
        return torch.device("cpu")
```

### Task 3: Environment Setup on Nibi

**Files to Create**:
1. **Module configuration** (`scripts/nibi/setup_modules.sh`)
   - Load required modules (Python, PyTorch, CUDA)
   - Example:
     ```bash
     module purge
     module load python/3.11
     module load cuda/12.1
     module load pytorch/2.1.0
     ```

2. **Virtual environment setup** (`scripts/nibi/setup_venv.sh`)
   - Option A: Use `uv` if available on nibi
   - Option B: Use standard Python venv + pip
   - Install project dependencies from `pyproject.toml`

3. **Environment activation script** (`scripts/nibi/activate_env.sh`)
   - Load modules
   - Activate virtual environment
   - Set environment variables

### Task 4: Data Transfer Strategy

**Files to Create**: `scripts/transfer/`

1. **Data transfer script** (`scripts/transfer/sync_data.sh`)
   - Use `rsync` or `scp` to transfer Zarr data
   - Transfer to `/project` or `/scratch` (not `/home` if data is large)
   - Handle incremental updates
   - Preserve Zarr structure

2. **Data location configuration**
   - Update paths in code to support cluster paths
   - Use environment variables or config files for path resolution
   - Support both local (`data/raw/`) and cluster (`/project/...` or `/scratch/...`) paths

**Configuration**:
```python
# src/constants.py or config file
import os

# Detect if running on nibi cluster
ON_NIBI = "nibi" in os.environ.get("SLURM_JOB_PARTITION", "").lower() or \
          "nibi" in os.environ.get("SLURM_JOB_NODELIST", "").lower()

# Set data paths based on environment
if ON_NIBI:
    ZARR_PATH = os.environ.get("PIV_DATA_PATH", "/project/<group>/data/raw/all_experiments.zarr/")
else:
    ZARR_PATH = "data/raw/all_experiments.zarr/"
```

### Task 5: Update DataLoader Configuration

**File to Modify**: `src/config.py`

**Changes**:
1. Update `configure_dataloader()` for GPU nodes
   - Set `num_workers` appropriately for compute nodes
   - Enable `pin_memory=True` for CUDA
   - Adjust based on available CPU cores

2. Batch size recommendations for H100 GPUs
   - H100 has 80GB memory, can handle larger batches
   - Update `get_recommended_batch_size()` for GPU nodes

### Task 6: Output and Logging Management

**Files to Create/Modify**:
1. **Output directory structure** (`scripts/slurm/`)
   - Create directory structure for logs, models, checkpoints
   - Use SLURM job ID in paths for uniqueness

2. **Update logging** (`src/train.py`, `src/tune.py`)
   - Ensure logs work with SLURM redirection
   - Use absolute paths for log files
   - Handle output buffering (use `sys.stdout.flush()` or `print(..., flush=True)`)

### Task 7: Create Deployment Documentation

**File to Create**: `docs/NIBI_DEPLOYMENT.md`

**Content**:
1. Prerequisites (DRAC account, SSH access)
2. Initial setup steps
3. Data transfer instructions
4. Environment setup
5. Job submission examples
6. Monitoring jobs
7. Troubleshooting common issues

### Task 8: Test and Validate

**Tasks**:
1. Test SSH connection to nibi
2. Test module loading
3. Test virtual environment creation
4. Test small job submission (quick validation)
5. Test data access
6. Test full training pipeline

## SLURM Job Script Example

**File**: `scripts/slurm/train.sh`

```bash
#!/bin/bash
#SBATCH --account=def-<your-account>      # Replace with your DRAC account
#SBATCH --time=24:00:00                   # 24 hours
#SBATCH --gres=gpu:1                      # 1 H100 GPU
#SBATCH --cpus-per-task=16                # 16 CPU cores
#SBATCH --mem=64G                         # 64GB RAM
#SBATCH --job-name=piv-train
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err

# Load required modules
module purge
module load python/3.11
module load cuda/12.1
module load pytorch/2.1.0

# Activate virtual environment
source $HOME/.venv/piv-bubble-prediction/bin/activate

# Set environment variables
export PIV_DATA_PATH=/project/<group>/data/raw/all_experiments.zarr/
export WANDB_API_KEY=${WANDB_API_KEY}  # Set in .bashrc or job script

# Create output directories
mkdir -p logs models checkpoints

# Run training
python -m src.train \
    --zarr-path $PIV_DATA_PATH \
    --sequence-length 20 \
    --batch-size 32 \
    --epochs 100 \
    --learning-rate 1e-4 \
    --use-wandb \
    --output-dir models/
```

## File Structure

After implementation:

```
piv-bubble-prediction/
├── scripts/
│   ├── slurm/
│   │   ├── train.sh
│   │   ├── tune.sh
│   │   └── evaluate.sh
│   ├── nibi/
│   │   ├── setup_modules.sh
│   │   ├── setup_venv.sh
│   │   └── activate_env.sh
│   └── transfer/
│       └── sync_data.sh
├── src/
│   └── config.py          # Updated for cluster support
├── docs/
│   └── NIBI_DEPLOYMENT.md
└── ...
```

## Environment Variables

Add to `~/.bashrc` on nibi (or set in job scripts):

```bash
# DRAC account
export SLURM_ACCOUNT=def-<your-account>
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT

# Project paths
export PIV_DATA_PATH=/project/<group>/data/raw/all_experiments.zarr/
export PIV_OUTPUT_DIR=/project/<group>/piv-bubble-prediction/outputs/

# Weights & Biases (optional)
export WANDB_API_KEY=<your-key>
```

## Next Steps

1. **Verify DRAC account access**
   - Log into nibi: `ssh nibi.alliancecan.ca`
   - Check account: `sacctmgr show user $USER`

2. **Transfer project code**
   - Clone or transfer repository to nibi
   - Place in `/project/<group>/` or user's home

3. **Set up environment**
   - Run `scripts/nibi/setup_modules.sh`
   - Run `scripts/nibi/setup_venv.sh`

4. **Transfer data**
   - Use `scripts/transfer/sync_data.sh` to transfer Zarr data

5. **Test with small job**
   - Submit a short training job to verify setup

6. **Full training run**
   - Submit training job using `scripts/slurm/train.sh`

## References

- [Nibi cluster documentation](https://docs.alliancecan.ca/wiki/Nibi/en)
- [Running jobs documentation](https://docs.alliancecan.ca/wiki/Running_jobs/en)
- [Using GPUs with Slurm](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en)
- [SLURM documentation](https://slurm.schedmd.com/)
