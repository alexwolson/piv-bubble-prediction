# Deploying PIV Bubble Prediction to DRAC Nibi Compute Cluster

This guide covers deploying and running the PIV bubble prediction project on DRAC's nibi compute cluster.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Initial Setup](#initial-setup)
- [Data Transfer](#data-transfer)
- [Environment Setup](#environment-setup)
- [Job Submission](#job-submission)
- [Monitoring Jobs](#monitoring-jobs)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### Account and Access

1. **DRAC Account**: You need a valid DRAC (Digital Research Alliance of Canada) account
   - Apply at: https://ccdb.alliancecan.ca/
   - Ensure you have access to the nibi cluster

2. **SSH Access**: Configure SSH access to nibi
   ```bash
   ssh awolson@nibi.alliancecan.ca
   ```

3. **Account Verification**: Check your account on nibi
   ```bash
   sacctmgr show user $USER
   ```

### Local Requirements

- Git (to clone the repository)
- rsync (for data transfer)
- SSH client configured with your credentials

## Initial Setup

### 1. Clone Repository on Nibi

SSH to nibi and clone your repository:

```bash
ssh awolson@nibi.alliancecan.ca
cd /home/awolson/projects/def-bussmann/awolson/
git clone <repository-url> piv-bubble-prediction
cd piv-bubble-prediction
```

The repository should be cloned at: `/home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction`

### 2. Verify Project Structure

Ensure the project structure is present:

```bash
ls -la
# Should see: src/, scripts/, pyproject.toml, etc.
```

## Data Transfer

### Transfer Zarr Archive

Use the provided transfer script to sync your Zarr data to the cluster:

```bash
# From your local machine
PROJECT_GROUP=mygroup bash scripts/transfer/sync_data.sh
```

**Default Location**: Data will be transferred to:
```
/home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw/
```

**Options:**

1. **Transfer to Project Directory** (default):
   ```bash
   bash scripts/transfer/sync_data.sh
   ```

2. **Transfer to Scratch Space** (temporary, 1TB quota):
   ```bash
   USE_SCRATCH=1 bash scripts/transfer/sync_data.sh
   ```

3. **Custom Paths**:
   ```bash
   bash scripts/transfer/sync_data.sh ./data/raw/all_experiments.zarr/ awolson@nibi.alliancecan.ca:/home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw/
   ```

**Note:** The first transfer may take a while depending on data size. Subsequent transfers will only sync changes (incremental).

**Default Username:** The script defaults to username `awolson`. To use a different username, set the `NIBI_USER` environment variable:
```bash
NIBI_USER=yourusername bash scripts/transfer/sync_data.sh
```

**Note:** The destination directory may need to be created first on nibi:
```bash
ssh awolson@nibi.alliancecan.ca
mkdir -p /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw
```

### Verify Data on Cluster

After transfer, verify data is accessible:

```bash
ssh awolson@nibi.alliancecan.ca
ls -lh /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw/all_experiments.zarr/
# or for scratch space
ls -lh /scratch/awolson/data/raw/all_experiments.zarr/
```

## Environment Setup

### 1. Load Required Modules

Load Python, CUDA, and PyTorch modules:

```bash
source scripts/nibi/setup_modules.sh
```

Or manually:

```bash
module purge
module load python/3.11
module load cuda/12.1
module load pytorch/2.1.0
```

### 2. Create Virtual Environment

Create and activate a virtual environment:

```bash
bash scripts/nibi/setup_venv.sh
```

This will:
- Check for `uv` (faster) or use standard `venv`
- Create virtual environment at `$HOME/.venv/piv-bubble-prediction`
- Install all dependencies from `pyproject.toml`

**Custom Location:**
```bash
# Default location: $HOME/.venv/piv-bubble-prediction
# Or specify custom path:
bash scripts/nibi/setup_venv.sh /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/.venv
```

### 3. Set Environment Variables

Add to `~/.bashrc` on nibi:

```bash
# DRAC account
export SLURM_ACCOUNT=def-<your-account>
export SBATCH_ACCOUNT=$SLURM_ACCOUNT
export SALLOC_ACCOUNT=$SLURM_ACCOUNT

# Project paths
export PIV_DATA_PATH=/home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/data/raw/all_experiments.zarr/
export PIV_OUTPUT_DIR=/home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction/outputs/

# Weights & Biases (optional)
export WANDB_API_KEY=<your-key>
```

Reload:
```bash
source ~/.bashrc
```

### 4. Verify Setup

Activate environment and verify:

```bash
source scripts/nibi/activate_env.sh
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Job Submission

### 1. Update SLURM Scripts

Edit SLURM job scripts to set your account and paths:

```bash
cd scripts/slurm/
```

Update `train.sh`, `tune.sh`, and `evaluate.sh`:
- Replace `def-<your-account>` with your DRAC account
- Update `PIV_DATA_PATH` if using a different location
- Adjust resource requests (time, memory, CPUs) as needed

### 2. Submit Training Job

```bash
cd /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction
sbatch scripts/slurm/train.sh
```

You should see:
```
Submitted batch job 123456
```

### 3. Submit Hyperparameter Tuning Job

```bash
sbatch scripts/slurm/tune.sh
```

### 4. Submit Evaluation Job

```bash
# Set model checkpoint path
export MODEL_CHECKPOINT=models/best_model.pt
sbatch scripts/slurm/evaluate.sh
```

## Monitoring Jobs

### Check Job Status

Use `sq` (short for `squeue -u $USER`) to see your jobs:

```bash
sq
```

Output shows:
- Job ID
- Status (PD = pending, R = running, CG = completing)
- Time remaining
- Node allocation

### View Job Output

Logs are written to `logs/` directory:

```bash
# View training log
tail -f logs/train_<job_id>.out

# View error log
tail -f logs/train_<job_id>.err

# View all logs
ls -lh logs/
```

### Cancel Job

```bash
scancel <job_id>
# Cancel all your jobs
scancel -u $USER
```

### Check Job Details

```bash
scontrol show job <job_id>
```

## Troubleshooting

### Common Issues

#### 1. Module Not Found

**Error:** `module: command not found`

**Solution:** Ensure you're in a login shell and modules are initialized:
```bash
ssh awolson@nibi.alliancecan.ca
source /etc/profile.d/modules.sh  # if needed
module avail python  # list available Python modules
```

#### 2. Virtual Environment Not Found

**Error:** Virtual environment activation fails

**Solution:** Create virtual environment:
```bash
bash scripts/nibi/setup_venv.sh
```

#### 3. CUDA Not Available

**Error:** `CUDA not available` in logs

**Solution:**
- SSH to nibi: `ssh awolson@nibi.alliancecan.ca`
- Check GPU is requested in SLURM script (`#SBATCH --gres=gpu:1`)
- Verify CUDA module is loaded: `module list`
- Check GPU availability: `nvidia-smi`

#### 4. Data Path Not Found

**Error:** `FileNotFoundError: Zarr archive not found`

**Solution:**
- Set `PIV_DATA_PATH` environment variable
- Verify data exists: `ls -lh $PIV_DATA_PATH`
- Check SLURM script sets correct path

#### 5. Out of Memory

**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
- Reduce batch size in SLURM script
- Check GPU memory: `nvidia-smi`
- Verify model fits in GPU memory (H100 has 80GB)

#### 6. Job Hangs / No Output

**Issue:** Job appears stuck, no output in log file

**Solution:**
- Check if job is actually running: `squeue -j <job_id>`
- Use `srun` for interactive debugging: `srun --pty bash`
- Check if data loading is stuck (large Zarr files)

#### 7. Account Not Set

**Error:** `You must specify an account`

**Solution:**
- Set `SBATCH_ACCOUNT` in `~/.bashrc`
- Or add `#SBATCH --account=def-<your-account>` to SLURM script

### Getting Help

1. **Check SLURM Logs**: `logs/train_<job_id>.out` and `.err`
2. **Interactive Job**: Test with short interactive job:
   ```bash
   salloc --time=1:0:0 --gres=gpu:1 --cpus-per-task=4 --mem=16G
   # Then run commands interactively
   ```
3. **DRAC Support**: https://docs.alliancecan.ca/ or contact support
4. **Nibi Documentation**: https://docs.alliancecan.ca/wiki/Nibi/en

## Best Practices

1. **Test with Small Jobs**: Submit short test jobs before long runs
2. **Use Project Space**: Store data in `/project/` not `/home/` (size limits)
3. **Monitor Quotas**: Check scratch quota (1TB soft limit on nibi)
4. **Clean Up**: Remove temporary files from scratch regularly
5. **Checkpoint Models**: Save checkpoints frequently during training
6. **Use WandB**: Enable Weights & Biases for experiment tracking
7. **Version Control**: Commit code changes before submitting jobs

## Resource Recommendations

### For Training:
- **GPU**: 1× H100 (80GB) - `--gres=gpu:1`
- **CPUs**: 16 cores - `--cpus-per-task=16`
- **Memory**: 64GB - `--mem=64G`
- **Time**: 24 hours - `--time=24:00:00`

### For Hyperparameter Tuning:
- Same as training, but longer time limit (48 hours)

### For Evaluation:
- **CPU**: 8 cores (GPU not required) - `--cpus-per-task=8`
- **Memory**: 32GB - `--mem=32G`
- **Time**: 4 hours - `--time=4:00:00`

## Quick Reference

```bash
# Setup (one-time, on nibi)
cd /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction
source scripts/nibi/setup_modules.sh
bash scripts/nibi/setup_venv.sh
source scripts/nibi/activate_env.sh

# Transfer data (from local machine)
bash scripts/transfer/sync_data.sh

# Submit jobs (on nibi)
cd /home/awolson/projects/def-bussmann/awolson/piv-bubble-prediction
sbatch scripts/slurm/train.sh
sbatch scripts/slurm/tune.sh
sbatch scripts/slurm/evaluate.sh

# Monitor
sq  # list jobs
tail -f logs/train_<job_id>.out  # view output

# Cancel
scancel <job_id>
```

## References

- [Nibi Cluster Documentation](https://docs.alliancecan.ca/wiki/Nibi/en)
- [Running Jobs on SLURM](https://docs.alliancecan.ca/wiki/Running_jobs/en)
- [PyTorch on Alliance Clusters](https://docs.alliancecan.ca/wiki/PyTorch)
- [Using GPUs with Slurm](https://docs.alliancecan.ca/wiki/Using_GPUs_with_Slurm/en)
- [SLURM Documentation](https://slurm.schedmd.com/)
