# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Machine learning project that predicts bubble counts (primary and secondary) from PIV (Particle Image Velocimetry) velocity field data using a CNN-LSTM architecture. Supports both local development (Apple Silicon MPS or CPU) and HPC cluster deployment (SLURM/CUDA on the "nibi" cluster with H100 GPUs).

## Package Management

Uses `uv` for dependency management. Install dependencies with:
```bash
uv sync
```

## Common Commands

All main scripts are run as Python modules from the repo root:

**Training:**
```bash
python -m src.train --zarr-path data/raw/all_experiments.zarr/ --epochs 100 --use-wandb
```

**Hyperparameter tuning (Optuna):**
```bash
python -m src.tune --zarr-path data/raw/all_experiments.zarr/ --n-trials 50 --storage sqlite:///optuna_study.db --pruning
```

**Evaluation:**
```bash
python -m src.evaluate --zarr-path data/raw/all_experiments.zarr/
```

**Quick smoke test (limit data):**
```bash
python -m src.train --limit 3 --epochs 2
```

## Architecture

### Data Pipeline

PIV experiments are stored in Zarr format (`data/raw/all_experiments.zarr/` locally, `/scratch/$USER/data/raw/all_experiments.zarr/` on cluster). The loading pipeline is:

1. **`src/zarr_reader.py`** – Low-level Zarr I/O: opens archive, finds experiments, loads PIV frames (`u`, `v` velocity components), bubble counts, and alignment indices.
2. **`src/data_loader.py`** – Higher-level loading: normalizes velocity fields per-experiment, stacks `u`/`v` into `(n_frames, height, width, 2)` tensors, creates sliding-window sequences aligned to bubble count timestamps via `alignment_indices`. Supports `lazy=True` mode that returns raw experiment dicts instead of pre-built sequences (crucial for memory efficiency).
3. **`src/dataset.py`** – `PIVBubbleDataset`: PyTorch Dataset that wraps lazy-loaded experiments and generates sequences on-the-fly during training. Supports data augmentation (temporal shifts, Gaussian noise). Converts from `(seq_len, H, W, C)` to `(seq_len, C, H, W)` for PyTorch.

### Model (`src/models/cnn_lstm.py`)

**`CNNEncoder`**: 3 Conv2D layers (32→64→128 channels) with BatchNorm, MaxPool, global average pooling, and linear projection to `cnn_feature_dim`.

**`CNNLSTM`**: For each frame in a sequence, the CNN encoder extracts spatial features. The sequence of features is fed to a (bidirectional) LSTM. The last LSTM output goes through two FC layers with dropout to produce 2 regression outputs (primary and secondary bubble counts).

Default input shape: `(batch, seq_len=20, 2, 22, 30)` → output: `(batch, 2)`.

### Evaluation (`src/evaluate.py`)

Loads `models/best_model.pt` (or a specified checkpoint), reconstructs the model architecture from the saved `args` dict, runs inference on all experiments in a Zarr archive, prints MAE/RMSE/R²/MAPE metrics, and saves `predictions.csv` + `metrics.json` to `results/evaluation/`. Uses lazy loading (same as training). Key flags: `--zarr-path`, `--model-path`, `--limit`, `--no-save`.

### Training (`src/train.py`)

- Loss: MAE (`nn.L1Loss`)
- Optimizer: Adam with weight decay
- Scheduler: `ReduceLROnPlateau` (factor=0.5, patience=10)
- Early stopping: `EarlyStopping` class with `restore_best_weights`
- Preferred split method: `--split-method experiment` (splits at experiment level to prevent data leakage)
- Checkpoints saved as `models/best_model.pt` containing `model_state_dict`, `optimizer_state_dict`, epoch, metrics, and `args` dict
- Optional Weights & Biases logging (`--use-wandb`), optional TensorBoard

### Hyperparameter Tuning (`src/tune.py`)

Optuna study with MedianPruner. Searches over: `learning_rate`, `weight_decay`, `batch_size`, `lstm_hidden_dim`, `lstm_num_layers`, `dropout`, `cnn_feature_dim`. Supports SQLite persistent storage for distributed/resumable tuning. Each trial logs per-epoch metrics to W&B. Best params saved as JSON to `optuna_studies/`.

### Environment Awareness (`src/paths.py`, `src/config.py`)

- Cluster detection via SLURM env vars and hostname (looks for "nibi", "login", "compute", "gpu")
- `get_device()`: CUDA > MPS (Apple Silicon) > CPU, with cluster preferring CUDA only
- `get_zarr_path()`: reads `PIV_DATA_PATH` env var, falls back to cluster scratch or local `data/raw/`
- `get_output_dir()`: reads `PIV_OUTPUT_DIR` env var
- SLURM scripts in `scripts/slurm/` set `PIV_DATA_PATH` automatically

### Key Environment Variables

| Variable | Purpose |
|---|---|
| `PIV_DATA_PATH` | Path to Zarr archive |
| `PIV_OUTPUT_DIR` | Output directory for models/results |
| `PIV_LOG_DIR` | Log directory |
| `WANDB_API_KEY` | Weights & Biases API key |

### HPC Cluster (Nibi)

Setup scripts in `scripts/nibi/`. SLURM job scripts in `scripts/slurm/` — submit with `sbatch scripts/slurm/train.sh`. The account is `def-bussmann`. Data transfer scripts in `scripts/transfer/sync_data.sh`.
