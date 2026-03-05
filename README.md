# PIV Bubble Prediction

Predicts primary and secondary bubble counts from PIV (Particle Image Velocimetry) velocity field data using a CNN-LSTM deep learning model. Supports local development on Apple Silicon (MPS) or CPU and HPC cluster deployment (SLURM/CUDA, H100 GPUs on the Nibi cluster).

## Setup

**Requirements:** Python ≥ 3.9, [uv](https://github.com/astral-sh/uv)

```bash
uv sync
```

Data should be placed at `data/raw/all_experiments.zarr/` (local) or configured via the `PIV_DATA_PATH` environment variable.

---

## Usage

All scripts are run as Python modules from the repo root.

### Training

```bash
python -m src.train \
  --zarr-path data/raw/all_experiments.zarr/ \
  --epochs 100 \
  --split-method experiment \
  --use-wandb
```

The best checkpoint is saved to `models/best_model.pt`. Key options:

| Flag | Default | Description |
|---|---|---|
| `--zarr-path` | auto-detect | Path to Zarr archive |
| `--sequence-length` | 20 | Frames per input sequence |
| `--stride` | 1 | Sliding window stride |
| `--batch-size` | 16 | Batch size |
| `--epochs` | 100 | Max training epochs |
| `--learning-rate` | 1e-4 | Adam learning rate |
| `--weight-decay` | 1e-5 | Adam weight decay |
| `--patience` | 15 | Early stopping patience |
| `--split-method` | experiment | `experiment` (no leakage) or `random` |
| `--test-split` | 0.2 | Fraction held out for test |
| `--augment` | off | Enable temporal shift + noise augmentation |
| `--config` | — | Load hyperparams from a JSON file (e.g. from tuning) |
| `--use-wandb` | off | Enable Weights & Biases logging |
| `--limit` | — | Cap number of experiments (for quick tests) |

Model architecture flags: `--cnn-feature-dim` (128), `--lstm-hidden-dim` (256), `--lstm-num-layers` (2), `--dropout` (0.5), `--bidirectional` (on).

**Quick smoke test:**
```bash
python -m src.train --limit 3 --epochs 2
```

### Evaluation

```bash
python -m src.evaluate --zarr-path data/raw/all_experiments.zarr/
```

Loads `models/best_model.pt`, reconstructs the model architecture from the saved checkpoint args, runs inference on all experiments, and prints MAE / RMSE / R² / MAPE metrics. Saves `predictions.csv` and `metrics.json` to `results/evaluation/`.

| Flag | Default | Description |
|---|---|---|
| `--zarr-path` | `data/raw/all_experiments.zarr` | Input data |
| `--model-path` | `models/best_model.pt` | Checkpoint to load |
| `--output-dir` | `results/evaluation` | Where to write outputs |
| `--batch-size` | 32 | Inference batch size |
| `--limit` | — | Cap experiments |
| `--no-save` | off | Skip writing outputs to disk |

### Hyperparameter Tuning

```bash
python -m src.tune \
  --zarr-path data/raw/all_experiments.zarr/ \
  --n-trials 50 \
  --storage sqlite:///optuna_study.db \
  --pruning \
  --objective validation_loss
```

Runs an Optuna study searching over `learning_rate`, `weight_decay`, `batch_size`, `cnn_feature_dim`, `lstm_hidden_dim`, `lstm_num_layers`, and `dropout`. Supports MedianPruner to kill unpromising trials early. Results are saved as JSON to `optuna_studies/`. Use `--storage` with a SQLite URL to make the study resumable and share it across distributed workers.

Key options:

| Flag | Default | Description |
|---|---|---|
| `--n-trials` | 50 | Number of Optuna trials |
| `--study-name` | `cnn_lstm_hyperopt` | Study identifier |
| `--storage` | — | SQLite URL for persistent/distributed study |
| `--pruning` | off | Enable MedianPruner |
| `--objective` | `validation_loss` | `validation_loss` or `validation_r2` |
| `--epochs` | 10 | Max epochs per trial |
| `--patience` | epochs // 3 | Early stopping per trial |
| `--use-wandb` | on | Log each trial to W&B |

After tuning, pass the best params JSON directly to training:
```bash
python -m src.train --config optuna_studies/cnn_lstm_tuning_best_params.json --epochs 100
```

---

## Architecture

### Data Pipeline

PIV experiments are stored in Zarr format. Each experiment contains:
- `u`, `v` velocity component arrays — shape `(n_frames, height, width)`
- bubble count arrays — primary and secondary counts per sensor row
- `alignment_indices` — maps each PIV frame index to the corresponding sensor row

The loading pipeline (`src/zarr_reader.py` → `src/data_loader.py` → `src/dataset.py`):

1. **`zarr_reader.py`** — Opens the archive, finds all experiments, loads PIV frames and bubble counts.
2. **`data_loader.py`** — Normalizes `u`/`v` per-experiment (zero mean, unit variance), stacks them into `(n_frames, H, W, 2)` arrays. With `lazy=True` it returns raw experiment dicts instead of pre-built sequences — used in training and evaluation to avoid loading everything into RAM.
3. **`dataset.py`** — `PIVBubbleDataset` takes the list of experiment dicts and generates sliding-window sequences on-the-fly during iteration. Optionally applies data augmentation (random temporal shifts ±N frames, Gaussian noise injection). Outputs `(seq_len, C, H, W)` tensors.

### Model (`src/models/cnn_lstm.py`)

```
Input: (batch, seq_len, 2, 22, 30)   # 2 channels = u, v velocity
         ↓  [for each frame]
    CNNEncoder
      Conv2D(2→32) + BN + ReLU + MaxPool
      Conv2D(32→64) + BN + ReLU + MaxPool
      Conv2D(64→128) + BN + ReLU
      AdaptiveAvgPool → Linear(128 → cnn_feature_dim)
         ↓  [sequence of features]
    Bidirectional LSTM (cnn_feature_dim → lstm_hidden_dim × 2)
         ↓  [last timestep output]
    FC(lstm_hidden_dim×2 → 256) + Dropout
    FC(256 → 128) + Dropout
    FC(128 → 2)
         ↓
Output: (batch, 2)   # [primary_count, secondary_count]
```

Default hyperparameters: `cnn_feature_dim=128`, `lstm_hidden_dim=256`, `lstm_num_layers=2`, `dropout=0.5`, bidirectional LSTM.

### Training Details

- **Loss:** MAE (`nn.L1Loss`)
- **Optimizer:** Adam with weight decay
- **Scheduler:** `ReduceLROnPlateau` — halves LR when validation loss plateaus (patience=10)
- **Early stopping:** Restores best weights when validation loss stops improving
- **Split:** Experiment-level split (recommended) prevents data leakage between train and test sets
- **Checkpoint:** Saves `model_state_dict`, `optimizer_state_dict`, epoch, metrics, and all training args to `models/best_model.pt`

---

## Project Structure

```
├── src/
│   ├── models/
│   │   └── cnn_lstm.py        # CNNEncoder + CNNLSTM architecture
│   ├── zarr_reader.py         # Low-level Zarr I/O
│   ├── data_loader.py         # Normalization, sequence creation, lazy loading
│   ├── dataset.py             # PyTorch Dataset with on-the-fly augmentation
│   ├── train.py               # Training loop, early stopping, logging
│   ├── evaluate.py            # Inference + metrics on a Zarr dataset
│   ├── tune.py                # Optuna hyperparameter search
│   ├── optuna_worker.py       # Distributed tuning worker (SLURM job arrays)
│   ├── visualize.py           # Plotting utilities
│   ├── piv_metrics.py         # Evaluation metrics
│   ├── config.py              # Device detection, DataLoader configuration
│   ├── paths.py               # Environment-aware path defaults
│   └── constants.py           # SEN geometry, quadrant mappings
├── scripts/
│   ├── slurm/                 # SLURM job scripts (train, tune, evaluate, arrays)
│   ├── nibi/                  # Cluster environment setup scripts
│   └── transfer/              # rsync data to cluster
├── data/raw/                  # Zarr archives (not committed)
├── models/                    # Saved checkpoints
├── optuna_studies/            # Tuning results (JSON + pickled studies)
└── pyproject.toml
```

---

## Environment Variables

| Variable | Purpose | Default |
|---|---|---|
| `PIV_DATA_PATH` | Path to Zarr archive | `data/raw/all_experiments.zarr/` (local) or `/scratch/$USER/data/raw/all_experiments.zarr/` (cluster) |
| `PIV_OUTPUT_DIR` | Output directory for models/results | `models/` |
| `PIV_LOG_DIR` | Log directory | `logs/` |
| `WANDB_API_KEY` | Weights & Biases authentication | — |

---

## HPC Cluster (Nibi)

The cluster uses SLURM and the Nibi/DRAC environment. Account: `def-bussmann`.

### First-time Setup

```bash
# On the cluster login node:
source scripts/nibi/setup_modules.sh   # Load StdEnv/2023, Python 3.11, CUDA
source scripts/nibi/setup_venv.sh      # Create venv and install dependencies
```

### Transfer Data

```bash
# From local machine:
NIBI_USER=youruser bash scripts/transfer/sync_data.sh
```

### Submit Jobs

```bash
# Single training run (24h, 1× H100, 16 CPUs, 64 GB RAM):
sbatch scripts/slurm/train.sh

# Hyperparameter tuning (48h, same resources):
sbatch scripts/slurm/tune.sh

# Evaluation (4h, CPU only):
sbatch scripts/slurm/evaluate.sh

# Distributed tuning (job array, multiple workers sharing a SQLite study):
bash scripts/slurm/submit_tuning.sh \
  --num-workers 4 \
  --trials-per-worker 25 \
  --study-name my_study \
  --storage sqlite:///optuna_study.db
```

Activate the environment interactively:
```bash
source scripts/nibi/activate_env.sh
```
