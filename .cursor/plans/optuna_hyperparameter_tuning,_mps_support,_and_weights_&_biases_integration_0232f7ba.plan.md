---
name: Optuna Hyperparameter Tuning, MPS Support, and Weights & Biases Integration
overview: Add Optuna hyperparameter tuning, MPS (Metal Performance Shaders) support for Apple Silicon (M4 Pro), memory-efficient configuration for 24GB RAM, and Weights & Biases experiment tracking.
todos: []
---

# Optuna Hyperparameter Tuning, MPS Support, and Weights & Biases Integration

## Overview

This plan adds three major enhancements:

1. **Optuna hyperparameter tuning** - Automated search for optimal hyperparameters
2. **MPS (Metal Performance Shaders) support** - Accelerate training on Apple Silicon (M4 Pro)
3. **Weights & Biases integration** - Experiment tracking and visualization
4. **Memory optimization** - Configurations optimized for 24GB RAM on M4 Pro MacBook

## Current Status

- ✅ Optuna is already in dependencies (`pyproject.toml`)
- ❌ No MPS device support (currently only CUDA/CPU)
- ❌ No Optuna integration
- ❌ No Weights & Biases integration
- ⚠️ Device selection doesn't detect Apple Silicon

## Implementation Tasks

### Task 1: MPS Device Support and Memory Optimization

**Files to Modify**: `src/train.py`, `src/evaluate.py`

**Changes**:

1. Add MPS device detection utility function
2. Update device selection logic to prioritize: MPS > CUDA > CPU
3. Add memory-efficient batch size recommendations for 24GB RAM
4. Configure DataLoader for optimal memory usage on Apple Silicon
5. Add memory monitoring utilities

**Key Functions to Add**:

```python
def get_device() -> torch.device:
    """Get best available device, prioritizing MPS for Apple Silicon."""
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def get_recommended_batch_size(device: torch.device, ram_gb: int = 24) -> int:
    """Get recommended batch size based on device and RAM."""
    # Conservative defaults for 24GB RAM on M4 Pro
    # Account for model size (~10-50MB), sequences, and overhead
    ...
```

**Memory Considerations for 24GB RAM**:

- Model size: ~10-50MB
- Batch size: Start with 8-16 (conservative), can go up to 32 if needed
- Sequence length: 20 frames (as designed)
- Use `pin_memory=False` for MPS (not supported)
- Consider gradient accumulation for effective larger batch sizes

**Configuration**:

- Default batch size: 16 (can adjust with flag)
- DataLoader `num_workers`: 0 (MPS doesn't benefit from multiprocessing)
- `pin_memory`: False for MPS
- Enable memory-efficient attention if available

### Task 2: Optuna Hyperparameter Tuning

**New File**: `src/tune.py` - Optuna-based hyperparameter tuning script

**Features**:

1. Define hyperparameter search space
2. Integrate with existing training pipeline
3. Support pruning (early stopping unpromising trials)
4. Save best hyperparameters and model
5. Visualize optimization progress

**Hyperparameters to Tune**:

- Learning rate: `loguniform(1e-5, 1e-3)`
- Weight decay: `loguniform(1e-6, 1e-4)`
- Batch size: `categorical([8, 16, 32])` (memory-constrained for 24GB)
- LSTM hidden dimension: `categorical([128, 256, 512])`
- LSTM num layers: `int(1, 3)`
- Dropout: `uniform(0.3, 0.7)`
- CNN feature dim: `categorical([64, 128, 256])`
- Sequence length: `categorical([10, 20, 30])` (optional, computationally expensive)

**Objective Function**:

- Optimize validation loss (MAE) or R²
- Support for multi-objective optimization (primary + secondary targets)

**Integration**:

- Reuse `train_epoch()` and `validate()` from `src/train.py`
- Support early stopping within Optuna trials
- Pruning support (e.g., `MedianPruner` or `SuccessiveHalvingPruner`)

**Command-line Interface**:

```bash
python -m src.tune \
    --n-trials 50 \
    --study-name cnn_lstm_hyperopt \
    --storage sqlite:///optuna_study.db \
    --pruning \
    --direction minimize \
    --objective validation_loss
```

**Outputs**:

- Best hyperparameters (JSON/YAML)
- Optuna study database
- Visualization plots (parallel coordinate, optimization history)

### Task 3: Weights & Biases Integration

**Files to Modify**: `src/train.py`, `src/tune.py` (optional)

**Features**:

1. Initialize wandb run at training start
2. Log hyperparameters, metrics, and system info
3. Track training/validation metrics per epoch
4. Log model artifacts (checkpoints)
5. Log visualizations (prediction plots, feature maps)
6. Optional: Compare multiple runs in wandb dashboard

**Integration Points**:

- **Start of training**: Initialize wandb, log config, system info
- **Per epoch**: Log train/val metrics, learning rate
- **Checkpoints**: Log model artifacts
- **End of training**: Log final metrics, best model
- **Visualizations**: Optional logging of plots to wandb

**Configuration**:

- Project name: `piv-bubble-prediction`
- Entity: User's wandb account (configurable)
- Tags: Include model type, dataset info
- Notes: Training configuration summary

**Command-line Arguments**:

```bash
python -m src.train \
    --use-wandb \
    --wandb-project piv-bubble-prediction \
    --wandb-entity <username> \
    --wandb-tags cnn-lstm baseline
```

**Logged Metrics**:

- Training: loss, all metrics (MAE, RMSE, MAPE, R²) per target
- Validation: loss, all metrics per target
- System: GPU/MPS utilization, memory usage (if available)
- Hyperparameters: All training arguments

**Optional Enhancements**:

- Log gradient norms for debugging
- Log sample predictions
- Log confusion matrices (if classification)
- Compare runs across hyperparameter sweeps

### Task 4: Update Dependencies

**File**: `pyproject.toml`

**Add**:

- `wandb>=0.16.0` (for experiment tracking)
- Verify `optuna>=4.6.0` is present (already there)

### Task 5: Memory-Efficient Configuration Utilities

**New File**: `src/config.py` or add to existing files

**Functions**:

- `get_device()` - MPS-aware device selection
- `get_recommended_batch_size()` - RAM-aware batch size recommendations
- `configure_dataloader()` - Memory-optimized DataLoader config
- `estimate_memory_usage()` - Estimate memory requirements for config

**M4 Pro Specific Optimizations**:

- Default batch size: 16 (conservative for 24GB RAM)
- Sequence length: 20 (balanced)
- Use gradient checkpointing if memory is tight
- Enable mixed precision (optional, MPS supports float16)

### Task 6: Update Training Script

**File**: `src/train.py`

**Changes**:

1. Replace device selection with `get_device()` utility
2. Add wandb initialization and logging
3. Update default batch size to be MPS/memory-aware
4. Add memory usage monitoring (optional)
5. Configure DataLoader for MPS (pin_memory=False, num_workers=0)

**Wandb Logging Points**:

- Initialize: `wandb.init()` at start
- Config: `wandb.config.update()` with all args
- Metrics: `wandb.log()` after each epoch
- Artifacts: `wandb.log_artifact()` for model checkpoints
- Finish: `wandb.finish()` at end

### Task 7: Update Evaluation Script

**File**: `src/evaluate.py`

**Changes**:

1. Add MPS device support
2. Optional wandb logging for evaluation runs
3. Memory-efficient evaluation batch size

## Implementation Details

### MPS Device Support

```python
def get_device(force_cpu: bool = False) -> torch.device:
    """Get best available device."""
    if force_cpu:
        return torch.device("cpu")
    
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")
```

### Optuna Study Structure

```python
def objective(trial: optuna.Trial) -> float:
    """Optuna objective function."""
    # Suggest hyperparameters
    lr = trial.suggest_loguniform("learning_rate", 1e-5, 1e-3)
    weight_decay = trial.suggest_loguniform("weight_decay", 1e-6, 1e-4)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    # ... more hyperparameters
    
    # Train model with suggested hyperparameters
    model = create_model(...)
    # ... training loop ...
    
    # Return validation metric to optimize
    return validation_loss
```

### Wandb Integration Pattern

```python
import wandb

# Initialize
wandb.init(
    project="piv-bubble-prediction",
    config={
        "learning_rate": args.learning_rate,
        "batch_size": args.batch_size,
        # ... all hyperparameters
    }
)

# Log metrics
wandb.log({
    "train/loss": train_loss,
    "val/loss": val_loss,
    "val/mae_primary": mae_primary,
    # ... all metrics
})

# Log artifacts
wandb.log_artifact(model_path, type="model")

# Finish
wandb.finish()
```

## File Structure

```
src/
├── train.py          # Updated with MPS and wandb
├── tune.py           # New: Optuna hyperparameter tuning
├── evaluate.py       # Updated with MPS support
├── config.py         # New: Device and memory utilities (optional)
└── [existing files]
```

## Configuration for 24GB RAM M4 Pro

**Recommended Defaults**:

- Batch size: 16 (can go up to 32 if needed)
- Sequence length: 20 (as designed)
- DataLoader workers: 0 (MPS doesn't benefit)
- Pin memory: False (MPS doesn't support)
- Gradient accumulation: Optional for effective larger batches

**Memory Estimation**:

- Model: ~20-50MB
- Per sample (sequence): ~20 frames × 22 × 30 × 2 channels × 4 bytes ≈ 105KB
- Batch of 16: ~1.7MB per batch
- Overhead: ~2-4GB for PyTorch, system, etc.
- Total per batch: ~2-3GB (leaves plenty of headroom)

## Success Criteria

1. ✅ MPS device automatically detected and used on Apple Silicon
2. ✅ Memory-efficient defaults work within 24GB RAM constraint
3. ✅ Optuna can run hyperparameter optimization
4. ✅ Wandb tracks all experiments with comprehensive metrics
5. ✅ Training script works seamlessly with MPS, Optuna, and wandb
6. ✅ All existing functionality preserved

## Testing Considerations

- Test MPS availability detection on Apple Silicon
- Verify memory usage stays within 24GB limit
- Test Optuna with small number of trials
- Verify wandb integration logs correctly
- Test fallback to CPU if MPS unavailable
- Verify mixed precision if implemented (optional)