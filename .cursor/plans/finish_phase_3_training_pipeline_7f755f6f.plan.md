---
name: Finish Phase 3 Training Pipeline
overview: "Complete Phase 3: Training Pipeline by adding early stopping, comprehensive evaluation metrics (MAPE, R², RMSE), weight decay regularization, experiment-based train/test split, and optional TensorBoard logging."
todos: []
---

# Complete Phase 3: Training Pipeline

## Current Status

Phase 3 is ~90% complete. The core training infrastructure exists in [`src/train.py`](src/train.py):

- ✅ Training loop with epoch management
- ✅ Validation loop
- ✅ MAE loss and basic metrics (MAE primary/secondary)
- ✅ Learning rate scheduling (ReduceLROnPlateau)
- ✅ Model checkpointing (saves best model)
- ✅ Rich console logging

## Missing Components

### 1. Early Stopping

**Status**: Mentioned in plan but not implemented
**Location**: [`src/train.py`](src/train.py)
**Required**: Add early stopping based on validation loss with configurable patience

### 2. Comprehensive Evaluation Metrics

**Status**: Only MAE implemented, missing MAPE, R², RMSE
**Location**: [`src/train.py`](src/train.py) - `validate()` function
**Required**: Compute all metrics per target (primary/secondary) and overall

### 3. Weight Decay Regularization

**Status**: Dropout exists, weight decay not configured
**Location**: [`src/train.py`](src/train.py) - optimizer initialization
**Required**: Add weight decay parameter to Adam optimizer

### 4. Experiment-Based Train/Test Split

**Status**: Currently uses random split
**Location**: [`src/train.py`](src/train.py) - data splitting logic
**Required**: Implement experiment-based split (80% experiments → train, 20% → test) for better generalization testing

### 5. Data Augmentation (Optional Enhancement)

**Status**: Not implemented
**Location**: Create new module or add to dataset
**Required**: Implement temporal shifts, spatial transforms, noise injection

### 6. TensorBoard Logging (Optional Enhancement)

**Status**: Only Rich console logging exists
**Location**: [`src/train.py`](src/train.py)
**Required**: Add TensorBoard writer for experiment tracking

## Implementation Tasks

### Task 1: Add Early Stopping

- Create `EarlyStopping` class or use PyTorch's callback pattern
- Track best validation loss and patience counter
- Stop training when no improvement for `patience` epochs
- Save best model before early stop
- Add `--patience` argument (default: 15 epochs)

### Task 2: Enhance Evaluation Metrics

- Update `validate()` function in [`src/train.py`](src/train.py) to compute:
- **MAPE**: Mean Absolute Percentage Error per target
- **R²**: Coefficient of determination per target
- **RMSE**: Root Mean Squared Error per target
- Return metrics dictionary with all computed values
- Update logging to display all metrics

### Task 3: Add Weight Decay

- Add `--weight-decay` argument to argument parser (default: 1e-5)
- Configure Adam optimizer with weight decay parameter

### Task 4: Implement Experiment-Based Split

- Modify data loading/splitting logic in [`src/train.py`](src/train.py)
- Use experiment metadata to split by experiment (not random sequences)
- Add `--split-method` argument: `"random"` or `"experiment"` (default: `"experiment"`)
- Ensure sequences from same experiment stay together (train or test)

### Task 5: Data Augmentation (Optional)

- Create augmentation utilities or add to [`src/dataset.py`](src/dataset.py)
- Implement temporal shifts (±2 frames with zero-padding or wrap)
- Optional: Spatial flips (if physically meaningful for PIV data)
- Optional: Noise injection (small Gaussian noise)
- Add `--augment` flag to enable/disable augmentation

### Task 6: TensorBoard Logging (Optional)

- Add `tensorboard` to dependencies (or make it optional)
- Create TensorBoard writer in training loop
- Log: train/test loss, all metrics, learning rate
- Add `--tensorboard-dir` argument (default: `logs/tensorboard`)

## Files to Modify

1. **`src/train.py`**:

- Add early stopping logic
- Enhance `validate()` function with all metrics
- Add weight decay to optimizer
- Implement experiment-based split
- Add TensorBoard logging (optional)

2. **`src/dataset.py`** (if adding augmentation):

- Add data augmentation transforms
- Apply augmentations in `__getitem__()` when enabled

3. **`pyproject.toml`** (if adding TensorBoard):

- Add `tensorboard` dependency (optional)

## Implementation Order

1. **Early Stopping** (Critical) - Prevents overfitting and saves time
2. **Evaluation Metrics** (Critical) - Needed for proper model assessment
3. **Weight Decay** (High Priority) - Improves generalization
4. **Experiment-Based Split** (High Priority) - Better for generalization testing
5. **TensorBoard Logging** (Optional) - Nice to have for tracking
6. **Data Augmentation** (Optional) - Can be added later if needed

## Testing Considerations

- Verify early stopping stops training appropriately
- Ensure all metrics are computed correctly (compare with manual calculation)
- Test experiment-based split ensures no experiment leakage
- Validate augmentation doesn't break model training