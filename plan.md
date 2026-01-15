# Implementation Plan: PIV Data → Bubble Count Prediction (Deep Learning)

## Overview

Build deep learning models to predict bubble counts from PIV velocity field data using CNN-LSTM architecture. This approach preserves the rich spatiotemporal structure of the PIV data.

## Target Variables

### Bubble Counts
- **bubble_count_primary**: From zero-shot model with EX1 exemplars
- **bubble_count_secondary**: From zero-shot model with EX2 exemplars
- **Source**: Stored in Zarr archive under `predictions/bubble_counts` (N × 2 array)
- **Alignment**: Already aligned to sensor timestamps (100 Hz)

## Model Architecture: CNN-LSTM Hybrid

### Design
- **CNN Encoder**: Extract spatial features from each PIV frame
- **LSTM**: Model temporal evolution of spatial features
- **Output**: Predict both bubble_count_primary and bubble_count_secondary

### Input Format
- **Shape**: (batch, sequence_length, height, width, channels)
- **Channels**: u, v velocity components (optionally add |V|, vorticity)
- **Sequence Length**: 10-20 frames per sample (to be determined)
- **Spatial Dimensions**: 22 × 30 (full resolution)

## Implementation Phases

### Phase 1: Data Pipeline

**Tasks:**
1. Load PIV data and bubble counts from Zarr
2. Use alignment indices to match PIV frames to bubble count rows
3. Create sequence datasets (sliding windows over time)
4. Normalize velocity fields
5. Create train/test splits

**Deliverables:**
- Data loading module (`src/data_loader.py`)
- Dataset class for PyTorch
- Data preprocessing utilities

### Phase 2: Model Architecture

**Tasks:**
1. Implement CNN encoder for spatial feature extraction
2. Implement LSTM for temporal modeling
3. Implement full CNN-LSTM model
4. Create model configuration system

**Deliverables:**
- Model architecture (`src/models/cnn_lstm.py`)
- Model configuration
- Architecture visualization

### Phase 3: Training Pipeline

**Tasks:**
1. Implement training loop
2. Add evaluation metrics (MAE, MAPE, R²)
3. Implement validation
4. Add checkpointing and logging
5. Add visualization tools

**Deliverables:**
- Training script (`src/train.py`)
- Evaluation utilities
- Logging and checkpointing

### Phase 4: Evaluation & Interpretation

**Tasks:**
1. Evaluate on test set
2. Visualize learned spatial features (CNN feature maps)
3. Analyze temporal patterns (LSTM hidden states)
4. Error analysis
5. Compare primary vs secondary predictions

**Deliverables:**
- Evaluation report
- Feature visualizations
- Error analysis

## Key Design Decisions

### Sequence Length
- **Options**: 10, 20, 30, 50 frames
- **Consideration**: Balance between temporal context and computational cost
- **Decision**: Start with 20 frames (~0.2s at 100 Hz alignment)

### Spatial Resolution
- **Options**: Full 22×30, or downsample to 11×15
- **Decision**: Start with full resolution, can downsample if needed

### Input Channels
- **Base**: u, v components
- **Optional**: Add velocity magnitude |V|, vorticity
- **Decision**: Start with u, v, add others if needed

### Model Output
- **Option 1**: Single model, multi-output (both bubble counts)
- **Option 2**: Separate models for primary and secondary
- **Decision**: Single model, multi-output (captures correlations)

### Loss Function
- **Options**: MSE, MAE, Huber loss
- **Decision**: Start with MAE (less sensitive to outliers)

## Data Pipeline Details

### Sequence Creation
- Sliding window over time-aligned PIV frames
- Window size: 20 frames
- Stride: 1 frame (overlapping sequences)
- Each sequence → one bubble count prediction

### Normalization
- Normalize u, v by experiment-level statistics
- Or use global statistics across all experiments
- Decision: Per-experiment normalization (handles different flow scales)

### Train/Test Split
- **Option 1**: Time-based (first 80% train, last 20% test)
- **Option 2**: Experiment-based (80% experiments train, 20% test)
- **Decision**: Experiment-based (tests generalization)

## Model Architecture Details

### CNN Encoder
```
Input: (batch, channels, height, width)  # Single frame
Conv2D(32, kernel=3) → ReLU → BatchNorm
Conv2D(64, kernel=3) → ReLU → BatchNorm → MaxPool
Conv2D(128, kernel=3) → ReLU → BatchNorm → MaxPool
Global Average Pooling → (batch, 128)
```

### LSTM
```
Input: (batch, sequence_length, 128)  # Sequence of CNN features
LSTM(256, num_layers=2, bidirectional=True)
Output: (batch, 512)  # Last hidden state
```

### Output Head
```
Dense(256) → ReLU → Dropout(0.5)
Dense(128) → ReLU → Dropout(0.5)
Dense(2)  # bubble_count_primary, bubble_count_secondary
```

## Training Details

### Hyperparameters
- **Learning Rate**: 1e-4 with ReduceLROnPlateau
- **Batch Size**: 16-32 (depending on GPU memory)
- **Epochs**: 100 with early stopping
- **Optimizer**: Adam
- **Loss**: MAE
- **Regularization**: Dropout, weight decay

### Data Augmentation
- Temporal shifts (±2 frames)
- Slight spatial flips (if physically meaningful)
- Noise injection (small random perturbations)

## Evaluation Metrics

### Per Target
- **MAE**: Mean Absolute Error
- **MAPE**: Mean Absolute Percentage Error
- **R²**: Coefficient of determination
- **RMSE**: Root Mean Squared Error

### Target Performance
- MAPE < 20% (bubble counting has inherent uncertainty)
- R² > 0.6 (moderate correlation expected)

## Implementation Checklist

### Phase 1: Data Pipeline
- [ ] Create data loading module
- [ ] Implement sequence creation
- [ ] Implement normalization
- [ ] Create PyTorch Dataset class
- [ ] Create DataLoader with proper batching
- [ ] Test data loading pipeline

### Phase 2: Model Architecture
- [ ] Implement CNN encoder
- [ ] Implement LSTM module
- [ ] Implement full CNN-LSTM model
- [ ] Add model configuration
- [ ] Test forward pass

### Phase 3: Training
- [ ] Implement training loop
- [ ] Add evaluation metrics
- [ ] Implement validation
- [ ] Add checkpointing
- [ ] Add logging (TensorBoard or similar)
- [ ] Add early stopping

### Phase 4: Evaluation
- [ ] Evaluate on test set
- [ ] Visualize CNN feature maps
- [ ] Analyze LSTM attention (if added)
- [ ] Error analysis
- [ ] Generate predictions for all experiments

## File Structure

```
piv-bubble-prediction/
├── src/
│   ├── data_loader.py          # Data loading and preprocessing
│   ├── dataset.py              # PyTorch Dataset class
│   ├── models/
│   │   ├── __init__.py
│   │   ├── cnn_lstm.py         # CNN-LSTM model
│   │   └── config.py            # Model configurations
│   ├── train.py                # Training script
│   ├── evaluate.py             # Evaluation script
│   ├── visualize.py             # Visualization utilities
│   ├── zarr_reader.py          # ✅ Zarr data loading
│   ├── piv_metrics.py          # ✅ PIV metric computation (for analysis)
│   └── constants.py            # ✅ Constants
├── data/
│   ├── raw/                    # Zarr archive
│   ├── intermediate/           # Processed sequences
│   └── processed/              # Model outputs
├── models/                     # Saved model checkpoints
├── figures/                    # Visualizations
└── logs/                       # Training logs
```

## Next Steps

1. **Immediate**: Implement data loading and sequence creation
2. **Short-term**: Implement CNN-LSTM model architecture
3. **Medium-term**: Set up training pipeline
4. **Long-term**: Evaluation and interpretation
