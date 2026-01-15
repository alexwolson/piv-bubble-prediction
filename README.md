# PIV Bubble Prediction

Machine learning models to predict bubble counts from PIV (Particle Image Velocimetry) velocity field data in a continuous steel casting water model.

## Overview

This project builds deep learning models (CNN-LSTM) to predict bubble counts from raw PIV velocity field data. The model preserves the rich spatiotemporal structure of the PIV data, learning directly from velocity fields without manual feature engineering.

**Key Concept**: Map from raw PIV velocity fields → bubble counts using deep learning

## Quick Start

### Setup

```bash
# Install dependencies
uv sync

# Activate virtual environment
source .venv/bin/activate  # or `.venv\Scripts\activate` on Windows
```

### Data

The Zarr archive should be located at `data/raw/all_experiments.zarr/`. This archive contains:
- PIV velocity fields (u, v components) at 300 Hz
- Bubble counts (primary and secondary) aligned to sensor timestamps
- Alignment indices mapping PIV frames to sensor rows

## Project Structure

```
piv-bubble-prediction/
├── data/
│   ├── raw/                    # Zarr archive (all_experiments.zarr)
│   ├── intermediate/           # Processed data (features + targets)
│   └── processed/             # Model outputs, predictions
├── src/                        # Source code
│   ├── zarr_reader.py         # Zarr data loading
│   ├── piv_metrics.py         # PIV metric computation
│   ├── constants.py           # Shared constants
│   └── [training modules]     # To be created
├── models/                     # Trained model files
├── figures/                    # Visualization outputs
├── plan.md                     # Implementation plan
└── README.md                   # This file
```

## Target Variables

### Bubble Counts
- **bubble_count_primary**: From zero-shot model with EX1 exemplars
- **bubble_count_secondary**: From zero-shot model with EX2 exemplars
- **Source**: Stored in Zarr archive under `predictions/bubble_counts` (N × 2 array)
- **Alignment**: Already aligned to sensor timestamps (100 Hz)

## Model Architecture

### CNN-LSTM Hybrid
- **CNN Encoder**: Extracts spatial features from each PIV frame (vortices, flow structures)
- **LSTM**: Models temporal evolution of spatial features
- **Input**: Raw velocity fields (u, v components) as sequences
- **Output**: Predicts both bubble_count_primary and bubble_count_secondary

The model learns directly from the spatiotemporal structure of PIV data without requiring manual feature engineering.

## Workflow

1. **Load Data**: Extract PIV velocity fields and bubble counts from Zarr archive
2. **Create Sequences**: Build temporal sequences from PIV frames aligned to bubble counts
3. **Train Model**: Train CNN-LSTM model on sequences
4. **Evaluate**: Assess model performance and visualize learned features

## Dependencies

- Python 3.9+
- `uv` for package management
- Core: numpy, pandas, zarr
- Deep Learning: torch, torchvision
- Visualization: matplotlib, seaborn
- CLI: rich

## Usage

### Training

Train the CNN-LSTM model:

```bash
uv run python -m src.train \
    --zarr-path data/raw/all_experiments.zarr \
    --sequence-length 20 \
    --batch-size 16 \
    --epochs 100 \
    --learning-rate 1e-4
```

### Data Requirements

The Zarr archive should contain:
- PIV velocity fields (u, v) at `piv/u` and `piv/v`
- Bubble counts at `predictions/bubble_counts`
- Alignment indices at `aligned/sensor_row_index_per_frame`

## Related Projects

- **sen-piv-sensor-archive**: Creates the Zarr archive used by this project
- **Steel-XGBoost**: Predicts bubble counts from sensor data (different approach)
- **piv-metrics-prediction**: Predicts PIV metrics from sensor data (complementary)

## Documentation

- [plan.md](plan.md): Complete implementation plan and design considerations

## License

[To be determined]
