"""
Evaluate trained CNN-LSTM model and generate predictions on PIV data.

Usage:
    python -m src.evaluate --zarr-path data/raw/all_experiments.zarr/
    python -m src.evaluate --zarr-path data/raw/new_data.zarr/ --model-path models/best_model.pt
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from rich.console import Console
from rich.logging import RichHandler

from src.config import get_device, configure_dataloader
from src.data_loader import load_all_experiments
from src.dataset import PIVBubbleDataset
from src.models.cnn_lstm import create_model
from src.train import compute_metrics

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def load_checkpoint(checkpoint_path: str, device: torch.device):
    """Load model from checkpoint, reconstructing architecture from saved args."""
    path = Path(checkpoint_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    logger.info(f"Loading checkpoint from {path}")
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    args = checkpoint.get("args", {})

    model = create_model(
        sequence_length=args.get("sequence_length", 20),
        height=args.get("height", 22),
        width=args.get("width", 30),
        input_channels=args.get("input_channels", 2),
        cnn_feature_dim=args.get("cnn_feature_dim", 128),
        lstm_hidden_dim=args.get("lstm_hidden_dim", 256),
        lstm_num_layers=args.get("lstm_num_layers", 2),
        dropout=args.get("dropout", 0.5),
        lstm_bidirectional=args.get("bidirectional", True),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    epoch = checkpoint.get("epoch", "?")
    test_loss = checkpoint.get("test_loss")
    info = f"epoch {epoch}"
    if test_loss is not None:
        info += f", checkpoint test loss {test_loss:.4f}"
    logger.info(f"Loaded model ({info})")

    return model, args


def run_predictions(model, dataloader, device):
    """Run inference and return (predictions, targets) as numpy arrays."""
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            preds = model(sequences)
            all_preds.append(preds.cpu().numpy())
            all_targets.append(targets.numpy())

    return np.concatenate(all_preds), np.concatenate(all_targets)


def main():
    parser = argparse.ArgumentParser(description="Run model predictions on PIV data")
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="data/raw/all_experiments.zarr",
        help="Path to Zarr archive",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="models/best_model.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save predictions CSV and metrics JSON",
    )
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of experiments (for quick testing)",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Skip saving predictions to disk",
    )
    args = parser.parse_args()

    device = get_device()
    logger.info(f"Device: {device}")

    # Load model
    model, checkpoint_args = load_checkpoint(args.model_path, device)
    sequence_length = checkpoint_args.get("sequence_length", 20)
    stride = checkpoint_args.get("stride", 1)

    # Load data
    logger.info(f"Loading data from {args.zarr_path}...")
    experiments, _, _, _ = load_all_experiments(
        args.zarr_path,
        sequence_length=sequence_length,
        stride=stride,
        normalize=True,
        limit=args.limit,
        lazy=True,
    )
    logger.info(f"Loaded {len(experiments)} experiments")

    dataset = PIVBubbleDataset(
        experiments=experiments,
        sequence_length=sequence_length,
        stride=stride,
    )
    logger.info(f"Dataset: {len(dataset)} sequences")

    dl_config = configure_dataloader(device)
    dataloader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, **dl_config
    )

    # Run predictions
    logger.info("Running predictions...")
    predictions, targets = run_predictions(model, dataloader, device)

    # Compute and print metrics
    metrics = compute_metrics(predictions, targets)
    logger.info("=" * 50)
    logger.info(
        f"Primary:   MAE={metrics['mae_primary']:.4f}  "
        f"RMSE={metrics['rmse_primary']:.4f}  "
        f"R²={metrics['r2_primary']:.4f}  "
        f"MAPE={metrics['mape_primary']:.1f}%"
    )
    logger.info(
        f"Secondary: MAE={metrics['mae_secondary']:.4f}  "
        f"RMSE={metrics['rmse_secondary']:.4f}  "
        f"R²={metrics['r2_secondary']:.4f}  "
        f"MAPE={metrics['mape_secondary']:.1f}%"
    )
    logger.info(
        f"Overall:   MAE={metrics['mae_total']:.4f}  "
        f"RMSE={metrics['rmse_total']:.4f}  "
        f"R²={metrics['r2_total']:.4f}"
    )
    logger.info("=" * 50)

    # Save outputs
    if not args.no_save:
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        df = pd.DataFrame(
            {
                "pred_primary": predictions[:, 0],
                "pred_secondary": predictions[:, 1],
                "target_primary": targets[:, 0],
                "target_secondary": targets[:, 1],
            }
        )
        csv_path = out / "predictions.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved predictions to {csv_path}")

        metrics_path = out / "metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(metrics, f, indent=2)
        logger.info(f"Saved metrics to {metrics_path}")


if __name__ == "__main__":
    main()
