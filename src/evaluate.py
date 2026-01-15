"""
Standalone evaluation script for CNN-LSTM model.

Loads trained model checkpoint and evaluates on test set or full dataset.
Generates comprehensive metrics, predictions, and evaluation reports.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, random_split

from rich.console import Console
from rich.logging import RichHandler

from src.config import get_device, configure_dataloader
from src.data_loader import load_all_experiments
from src.dataset import PIVBubbleDataset
from src.models.cnn_lstm import create_model
from src.train import compute_metrics
from src import visualize

console = Console()
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
)
logger = logging.getLogger(__name__)


def load_model_checkpoint(
    checkpoint_path: str,
    device: torch.device,
) -> Tuple[nn.Module, Dict]:
    """
    Load trained model from checkpoint file.
    
    Args:
        checkpoint_path: Path to model checkpoint (.pt file)
        device: Device to load model on
        
    Returns:
        model: Loaded model in eval mode
        checkpoint_info: Dictionary with checkpoint metadata
    """
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading model from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract model configuration from checkpoint args
    args = checkpoint.get("args", {})
    
    # Create model with saved configuration
    sequence_length = args.get("sequence_length", 20)
    height = args.get("height", 22)
    width = args.get("width", 30)
    input_channels = args.get("input_channels", 2)
    
    # For legacy checkpoints without args, try to infer from model state
    if not args:
        logger.warning("Checkpoint missing args, using defaults")
        model = create_model(
            sequence_length=sequence_length,
            height=height,
            width=width,
            input_channels=input_channels,
        )
    else:
        model = create_model(
            sequence_length=sequence_length,
            height=height,
            width=width,
            input_channels=input_channels,
        )
    
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    checkpoint_info = {
        "epoch": checkpoint.get("epoch", "unknown"),
        "test_loss": checkpoint.get("test_loss", "unknown"),
        "test_metrics": checkpoint.get("test_metrics", {}),
        "args": args,
    }
    
    logger.info(f"Loaded model from epoch {checkpoint_info['epoch']}")
    logger.info(f"Checkpoint test loss: {checkpoint_info['test_loss']}")
    
    return model, checkpoint_info


def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
    criterion: Optional[nn.Module] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """
    Evaluate model on dataset.
    
    Args:
        model: Trained model (in eval mode)
        dataloader: DataLoader for evaluation data
        device: Device to run evaluation on
        criterion: Optional loss function
        
    Returns:
        predictions: (n_samples, 2) - model predictions
        targets: (n_samples, 2) - ground truth targets
        metrics: Dictionary of evaluation metrics
    """
    model.eval()
    all_predictions = []
    all_targets = []
    total_loss = 0.0
    n_samples = 0
    
    if criterion is None:
        criterion = nn.L1Loss()
    
    logger.info("Running evaluation...")
    with torch.no_grad():
        for sequences, targets in dataloader:
            sequences = sequences.to(device)
            targets = targets.to(device)
            
            predictions = model(sequences)
            loss = criterion(predictions, targets)
            
            total_loss += loss.item() * len(sequences)
            n_samples += len(sequences)
            
            all_predictions.append(predictions.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    
    # Concatenate all predictions and targets
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    # Compute comprehensive metrics
    avg_loss = total_loss / n_samples
    metrics = compute_metrics(all_predictions, all_targets)
    metrics["loss"] = avg_loss
    
    logger.info(f"Evaluated {n_samples} samples")
    
    return all_predictions, all_targets, metrics


def save_predictions(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Path,
    metadata: Optional[List[Dict]] = None,
):
    """
    Save predictions to CSV file.
    
    Args:
        predictions: (n_samples, 2) - model predictions
        targets: (n_samples, 2) - ground truth targets
        output_path: Path to save CSV file
        metadata: Optional list of metadata dicts for each sample
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    data = {
        "pred_primary": predictions[:, 0],
        "pred_secondary": predictions[:, 1],
        "target_primary": targets[:, 0],
        "target_secondary": targets[:, 1],
        "error_primary": predictions[:, 0] - targets[:, 0],
        "error_secondary": predictions[:, 1] - targets[:, 1],
        "abs_error_primary": np.abs(predictions[:, 0] - targets[:, 0]),
        "abs_error_secondary": np.abs(predictions[:, 1] - targets[:, 1]),
    }
    
    df = pd.DataFrame(data)
    
    # Add metadata columns if provided
    if metadata:
        for key in metadata[0].keys():
            if key not in ["sequences", "targets"]:
                df[key] = [m.get(key, None) for m in metadata]
    
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions to {output_path}")


def generate_report(
    metrics: Dict[str, float],
    checkpoint_info: Dict,
    output_path: Path,
    split: str = "unknown",
):
    """
    Generate evaluation report in markdown format.
    
    Args:
        metrics: Dictionary of evaluation metrics
        checkpoint_info: Checkpoint metadata
        output_path: Path to save report
        split: Dataset split name (train/test/all)
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    report = f"""# Model Evaluation Report

## Checkpoint Information

- **Checkpoint Epoch**: {checkpoint_info.get('epoch', 'unknown')}
- **Checkpoint Test Loss**: {checkpoint_info.get('test_loss', 'unknown'):.4f}
- **Evaluation Split**: {split}
- **Model Architecture**: CNN-LSTM

## Evaluation Metrics

### Primary Bubble Count

- **MAE**: {metrics['mae_primary']:.4f}
- **RMSE**: {metrics['rmse_primary']:.4f}
- **MAPE**: {metrics['mape_primary']:.2f}%
- **R²**: {metrics['r2_primary']:.4f}

### Secondary Bubble Count

- **MAE**: {metrics['mae_secondary']:.4f}
- **RMSE**: {metrics['rmse_secondary']:.4f}
- **MAPE**: {metrics['mape_secondary']:.2f}%
- **R²**: {metrics['r2_secondary']:.4f}

### Overall Metrics

- **Average MAE**: {metrics['mae_total']:.4f}
- **Average RMSE**: {metrics['rmse_total']:.4f}
- **Average MAPE**: {metrics['mape_total']:.2f}%
- **Average R²**: {metrics['r2_total']:.4f}
- **Loss (MAE)**: {metrics['loss']:.4f}

## Performance Summary

"""
    
    # Add performance assessment
    if metrics['r2_total'] > 0.6:
        report += "✅ Model shows **good predictive performance** (R² > 0.6)\n\n"
    elif metrics['r2_total'] > 0.4:
        report += "⚠️ Model shows **moderate predictive performance** (0.4 < R² < 0.6)\n\n"
    else:
        report += "❌ Model shows **poor predictive performance** (R² < 0.4)\n\n"
    
    if metrics['mape_total'] < 20:
        report += "✅ MAPE is below 20% threshold\n\n"
    else:
        report += f"⚠️ MAPE is {metrics['mape_total']:.2f}% (above 20% threshold)\n\n"
    
    # Save report
    with open(output_path, "w") as f:
        f.write(report)
    
    logger.info(f"Saved evaluation report to {output_path}")


def create_data_split(
    sequences: np.ndarray,
    targets: np.ndarray,
    split: str,
    test_split: float = 0.2,
    random_seed: int = 42,
    experiment_ids: Optional[np.ndarray] = None,
) -> Tuple[Dataset, Dataset]:
    """
    Create train/test dataset split.
    
    Args:
        sequences: All sequences
        targets: All targets
        split: Which split to return ('train', 'test', or 'all')
        test_split: Fraction for test set (if split='all', ignored)
        random_seed: Random seed for reproducibility
        experiment_ids: Optional experiment IDs for experiment-based split
        
    Returns:
        dataset: Dataset for the requested split
    """
    if split == "all":
        return PIVBubbleDataset(sequences, targets, augment=False), None
    
    if split == "test":
        # Need to create same split as training
        dataset = PIVBubbleDataset(sequences, targets, augment=False)
        n_total = len(dataset)
        n_test = int(n_total * test_split)
        n_train = n_total - n_test
        
        train_dataset, test_dataset = random_split(
            dataset, [n_train, n_test], generator=torch.Generator().manual_seed(random_seed)
        )
        return test_dataset, None
    
    elif split == "train":
        # Need to create same split as training
        dataset = PIVBubbleDataset(sequences, targets, augment=False)
        n_total = len(dataset)
        n_test = int(n_total * test_split)
        n_train = n_total - n_test
        
        train_dataset, test_dataset = random_split(
            dataset, [n_train, n_test], generator=torch.Generator().manual_seed(random_seed)
        )
        return train_dataset, None
    
    else:
        raise ValueError(f"Unknown split: {split}. Must be 'train', 'test', or 'all'")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained CNN-LSTM model on dataset"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--zarr-path",
        type=str,
        default="data/raw/all_experiments.zarr",
        help="Path to Zarr archive",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation outputs",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["train", "test", "all"],
        default="test",
        help="Dataset split to evaluate on",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (mps/cuda/cpu). If not specified, auto-detects best device.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage even if MPS/CUDA available",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for test set (for train/test splits)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=None,
        help="Sequence length (if not in checkpoint, optional)",
    )
    parser.add_argument(
        "--generate-plots",
        action="store_true",
        help="Generate visualization plots (prediction vs actual, residuals, metrics comparison)",
    )
    
    args = parser.parse_args()
    
    # Get device (MPS-aware)
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(force_cpu=args.force_cpu)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load model checkpoint
    model, checkpoint_info = load_model_checkpoint(args.model_path, device)
    
    # Load data
    logger.info(f"Loading data from {args.zarr_path}...")
    
    # Get sequence length from checkpoint args or use provided/default
    if args.sequence_length:
        sequence_length = args.sequence_length
    elif checkpoint_info.get("args", {}).get("sequence_length"):
        sequence_length = checkpoint_info["args"]["sequence_length"]
    else:
        sequence_length = 20
        logger.warning(f"Using default sequence_length={sequence_length}")
    
    # Load all data
    # Always load with experiment IDs for consistency
    result = load_all_experiments(
        args.zarr_path,
        sequence_length=sequence_length,
        stride=1,
        normalize=True,
        return_per_experiment=True,
    )
    
    if args.split == "all":
        sequences, targets, metadata, experiment_ids = result
    else:
        sequences, targets, metadata, experiment_ids = result
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Create dataset split
    eval_dataset, _ = create_data_split(
        sequences,
        targets,
        split=args.split,
        test_split=args.test_split,
        experiment_ids=experiment_ids,
    )
    
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    
    # Configure DataLoader for device (MPS-aware)
    dataloader_config = configure_dataloader(device)
    
    # Create data loader
    eval_loader = DataLoader(
        eval_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_config,
    )
    
    # Evaluate model
    predictions, targets, metrics = evaluate_model(model, eval_loader, device)
    
    # Print metrics
    logger.info("\n" + "=" * 60)
    logger.info("Evaluation Results")
    logger.info("=" * 60)
    logger.info(f"Dataset Split: {args.split}")
    logger.info(f"Samples Evaluated: {len(predictions)}")
    logger.info("\nPrimary Bubble Count:")
    logger.info(f"  MAE:  {metrics['mae_primary']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse_primary']:.4f}")
    logger.info(f"  MAPE: {metrics['mape_primary']:.2f}%")
    logger.info(f"  R²:   {metrics['r2_primary']:.4f}")
    logger.info("\nSecondary Bubble Count:")
    logger.info(f"  MAE:  {metrics['mae_secondary']:.4f}")
    logger.info(f"  RMSE: {metrics['rmse_secondary']:.4f}")
    logger.info(f"  MAPE: {metrics['mape_secondary']:.2f}%")
    logger.info(f"  R²:   {metrics['r2_secondary']:.4f}")
    logger.info("\nOverall:")
    logger.info(f"  Average MAE:  {metrics['mae_total']:.4f}")
    logger.info(f"  Average RMSE: {metrics['rmse_total']:.4f}")
    logger.info(f"  Average MAPE: {metrics['mape_total']:.2f}%")
    logger.info(f"  Average R²:   {metrics['r2_total']:.4f}")
    logger.info(f"  Loss (MAE):   {metrics['loss']:.4f}")
    logger.info("=" * 60)
    
    # Save predictions
    predictions_path = output_dir / "predictions.csv"
    save_predictions(predictions, targets, predictions_path)
    
    # Generate report
    report_path = output_dir / "report.md"
    generate_report(metrics, checkpoint_info, report_path, split=args.split)
    
    # Save metrics as JSON for programmatic access
    import json
    metrics_path = output_dir / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.info(f"Saved metrics to {metrics_path}")
    
    # Generate visualization plots if requested
    if args.generate_plots:
        logger.info("Generating visualization plots...")
        figures_dir = output_dir / "figures"
        figures_dir.mkdir(parents=True, exist_ok=True)
        
        # Prediction vs actual
        visualize.plot_prediction_vs_actual(
            predictions,
            targets,
            output_path=figures_dir / "prediction_vs_actual.png",
        )
        
        # Residual analysis
        visualize.plot_residuals(
            predictions,
            targets,
            output_path=figures_dir / "residuals.png",
        )
        
        # Metrics comparison
        visualize.plot_metrics_comparison(
            metrics,
            output_path=figures_dir / "metrics_comparison.png",
        )
        
        # Comprehensive error analysis
        visualize.analyze_errors(
            predictions,
            targets,
            output_dir=figures_dir,
            prefix="error",
        )
        
        logger.info(f"Saved visualization plots to {figures_dir}")
    
    logger.info(f"\nEvaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    main()
