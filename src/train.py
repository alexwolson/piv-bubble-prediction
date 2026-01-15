"""
Training script for CNN-LSTM model to predict bubble counts from PIV data.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, random_split
from rich.console import Console
from rich.logging import RichHandler
from rich.progress import (
    Progress,
    BarColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from src.config import get_device, get_recommended_batch_size, configure_dataloader
from src.data_loader import load_all_experiments
from src.dataset import PIVBubbleDataset
from src.models.cnn_lstm import create_model
from src.paths import get_zarr_path, get_output_dir

console = Console(force_terminal=None)  # Auto-detect terminal, fallback for SLURM
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
    force=True,  # Ensure configuration is applied
)
logger = logging.getLogger(__name__)

# Ensure output is flushed immediately for SLURM log files
# This prevents buffering issues when logs are redirected
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None


class EarlyStopping:
    """Early stopping to stop training when validation loss doesn't improve."""
    
    def __init__(self, patience: int = 15, min_delta: float = 0.0, restore_best_weights: bool = True):
        """
        Args:
            patience: Number of epochs to wait before stopping
            min_delta: Minimum change to qualify as an improvement
            restore_best_weights: Whether to restore best weights when stopping
        """
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = float("inf")
        self.counter = 0
        self.best_weights = None
        self.best_epoch = 0
        
    def __call__(self, val_loss: float, model: nn.Module, epoch: int) -> bool:
        """
        Check if training should stop.
        
        Args:
            val_loss: Current validation loss
            model: Model to save weights from
            epoch: Current epoch number
            
        Returns:
            True if training should stop, False otherwise
        """
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.best_epoch = epoch
            if self.restore_best_weights:
                self.best_weights = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return False
        else:
            self.counter += 1
            return self.counter >= self.patience
    
    def restore_best_model(self, model: nn.Module):
        """Restore model to best weights."""
        if self.best_weights is not None:
            model.load_state_dict(self.best_weights)
            logger.info(f"Restored model weights from epoch {self.best_epoch} (best loss: {self.best_loss:.4f})")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
) -> Dict[str, float]:
    """Train for one epoch."""
    model.train()
    total_loss = 0.0
    n_samples = 0
    
    for sequences, targets in dataloader:
        sequences = sequences.to(device)  # (batch, seq_len, channels, height, width)
        targets = targets.to(device)  # (batch, 2)
        
        # Forward pass
        optimizer.zero_grad()
        predictions = model(sequences)  # (batch, 2)
        
        # Compute loss
        loss = criterion(predictions, targets)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(sequences)
        n_samples += len(sequences)
    
    avg_loss = total_loss / n_samples
    return {"loss": avg_loss}


def compute_metrics(predictions: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
    """
    Compute comprehensive evaluation metrics.
    
    Args:
        predictions: (n_samples, 2) - [primary, secondary]
        targets: (n_samples, 2) - [primary, secondary]
        
    Returns:
        Dictionary of metrics per target and overall
    """
    metrics = {}
    
    for target_idx, target_name in enumerate(["primary", "secondary"]):
        pred = predictions[:, target_idx]
        targ = targets[:, target_idx]
        
        # MAE (Mean Absolute Error)
        mae = np.mean(np.abs(pred - targ))
        metrics[f"mae_{target_name}"] = mae
        
        # RMSE (Root Mean Squared Error)
        rmse = np.sqrt(np.mean((pred - targ) ** 2))
        metrics[f"rmse_{target_name}"] = rmse
        
        # MAPE (Mean Absolute Percentage Error)
        # Avoid division by zero
        nonzero_mask = np.abs(targ) > 1e-8
        if np.any(nonzero_mask):
            mape = np.mean(np.abs((pred[nonzero_mask] - targ[nonzero_mask]) / targ[nonzero_mask])) * 100
        else:
            mape = np.inf
        metrics[f"mape_{target_name}"] = mape
        
        # R² (Coefficient of determination)
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        if ss_tot > 1e-8:
            r2 = 1 - (ss_res / ss_tot)
        else:
            r2 = 0.0
        metrics[f"r2_{target_name}"] = r2
    
    # Overall metrics (averaged across both targets)
    metrics["mae_total"] = (metrics["mae_primary"] + metrics["mae_secondary"]) / 2
    metrics["rmse_total"] = (metrics["rmse_primary"] + metrics["rmse_secondary"]) / 2
    metrics["mape_total"] = (metrics["mape_primary"] + metrics["mape_secondary"]) / 2
    metrics["r2_total"] = (metrics["r2_primary"] + metrics["r2_secondary"]) / 2
    
    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Validate model and compute comprehensive metrics."""
    model.eval()
    total_loss = 0.0
    n_samples = 0
    
    all_predictions = []
    all_targets = []
    
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
    
    avg_loss = total_loss / n_samples
    
    # Compute comprehensive metrics
    all_predictions = np.concatenate(all_predictions, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    metrics = compute_metrics(all_predictions, all_targets)
    metrics["loss"] = avg_loss
    
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Train CNN-LSTM model for bubble count prediction")
    parser.add_argument(
        "--zarr-path",
        type=str,
        default=None,
        help="Path to Zarr archive (default: auto-detect based on environment)",
    )
    parser.add_argument(
        "--sequence-length",
        type=int,
        default=20,
        help="Number of frames per sequence",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=100,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-5,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="models",
        help="Directory to save model checkpoints",
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
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--split-method",
        type=str,
        choices=["random", "experiment"],
        default="experiment",
        help="Train/test split method: 'random' or 'experiment'",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of experiments (for testing)",
    )
    parser.add_argument(
        "--tensorboard-dir",
        type=str,
        default="logs/tensorboard",
        help="Directory for TensorBoard logs (optional, set to empty to disable)",
    )
    parser.add_argument(
        "--use-wandb",
        action="store_true",
        help="Enable Weights & Biases logging",
    )
    parser.add_argument(
        "--wandb-project",
        type=str,
        default="piv-bubble-prediction",
        help="Wandb project name",
    )
    parser.add_argument(
        "--wandb-entity",
        type=str,
        default=None,
        help="Wandb entity (username or team)",
    )
    parser.add_argument(
        "--wandb-tags",
        type=str,
        nargs="*",
        default=[],
        help="Wandb tags for this run",
    )
    parser.add_argument(
        "--wandb-name",
        type=str,
        default=None,
        help="Wandb run name (auto-generated if not specified)",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation (temporal shifts, noise)",
    )
    parser.add_argument(
        "--temporal-shift-max",
        type=int,
        default=2,
        help="Maximum temporal shift in frames for augmentation",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Standard deviation for Gaussian noise injection (0 to disable)",
    )
    
    args = parser.parse_args()
    
    # Set default zarr path if not provided (environment-aware)
    if args.zarr_path is None:
        args.zarr_path = get_zarr_path()
    
    # Set default output dir if not provided (environment-aware)
    if args.output_dir == "models":
        args.output_dir = get_output_dir("models")
    
    # Get device (MPS-aware, cluster-aware)
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(force_cpu=args.force_cpu)
    
    logger.info(f"Using device: {device}")
    logger.info(f"Zarr path: {args.zarr_path}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize Weights & Biases if enabled
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                name=args.wandb_name,
                tags=args.wandb_tags if args.wandb_tags else None,
                config=vars(args),
            )
            wandb_run = wandb.run
            logger.info("Weights & Biases logging enabled")
        except ImportError:
            logger.warning("wandb not available. Install with: pip install wandb")
            args.use_wandb = False
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    logger.info("Loading data from Zarr archive...")
    # Always load with experiment IDs for consistency
    sequences, targets, metadata, experiment_ids = load_all_experiments(
        args.zarr_path,
        sequence_length=args.sequence_length,
        stride=args.stride,
        normalize=True,
        limit=args.limit,
        return_per_experiment=(args.split_method == "experiment"),
    )
    
    logger.info(f"Loaded {len(sequences)} sequences")
    logger.info(f"Sequence shape: {sequences.shape}")
    logger.info(f"Target shape: {targets.shape}")
    
    # Split into train and test
    if args.split_method == "experiment":
        logger.info("Using experiment-based split...")
        # Split by experiments - returns numpy arrays
        rng = np.random.RandomState(42)
        unique_experiments = np.unique(experiment_ids)
        n_experiments = len(unique_experiments)
        n_test_experiments = max(1, int(n_experiments * args.test_split))
        
        # Shuffle experiment IDs
        shuffled_experiments = unique_experiments.copy()
        rng.shuffle(shuffled_experiments)
        
        test_experiments = set(shuffled_experiments[:n_test_experiments])
        
        # Create masks for train/test
        train_mask = np.array([exp_id not in test_experiments for exp_id in experiment_ids])
        test_mask = ~train_mask
        
        train_sequences = sequences[train_mask]
        train_targets = targets[train_mask]
        test_sequences = sequences[test_mask]
        test_targets = targets[test_mask]
        
        logger.info(
            f"Split by experiments: {n_experiments - n_test_experiments} train experiments, "
            f"{n_test_experiments} test experiments"
        )
        
        # Create datasets with augmentation for train if enabled
        train_dataset = PIVBubbleDataset(
            train_sequences,
            train_targets,
            device=str(device),
            augment=args.augment,
            temporal_shift_max=args.temporal_shift_max,
            noise_std=args.noise_std,
        )
        test_dataset = PIVBubbleDataset(
            test_sequences,
            test_targets,
            device=str(device),
            augment=False,  # No augmentation for test
        )
        
        n_train = len(train_dataset)
        n_test = len(test_dataset)
        logger.info(f"Train samples: {n_train}, Test samples: {n_test} (split by experiments)")
    else:
        # Random split
        dataset = PIVBubbleDataset(
            sequences,
            targets,
            device=str(device),
            augment=False,  # No augmentation for full dataset before split
        )
        n_total = len(dataset)
        n_test = int(n_total * args.test_split)
        n_train = n_total - n_test
        
        train_subset, test_subset = random_split(
            dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
        )
        
        # Apply augmentation to train dataset if enabled
        if args.augment:
            # Extract sequences and targets from split datasets
            train_indices = train_subset.indices
            train_seq = sequences[train_indices]
            train_targ = targets[train_indices]
            train_dataset = PIVBubbleDataset(
                train_seq,
                train_targ,
                device=str(device),
                augment=True,
                temporal_shift_max=args.temporal_shift_max,
                noise_std=args.noise_std,
            )
        else:
            train_dataset = train_subset
        
        test_dataset = test_subset
        
        if args.augment:
            logger.info(
                f"Data augmentation enabled: temporal_shift_max={args.temporal_shift_max}, "
                f"noise_std={args.noise_std}"
            )
        
        logger.info(f"Train samples: {n_train}, Test samples: {n_test} (random split)")
    
    # Configure DataLoader for device (MPS-aware)
    dataloader_config = configure_dataloader(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        **dataloader_config,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        **dataloader_config,
    )
    
    
    # Create model
    sequence_length, height, width, channels = sequences.shape[1:]
    logger.info(f"Creating model for input shape: ({sequence_length}, {height}, {width}, {channels})")
    
    model = create_model(
        sequence_length=sequence_length,
        height=height,
        width=width,
        input_channels=channels,
    ).to(device)
    
    logger.info(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Loss and optimizer
    criterion = nn.L1Loss()  # MAE loss
    optimizer = optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    
    # Early stopping
    early_stopping = EarlyStopping(patience=args.patience, min_delta=0.0, restore_best_weights=True)
    
    # TensorBoard logging (optional)
    tensorboard_writer = None
    if args.tensorboard_dir:
        try:
            from torch.utils.tensorboard import SummaryWriter
            tensorboard_dir = Path(args.tensorboard_dir)
            tensorboard_dir.mkdir(parents=True, exist_ok=True)
            tensorboard_writer = SummaryWriter(log_dir=str(tensorboard_dir))
            logger.info(f"TensorBoard logging enabled: {tensorboard_dir}")
        except ImportError:
            logger.warning("TensorBoard not available. Install with: pip install tensorboard")
    
    # Training loop
    best_test_loss = float("inf")
    
    logger.info("Starting training...")
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        for epoch in range(args.epochs):
            # Train
            train_metrics = train_epoch(model, train_loader, criterion, optimizer, device)
            
            # Validate
            test_metrics = validate(model, test_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(test_metrics["loss"])
            
            # Log progress
            logger.info(
                f"Epoch {epoch+1}/{args.epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} | "
                f"Test Loss: {test_metrics['loss']:.4f} | "
                f"Test MAE Primary: {test_metrics['mae_primary']:.4f} | "
                f"Test MAE Secondary: {test_metrics['mae_secondary']:.4f} | "
                f"Test R² Primary: {test_metrics['r2_primary']:.4f} | "
                f"Test R² Secondary: {test_metrics['r2_secondary']:.4f} | "
                f"Test MAPE Primary: {test_metrics['mape_primary']:.2f}% | "
                f"Test MAPE Secondary: {test_metrics['mape_secondary']:.2f}%"
            )
            
            # TensorBoard logging
            if tensorboard_writer is not None:
                tensorboard_writer.add_scalar("Loss/Train", train_metrics["loss"], epoch)
                tensorboard_writer.add_scalar("Loss/Test", test_metrics["loss"], epoch)
                tensorboard_writer.add_scalar("MAE/Test_Primary", test_metrics["mae_primary"], epoch)
                tensorboard_writer.add_scalar("MAE/Test_Secondary", test_metrics["mae_secondary"], epoch)
                tensorboard_writer.add_scalar("RMSE/Test_Primary", test_metrics["rmse_primary"], epoch)
                tensorboard_writer.add_scalar("RMSE/Test_Secondary", test_metrics["rmse_secondary"], epoch)
                tensorboard_writer.add_scalar("MAPE/Test_Primary", test_metrics["mape_primary"], epoch)
                tensorboard_writer.add_scalar("MAPE/Test_Secondary", test_metrics["mape_secondary"], epoch)
                tensorboard_writer.add_scalar("R2/Test_Primary", test_metrics["r2_primary"], epoch)
                tensorboard_writer.add_scalar("R2/Test_Secondary", test_metrics["r2_secondary"], epoch)
                tensorboard_writer.add_scalar("Learning_Rate", optimizer.param_groups[0]["lr"], epoch)
            
            # Wandb logging
            if args.use_wandb and wandb_run:
                import wandb
                log_dict = {
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "test/loss": test_metrics["loss"],
                    "test/mae_primary": test_metrics["mae_primary"],
                    "test/mae_secondary": test_metrics["mae_secondary"],
                    "test/mae_total": test_metrics["mae_total"],
                    "test/rmse_primary": test_metrics["rmse_primary"],
                    "test/rmse_secondary": test_metrics["rmse_secondary"],
                    "test/rmse_total": test_metrics["rmse_total"],
                    "test/mape_primary": test_metrics["mape_primary"],
                    "test/mape_secondary": test_metrics["mape_secondary"],
                    "test/mape_total": test_metrics["mape_total"],
                    "test/r2_primary": test_metrics["r2_primary"],
                    "test/r2_secondary": test_metrics["r2_secondary"],
                    "test/r2_total": test_metrics["r2_total"],
                    "learning_rate": optimizer.param_groups[0]["lr"],
                }
                wandb.log(log_dict)
            
            # Save best model
            if test_metrics["loss"] < best_test_loss:
                best_test_loss = test_metrics["loss"]
                model_path = output_dir / "best_model.pt"
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "test_loss": test_metrics["loss"],
                    "test_metrics": test_metrics,
                    "args": vars(args),  # Save training arguments
                }
                torch.save(checkpoint, model_path)
                logger.info(f"Saved best model to {model_path}")
                
                # Log model artifact to wandb
                if args.use_wandb and wandb_run:
                    import wandb
                    artifact = wandb.Artifact("best_model", type="model")
                    artifact.add_file(str(model_path))
                    wandb.log_artifact(artifact)
            
            # Early stopping check
            if early_stopping(test_metrics["loss"], model, epoch):
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs. "
                    f"No improvement for {args.patience} epochs."
                )
                if early_stopping.restore_best_weights:
                    early_stopping.restore_best_model(model)
                break
    
    # Close TensorBoard writer
    if tensorboard_writer is not None:
        tensorboard_writer.close()
    
    # Finish wandb run
    if args.use_wandb and wandb_run:
        import wandb
        wandb.finish()
        logger.info("Wandb run finished")
    
    logger.info("Training complete!")


if __name__ == "__main__":
    main()
