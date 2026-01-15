"""
Optuna hyperparameter tuning for CNN-LSTM model.

Automated hyperparameter optimization using Optuna framework.
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, Optional

import optuna
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

from src.config import get_device, configure_dataloader
from src.data_loader import load_all_experiments
from src.dataset import PIVBubbleDataset
from src.models.cnn_lstm import create_model
from src.paths import get_zarr_path, get_output_dir
from src.train import (
    EarlyStopping,
    compute_metrics,
    train_epoch,
    validate,
)

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


class OptunaPruningCallback:
    """Callback to integrate Optuna pruning with training loop."""
    
    def __init__(self, trial: optuna.Trial, epoch: int = 0):
        self.trial = trial
        self.epoch = epoch
    
    def __call__(self, metrics: Dict[str, float]) -> bool:
        """
        Report metrics to Optuna and check if trial should be pruned.
        
        Args:
            metrics: Dictionary with validation metrics
            
        Returns:
            True if trial should be stopped (pruned), False otherwise
        """
        # Report intermediate value for pruning
        self.trial.report(metrics["loss"], self.epoch)
        self.epoch += 1
        
        # Check if trial should be pruned
        if self.trial.should_prune():
            logger.info(f"Trial {self.trial.number} pruned at epoch {self.epoch}")
            raise optuna.TrialPruned()
        
        return False


def train_trial(
    trial: optuna.Trial,
    sequences: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
) -> float:
    """
    Train a single trial with suggested hyperparameters.
    
    Args:
        trial: Optuna trial object
        sequences: Training sequences
        targets: Training targets
        device: Device to train on
        args: Command-line arguments
        
    Returns:
        Validation loss (objective value)
    """
    # Suggest hyperparameters (using new Optuna API)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-4, log=True)
    batch_size = trial.suggest_categorical("batch_size", [8, 16, 32])
    
    lstm_hidden_dim = trial.suggest_categorical("lstm_hidden_dim", [128, 256, 512])
    lstm_num_layers = trial.suggest_int("lstm_num_layers", 1, 3)
    dropout = trial.suggest_float("dropout", 0.3, 0.7)
    cnn_feature_dim = trial.suggest_categorical("cnn_feature_dim", [64, 128, 256])
    
    logger.info(
        f"Trial {trial.number}: lr={learning_rate:.6f}, batch_size={batch_size}, "
        f"lstm_hidden={lstm_hidden_dim}, lstm_layers={lstm_num_layers}, "
        f"dropout={dropout:.3f}, cnn_feat={cnn_feature_dim}"
    )
    
    # Get sequence dimensions
    sequence_length, height, width, channels = sequences.shape[1:]
    
    # Create dataset
    dataset = PIVBubbleDataset(
        sequences,
        targets,
        device=str(device),
        augment=args.augment,
        temporal_shift_max=args.temporal_shift_max,
        noise_std=args.noise_std,
    )
    
    # Split into train and test
    n_total = len(dataset)
    n_test = int(n_total * args.test_split)
    n_train = n_total - n_test
    
    train_dataset, test_dataset = random_split(
        dataset, [n_train, n_test], generator=torch.Generator().manual_seed(42)
    )
    
    # Configure DataLoader
    dataloader_config = configure_dataloader(device)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        **dataloader_config,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        **dataloader_config,
    )
    
    # Create model with suggested hyperparameters
    model = create_model(
        sequence_length=sequence_length,
        height=height,
        width=width,
        input_channels=channels,
        cnn_feature_dim=cnn_feature_dim,
        lstm_hidden_dim=lstm_hidden_dim,
        lstm_num_layers=lstm_num_layers,
        lstm_bidirectional=True,
        dropout=dropout,
    ).to(device)
    
    # Loss and optimizer
    criterion = nn.L1Loss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=10
    )
    
    # Early stopping (internal to trial)
    early_stopping = EarlyStopping(
        patience=args.patience, min_delta=0.0, restore_best_weights=True
    )
    
    # Optuna pruning callback
    pruning_callback = OptunaPruningCallback(trial, epoch=0)
    
    # Training loop with progress bar (by iteration)
    best_val_loss = float("inf")
    n_epochs = args.epochs
    n_batches_per_epoch = len(train_loader)
    total_iterations = n_epochs * n_batches_per_epoch
    current_iteration = 0
    
    with Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
        TimeRemainingColumn(),
        console=console,
    ) as progress:
        task = progress.add_task(
            f"[cyan]Trial {trial.number}",
            total=total_iterations,
        )
        
        for epoch in range(n_epochs):
            # Train with iteration tracking
            model.train()
            total_loss = 0.0
            n_samples = 0
            
            for batch_idx, (sequences, targets) in enumerate(train_loader):
                sequences = sequences.to(device)
                targets = targets.to(device)
                
                optimizer.zero_grad()
                
                # Forward pass
                predictions = model(sequences)
                loss = criterion(predictions, targets)
                
                # Backward pass
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item() * len(sequences)
                n_samples += len(sequences)
                
                # Update progress bar after each iteration
                current_iteration += 1
                current_loss = total_loss / n_samples if n_samples > 0 else 0.0
                progress.update(
                    task,
                    advance=1,
                    description=f"[cyan]Trial {trial.number} | "
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Batch {batch_idx+1}/{n_batches_per_epoch} | "
                    f"Loss: {current_loss:.4f}",
                )
            
            train_metrics = {"loss": total_loss / n_samples if n_samples > 0 else 0.0}
            
            # Validate at end of epoch
            test_metrics = validate(model, test_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(test_metrics["loss"])
            
            # Update progress bar with validation metrics
            progress.update(
                task,
                description=f"[cyan]Trial {trial.number} | "
                f"Epoch {epoch+1}/{n_epochs} | "
                f"Val Loss: {test_metrics['loss']:.4f} | "
                f"R²: {test_metrics['r2_total']:.4f}",
            )
            
            # Check pruning
            try:
                pruning_callback(test_metrics)
            except optuna.TrialPruned:
                # Trial was pruned, return current best
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                return best_val_loss
            
            # Track best validation loss
            if test_metrics["loss"] < best_val_loss:
                best_val_loss = test_metrics["loss"]
            
            # Early stopping
            if early_stopping(test_metrics["loss"], model, epoch):
                logger.info(f"Trial {trial.number} early stopped at epoch {epoch+1}")
                break
    
    # Restore best weights
    if early_stopping.restore_best_weights and early_stopping.best_weights:
        early_stopping.restore_best_model(model)
        
        # Re-validate with best weights
        final_metrics = validate(model, test_loader, criterion, device)
        best_val_loss = final_metrics["loss"]
    
    return best_val_loss


def main():
    parser = argparse.ArgumentParser(
        description="Hyperparameter tuning with Optuna for CNN-LSTM model"
    )
    
    # Data arguments
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
        help="Number of frames per sequence (fixed for now)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Stride for sliding window",
    )
    parser.add_argument(
        "--test-split",
        type=float,
        default=0.2,
        help="Fraction of data for testing",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of experiments (for testing)",
    )
    
    # Optuna arguments
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of Optuna trials",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default="cnn_lstm_hyperopt",
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage",
        type=str,
        default=None,
        help="Optuna storage URL (e.g., sqlite:///optuna_study.db)",
    )
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable Optuna pruning",
    )
    parser.add_argument(
        "--direction",
        type=str,
        choices=["minimize", "maximize"],
        default="minimize",
        help="Direction of optimization",
    )
    parser.add_argument(
        "--objective",
        type=str,
        choices=["validation_loss", "validation_r2"],
        default="validation_loss",
        help="Objective metric to optimize",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=10,
        help="Maximum number of epochs per trial",
    )
    parser.add_argument(
        "--patience",
        type=int,
        default=10,
        help="Early stopping patience",
    )
    
    # Device and augmentation
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (mps/cuda/cpu). Auto-detects if not specified.",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU usage",
    )
    parser.add_argument(
        "--augment",
        action="store_true",
        help="Enable data augmentation",
    )
    parser.add_argument(
        "--temporal-shift-max",
        type=int,
        default=2,
        help="Maximum temporal shift for augmentation",
    )
    parser.add_argument(
        "--noise-std",
        type=float,
        default=0.01,
        help="Noise std for augmentation",
    )
    
    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="optuna_studies",
        help="Directory to save study results",
    )
    
    # Wandb integration (optional)
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
    
    args = parser.parse_args()
    
    # Set default zarr path if not provided (environment-aware)
    if args.zarr_path is None:
        args.zarr_path = get_zarr_path()
    
    # Set default output dir if not provided (environment-aware)
    if args.output_dir == "optuna_studies":
        args.output_dir = get_output_dir("optuna_studies")
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(force_cpu=args.force_cpu)
    
    logger.info(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize wandb if enabled
    wandb_run = None
    if args.use_wandb:
        try:
            import wandb
            wandb.init(
                project=args.wandb_project,
                entity=args.wandb_entity,
                config=vars(args),
                name=f"optuna-{args.study_name}",
            )
            wandb_run = wandb.run
            logger.info("Weights & Biases logging enabled")
        except ImportError:
            logger.warning("wandb not available. Install with: pip install wandb")
            args.use_wandb = False
    
    # Load data
    logger.info(f"Loading data from {args.zarr_path}...")
    sequences, targets, metadata, _ = load_all_experiments(
        args.zarr_path,
        sequence_length=args.sequence_length,
        stride=args.stride,
        normalize=True,
        limit=args.limit,
        return_per_experiment=True,
    )
    
    logger.info(f"Loaded {len(sequences)} sequences")
    
    # Create or load study
    if args.storage:
        study = optuna.create_study(
            study_name=args.study_name,
            storage=args.storage,
            direction=args.direction,
            load_if_exists=True,
            pruner=optuna.pruners.MedianPruner() if args.pruning else None,
        )
        logger.info(f"Loaded/created study: {args.study_name} (storage: {args.storage})")
    else:
        # In-memory study
        study = optuna.create_study(
            study_name=args.study_name,
            direction=args.direction,
            pruner=optuna.pruners.MedianPruner() if args.pruning else None,
        )
        logger.info(f"Created in-memory study: {args.study_name}")
    
    # Define objective function
    def objective(trial: optuna.Trial) -> float:
        try:
            val_loss = train_trial(trial, sequences, targets, device, args)
            
            # Log to wandb if enabled
            if args.use_wandb and wandb_run:
                import wandb
                wandb.log({
                    "trial": trial.number,
                    "validation_loss": val_loss,
                    **trial.params,
                })
            
            return val_loss
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return a poor value instead of failing
            return float("inf") if args.direction == "minimize" else float("-inf")
    
    # Run optimization
    logger.info(f"Starting optimization with {args.n_trials} trials...")
    study.optimize(objective, n_trials=args.n_trials, show_progress_bar=True)
    
    # Save results
    best_params_path = output_dir / f"{args.study_name}_best_params.json"
    with open(best_params_path, "w") as f:
        json.dump(study.best_params, f, indent=2)
    logger.info(f"Saved best hyperparameters to {best_params_path}")
    
    # Print results
    logger.info("\n" + "=" * 60)
    logger.info("Optimization Results")
    logger.info("=" * 60)
    logger.info(f"Best trial: {study.best_trial.number}")
    logger.info(f"Best value: {study.best_value:.4f}")
    logger.info("\nBest hyperparameters:")
    for key, value in study.best_params.items():
        logger.info(f"  {key}: {value}")
    logger.info("=" * 60)
    
    # Save study
    if args.storage:
        logger.info(f"Study saved to: {args.storage}")
    else:
        # Save in-memory study to file
        import pickle
        study_path = output_dir / f"{args.study_name}_study.pkl"
        with open(study_path, "wb") as f:
            pickle.dump(study, f)
        logger.info(f"Study saved to: {study_path}")
    
    # Finish wandb
    if args.use_wandb and wandb_run:
        import wandb
        wandb.finish()
    
    logger.info("Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
