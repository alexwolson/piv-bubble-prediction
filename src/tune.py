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

import numpy as np
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
    wandb_run=None,
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
    best_test_metrics = None
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
        
        # Track last completed epoch's metrics to show in progress bar
        last_epoch_metrics = None  # (train_metrics, test_metrics)
        
        for epoch in range(n_epochs):
            # Train with iteration tracking
            model.train()
            total_loss = 0.0
            n_samples = 0
            all_predictions = []
            all_targets = []
            
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
                
                batch_loss = loss.item()
                total_loss += batch_loss * len(sequences)
                n_samples += len(sequences)
                
                # Collect predictions and targets for full metrics computation
                all_predictions.append(predictions.detach().cpu().numpy())
                all_targets.append(targets.detach().cpu().numpy())
                
                # Log per-batch loss to command line and wandb
                logger.debug(
                    f"Trial {trial.number} | Epoch {epoch+1}/{n_epochs} | "
                    f"Batch {batch_idx+1}/{n_batches_per_epoch} | Loss: {batch_loss:.4f}"
                )
                
                if wandb_run is not None:
                    import wandb
                    step = epoch * n_batches_per_epoch + batch_idx
                    wandb.log({
                        "train/batch_loss": batch_loss,
                        "trial": trial.number,
                        "batch": batch_idx,
                        "epoch": epoch,
                    }, step=step)
                
                # Update progress bar after each iteration
                current_iteration += 1
                current_loss = total_loss / n_samples if n_samples > 0 else 0.0
                
                # Build description with current batch info and last epoch metrics
                desc_parts = [
                    f"[cyan]Trial {trial.number} | "
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Batch {batch_idx+1}/{n_batches_per_epoch} | "
                    f"Loss: {current_loss:.4f}"
                ]
                
                # Add last epoch metrics if available (after first epoch completes)
                if last_epoch_metrics is not None:
                    last_train, last_test = last_epoch_metrics
                    desc_parts.append(
                        f" | Last: Train R²={last_train.get('r2_total', 0):.3f} "
                        f"Val R²={last_test['r2_total']:.3f} "
                        f"Val MAPE={last_test['mape_total']:.1f}%"
                    )
                
                progress.update(
                    task,
                    advance=1,
                    description="".join(desc_parts),
                )
            
            # Compute full training metrics
            avg_loss = total_loss / n_samples if n_samples > 0 else 0.0
            train_metrics = {"loss": avg_loss}
            if all_predictions:
                all_predictions = np.concatenate(all_predictions, axis=0)
                all_targets = np.concatenate(all_targets, axis=0)
                full_metrics = compute_metrics(all_predictions, all_targets)
                train_metrics.update(full_metrics)
            
            # Validate at end of epoch
            test_metrics = validate(model, test_loader, criterion, device)
            
            # Update learning rate
            scheduler.step(test_metrics["loss"])
            
            # Log per-epoch metrics to wandb
            # Use step after last batch of epoch to maintain monotonic step ordering
            if wandb_run is not None:
                import wandb
                epoch_step = epoch * n_batches_per_epoch + n_batches_per_epoch
                log_dict = {
                    "trial": trial.number,
                    "epoch": epoch,
                    "train/loss": train_metrics["loss"],
                    "train/mae_primary": train_metrics.get("mae_primary", 0),
                    "train/mae_secondary": train_metrics.get("mae_secondary", 0),
                    "train/mae_total": train_metrics.get("mae_total", 0),
                    "train/rmse_primary": train_metrics.get("rmse_primary", 0),
                    "train/rmse_secondary": train_metrics.get("rmse_secondary", 0),
                    "train/rmse_total": train_metrics.get("rmse_total", 0),
                    "train/mape_primary": train_metrics.get("mape_primary", 0),
                    "train/mape_secondary": train_metrics.get("mape_secondary", 0),
                    "train/mape_total": train_metrics.get("mape_total", 0),
                    "train/r2_primary": train_metrics.get("r2_primary", 0),
                    "train/r2_secondary": train_metrics.get("r2_secondary", 0),
                    "train/r2_total": train_metrics.get("r2_total", 0),
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
                wandb.log(log_dict, step=epoch_step)
            
            # Store metrics for next epoch's progress bar
            last_epoch_metrics = (train_metrics, test_metrics)
            
            # Update progress bar with validation metrics
            progress.update(
                task,
                description=(
                    f"[cyan]Trial {trial.number} | "
                    f"Epoch {epoch+1}/{n_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train R²: {train_metrics.get('r2_total', 0):.4f} | "
                    f"Train MAPE: {train_metrics.get('mape_total', 0):.2f}% | "
                    f"Val Loss: {test_metrics['loss']:.4f} | "
                    f"Val R²: {test_metrics['r2_total']:.4f} | "
                    f"Val MAPE: {test_metrics['mape_total']:.2f}%"
                ),
            )
            
            # Check pruning
            try:
                pruning_callback(test_metrics)
            except optuna.TrialPruned:
                # Trial was pruned, store all performance metrics before returning
                metrics_to_store = best_test_metrics if best_test_metrics is not None else test_metrics
                if metrics_to_store:
                    # Store all metrics: MAE, RMSE, MAPE, R² for primary, secondary, and total
                    metric_types = ["mae", "rmse", "mape", "r2"]
                    metric_targets = ["primary", "secondary", "total"]
                    for metric_type in metric_types:
                        for target in metric_targets:
                            metric_name = f"{metric_type}_{target}"
                            value = metrics_to_store.get(metric_name, np.inf if metric_type == "mape" else 0.0)
                            # Convert np.inf to None for storage (optuna handles None better than inf)
                            if np.isfinite(value):
                                trial.set_user_attr(metric_name, float(value))
                            else:
                                trial.set_user_attr(metric_name, None)
                logger.info(f"Trial {trial.number} pruned at epoch {epoch+1}")
                return best_val_loss
            
            # Track best validation loss and metrics
            if test_metrics["loss"] < best_val_loss:
                best_val_loss = test_metrics["loss"]
                best_test_metrics = test_metrics.copy()
            
            # Early stopping
            if early_stopping(test_metrics["loss"], model, epoch):
                logger.info(f"Trial {trial.number} early stopped at epoch {epoch+1}")
                break
    
    # Restore best weights
    final_metrics = None
    if early_stopping.restore_best_weights and early_stopping.best_weights:
        early_stopping.restore_best_model(model)
        
        # Re-validate with best weights
        final_metrics = validate(model, test_loader, criterion, device)
        best_val_loss = final_metrics["loss"]
    
    # Store all performance metrics as optuna trial user attributes for monitoring
    # Use final_metrics if available (best weights), otherwise use best_test_metrics
    metrics_to_store = final_metrics if final_metrics is not None else best_test_metrics
    if metrics_to_store:
        # Store all metrics: MAE, RMSE, MAPE, R² for primary, secondary, and total
        metric_types = ["mae", "rmse", "mape", "r2"]
        metric_targets = ["primary", "secondary", "total"]
        for metric_type in metric_types:
            for target in metric_targets:
                metric_name = f"{metric_type}_{target}"
                value = metrics_to_store.get(metric_name, np.inf if metric_type == "mape" else 0.0)
                # Convert np.inf to None for storage (optuna handles None better than inf)
                if np.isfinite(value):
                    trial.set_user_attr(metric_name, float(value))
                else:
                    trial.set_user_attr(metric_name, None)
    
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
        default=True,
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
    
    # Wandb will be initialized per trial in the objective function
    # No global initialization needed
    
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
        wandb_run = None
        try:
            # Initialize wandb for this trial
            if args.use_wandb:
                try:
                    import wandb
                    wandb.init(
                        project=args.wandb_project,
                        entity=args.wandb_entity,
                        config={**vars(args), **trial.params},
                        name=f"{args.study_name}-trial-{trial.number}",
                        reinit=True,  # Allow reinitialization for multiple trials
                    )
                    wandb_run = wandb.run
                    logger.info(f"Wandb run initialized for trial {trial.number}")
                except ImportError:
                    logger.warning("wandb not available. Install with: pip install wandb")
                    wandb_run = None
            
            val_loss = train_trial(trial, sequences, targets, device, args, wandb_run=wandb_run)
            
            # Log final trial metrics to wandb
            if wandb_run:
                import wandb
                log_dict = {
                    "trial": trial.number,
                    "validation_loss": val_loss,
                    **trial.params,
                }
                # Add all performance metrics from trial user attributes
                metric_types = ["mae", "rmse", "mape", "r2"]
                metric_targets = ["primary", "secondary", "total"]
                for metric_type in metric_types:
                    for target in metric_targets:
                        metric_name = f"{metric_type}_{target}"
                        value = trial.user_attrs.get(metric_name)
                        if value is not None:
                            log_dict[f"trial/{metric_name}"] = value
                wandb.log(log_dict)
            
            return val_loss
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}")
            # Return a poor value instead of failing
            return float("inf") if args.direction == "minimize" else float("-inf")
        finally:
            # Always finish wandb run for this trial
            if wandb_run:
                try:
                    import wandb
                    wandb.finish()
                except Exception as e:
                    logger.warning(f"Error finishing wandb run for trial {trial.number}: {e}")
    
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
    
    # Wandb runs are finished per trial in the objective function
    logger.info("Hyperparameter tuning complete!")


if __name__ == "__main__":
    main()
