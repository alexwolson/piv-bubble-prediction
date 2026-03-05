"""
Optuna worker for distributed hyperparameter tuning.

This module runs as a worker in a SLURM job array, loading an Optuna study
from shared storage and running trials independently.
"""

import argparse
import logging
import os
import sys
from typing import Optional

import optuna
import torch

from rich.console import Console
from rich.logging import RichHandler

from src.config import get_device
from src.data_loader import load_all_experiments
from src.paths import get_zarr_path
from src.tune import OptunaPruningCallback, train_trial

console = Console(force_terminal=None)  # Auto-detect terminal, fallback for SLURM
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(console=console)],
    force=True,
)
logger = logging.getLogger(__name__)

# Ensure output is flushed immediately for SLURM log files
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None
sys.stderr.reconfigure(line_buffering=True) if hasattr(sys.stderr, 'reconfigure') else None


def create_objective_function(
    sequences: torch.Tensor,
    targets: torch.Tensor,
    device: torch.device,
    args: argparse.Namespace,
):
    """
    Create an objective function for Optuna optimization.
    
    Args:
        sequences: Training sequences
        targets: Training targets
        device: Device to train on
        args: Command-line arguments
        
    Returns:
        Objective function that takes a trial and returns a float
    """
    def objective(trial: optuna.Trial) -> float:
        """
        Objective function for a single trial.
        
        Args:
            trial: Optuna trial object
            
        Returns:
            Validation loss (objective value)
        """
        wandb_run = None
        try:
            # Initialize wandb for this trial
            if args.use_wandb:
                try:
                    import wandb
                    wandb_mode = args.wandb_mode or os.environ.get("WANDB_MODE_OVERRIDE", "online")
                    if os.environ.get("DISABLE_WANDB") == "true":
                        wandb_run = None
                    else:
                        wandb.init(
                            project=args.wandb_project,
                            entity=args.wandb_entity,
                            config={**vars(args), **trial.params},
                            name=f"{args.study_name}-trial-{trial.number}-worker-{args.worker_id}",
                            mode=wandb_mode,
                            reinit=True,  # Allow reinitialization for multiple trials
                        )
                        wandb_run = wandb.run
                        logger.info(f"Wandb run initialized for trial {trial.number} (worker {args.worker_id})")
                except ImportError:
                    logger.warning("wandb not available. Install with: pip install wandb")
                    wandb_run = None
            
            # Determine metric name based on args.objective
            # args.objective is "validation_loss" or "validation_r2"
            metric_map = {
                "validation_loss": "loss",
                "validation_r2": "r2_total"
            }
            target_metric_name = metric_map.get(args.objective, "loss")
            
            best_metrics = train_trial(
                trial, 
                sequences, 
                targets, 
                device, 
                args, 
                wandb_run=wandb_run,
                objective_metric=target_metric_name
            )
            
            val_loss = best_metrics.get("loss", float("inf"))
            objective_value = best_metrics.get(target_metric_name)
            
            if objective_value is None:
                logger.warning(f"Objective metric {target_metric_name} not found in results. Using loss.")
                objective_value = val_loss
            
            # Log final trial metrics to wandb
            if wandb_run:
                import wandb
                log_dict = {
                    "trial": trial.number,
                    "validation_loss": val_loss,
                    "objective_value": objective_value,
                    "worker_id": args.worker_id,
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
            
            return objective_value
        except optuna.TrialPruned:
            raise
        except Exception as e:
            logger.error(f"Trial {trial.number} failed: {e}", exc_info=True)
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
    
    return objective


def main():
    parser = argparse.ArgumentParser(
        description="Optuna worker for distributed hyperparameter tuning"
    )
    
    # Required arguments
    parser.add_argument(
        "--study-name",
        type=str,
        required=True,
        help="Optuna study name",
    )
    parser.add_argument(
        "--storage-url",
        type=str,
        required=True,
        help="Optuna storage URL (e.g., sqlite:///optuna.db)",
    )
    parser.add_argument(
        "--trials-per-worker",
        type=int,
        default=1,
        help="Number of trials to run in this worker",
    )
    parser.add_argument(
        "--worker-id",
        type=str,
        default=None,
        help="Unique worker ID (default: from SLURM_ARRAY_TASK_ID)",
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
        help="Number of frames per sequence",
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
    parser.add_argument(
        "--pruning",
        action="store_true",
        help="Enable Optuna pruning",
    )
    parser.add_argument(
        "--sqlite-timeout",
        type=int,
        default=60,
        help="SQLite connection timeout in seconds",
    )
    
    # Training arguments
    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
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
    parser.add_argument(
        "--wandb-mode",
        type=str,
        choices=["online", "offline"],
        default=None,
        help="Wandb mode override (online/offline)",
    )
    
    # Logging
    parser.add_argument(
        "--verbosity",
        type=int,
        default=20,
        help="Logging verbosity (10=DEBUG, 20=INFO, 30=WARNING, 40=ERROR)",
    )
    
    args = parser.parse_args()
    
    # Set logging level
    logging.getLogger().setLevel(args.verbosity)
    
    # Get worker ID from SLURM or argument
    if args.worker_id is None:
        slurm_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
        slurm_job_id = os.environ.get("SLURM_JOB_ID", "unknown")
        if slurm_task_id:
            args.worker_id = f"{slurm_job_id}_{slurm_task_id}"
        else:
            args.worker_id = f"worker_{os.getpid()}"
    
    logger.info("=" * 60)
    logger.info("Optuna Worker Starting")
    logger.info("=" * 60)
    logger.info(f"Worker ID: {args.worker_id}")
    logger.info(f"Study name: {args.study_name}")
    logger.info(f"Storage URL: {args.storage_url}")
    logger.info(f"Trials per worker: {args.trials_per_worker}")
    logger.info(f"Node: {os.environ.get('SLURM_NODELIST', 'unknown')}")
    logger.info("=" * 60)
    
    # Set default zarr path if not provided
    if args.zarr_path is None:
        args.zarr_path = get_zarr_path()
    
    # Get device
    if args.device:
        device = torch.device(args.device)
    else:
        device = get_device(force_cpu=args.force_cpu)
    
    logger.info(f"Using device: {device}")
    
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
    
    # Load or create study from storage
    try:
        # Try to load existing study first
        try:
            study = optuna.load_study(
                study_name=args.study_name,
                storage=args.storage_url,
            )
            logger.info(f"Loaded existing study: {args.study_name}")
        except KeyError:
            # Study doesn't exist, create it
            logger.info(f"Study not found, creating new study: {args.study_name}")
            study = optuna.create_study(
                study_name=args.study_name,
                storage=args.storage_url,
                direction=args.direction,
                pruner=optuna.pruners.MedianPruner() if args.pruning else None,
                load_if_exists=True,  # In case another worker created it simultaneously
            )
            logger.info(f"Created study: {args.study_name} (direction: {args.direction})")
    except Exception as e:
        logger.error(f"Failed to load/create study: {e}")
        logger.error("Make sure storage URL is correct and accessible")
        sys.exit(1)
    
    # Create objective function
    objective = create_objective_function(sequences, targets, device, args)
    
    # Run trials
    logger.info(f"Starting {args.trials_per_worker} trial(s)...")
    trials_completed = 0
    trials_failed = 0
    
    for i in range(args.trials_per_worker):
        try:
            logger.info(f"Starting trial {i+1}/{args.trials_per_worker}...")
            study.optimize(objective, n_trials=1, show_progress_bar=False)
            trials_completed += 1
            logger.info(f"Trial {i+1} completed successfully")
        except optuna.TrialPruned:
            logger.info(f"Trial {i+1} was pruned")
            trials_completed += 1
        except Exception as e:
            logger.error(f"Trial {i+1} failed with error: {e}", exc_info=True)
            trials_failed += 1
    
    logger.info("=" * 60)
    logger.info("Worker Summary")
    logger.info("=" * 60)
    logger.info(f"Trials completed: {trials_completed}")
    logger.info(f"Trials failed: {trials_failed}")
    logger.info(f"Best value so far: {study.best_value:.4f}" if study.trials else "No trials completed")
    logger.info("=" * 60)
    
    # Wandb runs are finished per trial in the objective function
    logger.info("Worker finished successfully")


if __name__ == "__main__":
    main()
