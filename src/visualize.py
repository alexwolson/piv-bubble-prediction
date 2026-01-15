"""
Visualization utilities for CNN-LSTM model interpretation.

Includes functions for:
- CNN feature map visualization
- LSTM hidden state analysis
- Error analysis and visualization
- Prediction comparison plots
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# Set style
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)


class FeatureExtractor:
    """Helper class to extract intermediate features from model."""
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.activations = {}
        self.hooks = []
        
    def _get_activation_hook(self, name: str):
        """Create a hook function to capture activations."""
        def hook(module, input, output):
            self.activations[name] = output.detach()
        return hook
    
    def register_hooks(self, layer_names: Optional[List[str]] = None):
        """Register forward hooks on specified layers."""
        # Clear previous hooks
        self.remove_hooks()
        
        if layer_names is None:
            # Default: capture all CNN conv layers
            layer_names = ["conv1", "conv2", "conv3"]
        
        # Register hooks on CNN encoder layers
        cnn_encoder = self.model.cnn_encoder
        for name in layer_names:
            if hasattr(cnn_encoder, name):
                layer = getattr(cnn_encoder, name)
                hook = layer.register_forward_hook(
                    self._get_activation_hook(f"cnn_{name}")
                )
                self.hooks.append(hook)
    
    def remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def clear_activations(self):
        """Clear stored activations."""
        self.activations = {}


def extract_cnn_features(
    model: nn.Module,
    input_tensor: torch.Tensor,
    layer_names: Optional[List[str]] = None,
) -> Dict[str, torch.Tensor]:
    """
    Extract CNN feature maps from specified layers.
    
    Args:
        model: Trained CNN-LSTM model (in eval mode)
        input_tensor: Input tensor (batch, channels, height, width) for single frame
                      or (batch, seq_len, channels, height, width) for sequence
        layer_names: List of layer names to extract from (e.g., ['conv1', 'conv2', 'conv3'])
                    If None, extracts from all conv layers
    
    Returns:
        Dictionary mapping layer names to feature maps
    """
    model.eval()
    extractor = FeatureExtractor(model)
    
    if layer_names is None:
        layer_names = ["conv1", "conv2", "conv3"]
    
    # Register hooks
    extractor.register_hooks(layer_names)
    
    # Handle input format
    if input_tensor.dim() == 5:
        # Sequence input: process first frame only for visualization
        single_frame = input_tensor[0, 0, :, :, :].unsqueeze(0).unsqueeze(0)  # (1, 1, channels, H, W)
        single_frame = single_frame.squeeze(1)  # (1, channels, H, W)
    elif input_tensor.dim() == 4:
        # Single frame input
        single_frame = input_tensor[0:1]  # Take first batch item
    else:
        raise ValueError(f"Unexpected input tensor shape: {input_tensor.shape}")
    
    # Forward pass to trigger hooks
    with torch.no_grad():
        _ = model.cnn_encoder(single_frame)
    
    # Extract activations
    features = extractor.activations.copy()
    
    # Clean up
    extractor.remove_hooks()
    
    return features


def visualize_feature_maps(
    features: Dict[str, torch.Tensor],
    input_frame: Optional[torch.Tensor] = None,
    layer_name: Optional[str] = None,
    n_features: int = 16,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (16, 12),
):
    """
    Visualize CNN feature maps.
    
    Args:
        features: Dictionary of feature maps from extract_cnn_features()
        input_frame: Optional original input frame for reference (channels, H, W)
        layer_name: Specific layer to visualize (if None, visualize all)
        n_features: Number of feature maps to display per layer
        output_path: Path to save figure (if None, display)
        figsize: Figure size
    """
    if layer_name:
        layers_to_plot = [layer_name] if f"cnn_{layer_name}" in features else []
    else:
        layers_to_plot = [k.replace("cnn_", "") for k in features.keys() if k.startswith("cnn_")]
    
    if not layers_to_plot:
        logger.warning("No features to visualize")
        return
    
    # Calculate grid size
    n_layers = len(layers_to_plot)
    if input_frame is not None:
        n_rows = n_layers + 1
    else:
        n_rows = n_layers
    
    fig, axes = plt.subplots(n_rows, 1, figsize=figsize)
    if n_rows == 1:
        axes = [axes]
    
    row_idx = 0
    
    # Plot input frame if provided
    if input_frame is not None:
        if isinstance(input_frame, torch.Tensor):
            input_frame = input_frame.cpu().numpy()
        
        # Input is (channels, H, W) - show velocity magnitude
        if input_frame.shape[0] == 2:  # u, v channels
            u = input_frame[0]
            v = input_frame[1]
            vel_mag = np.sqrt(u**2 + v**2)
        else:
            vel_mag = input_frame[0]
        
        axes[row_idx].imshow(vel_mag, cmap="viridis", aspect="auto")
        axes[row_idx].set_title("Input Frame (Velocity Magnitude)", fontsize=12)
        axes[row_idx].axis("off")
        row_idx += 1
    
    # Plot feature maps for each layer
    for layer_name in layers_to_plot:
        feat_key = f"cnn_{layer_name}"
        if feat_key not in features:
            continue
            
        feat = features[feat_key]
        if isinstance(feat, torch.Tensor):
            feat = feat.cpu().numpy()
        
        # Feature map shape: (batch, channels, H, W) or (batch, channels, 1, 1) after pooling
        if feat.ndim == 4:
            feat = feat[0]  # Take first batch item
        elif feat.ndim == 2:
            # After global pooling - skip visualization
            logger.warning(f"Layer {layer_name} is after global pooling, skipping visualization")
            continue
        
        # Limit number of feature maps
        n_channels = min(n_features, feat.shape[0])
        
        # Create subplot grid for this layer
        n_cols = min(8, n_channels)
        n_feat_rows = (n_channels + n_cols - 1) // n_cols
        
        # Create inset axes for feature maps
        ax_main = axes[row_idx]
        ax_main.set_title(f"Layer: {layer_name} ({n_channels} feature maps)", fontsize=12)
        ax_main.axis("off")
        
        # Calculate positions for subplots
        for i in range(n_channels):
            row = i // n_cols
            col = i % n_cols
            
            # Create inset axis
            inset = ax_main.inset_axes([
                col / n_cols, 1 - (row + 1) / n_feat_rows,
                1 / n_cols - 0.01, 1 / n_feat_rows - 0.01
            ])
            
            feat_map = feat[i]
            # Average over remaining dimensions if needed
            if feat_map.ndim > 2:
                feat_map = np.mean(feat_map, axis=tuple(range(feat_map.ndim - 2)))
            
            inset.imshow(feat_map, cmap="hot", aspect="auto")
            inset.axis("off")
        
        row_idx += 1
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved feature map visualization to {output_path}")
    else:
        plt.show()
    
    plt.close()


def extract_lstm_states(
    model: nn.Module,
    input_sequence: torch.Tensor,
    return_all_timesteps: bool = True,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extract LSTM hidden states for a sequence.
    
    Args:
        model: Trained CNN-LSTM model (in eval mode)
        input_sequence: Input sequence (batch, seq_len, channels, H, W)
        return_all_timesteps: If True, return states for all time steps; else only last
    
    Returns:
        lstm_outputs: (batch, seq_len, hidden_dim) - LSTM outputs at each time step
        hidden_states: (batch, hidden_dim) - Final hidden state (if return_all_timesteps=False)
    """
    model.eval()
    batch_size, seq_len = input_sequence.shape[0], input_sequence.shape[1]
    
    # Reshape for CNN encoder
    if input_sequence.dim() == 5 and input_sequence.shape[-1] == 2:
        input_sequence = input_sequence.permute(0, 1, 4, 2, 3)  # (batch, seq_len, channels, H, W)
    
    x = input_sequence.contiguous().view(batch_size * seq_len, *input_sequence.shape[2:])
    
    # Extract CNN features
    with torch.no_grad():
        cnn_features = model.cnn_encoder(x)  # (batch * seq_len, feature_dim)
        cnn_features = cnn_features.view(batch_size, seq_len, model.cnn_feature_dim)
        
        # Forward through LSTM
        lstm_out, (h_n, c_n) = model.lstm(cnn_features)
    
    if return_all_timesteps:
        return lstm_out.cpu(), h_n.cpu()
    else:
        return lstm_out[:, -1, :].cpu(), h_n.cpu()


def plot_hidden_state_evolution(
    lstm_outputs: torch.Tensor,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 8),
    n_components: int = 8,
):
    """
    Plot temporal evolution of LSTM hidden states.
    
    Args:
        lstm_outputs: (batch, seq_len, hidden_dim) - LSTM outputs
        output_path: Path to save figure
        figsize: Figure size
        n_components: Number of hidden state components to plot
    """
    if isinstance(lstm_outputs, torch.Tensor):
        lstm_outputs = lstm_outputs.numpy()
    
    # Average over batch dimension
    lstm_avg = np.mean(lstm_outputs, axis=0)  # (seq_len, hidden_dim)
    
    seq_len, hidden_dim = lstm_avg.shape
    n_components = min(n_components, hidden_dim)
    
    fig, axes = plt.subplots(2, 1, figsize=figsize)
    
    # Plot individual components
    time_steps = np.arange(seq_len)
    for i in range(n_components):
        axes[0].plot(time_steps, lstm_avg[:, i], alpha=0.7, label=f"Component {i}")
    
    axes[0].set_xlabel("Time Step")
    axes[0].set_ylabel("Hidden State Value")
    axes[0].set_title(f"LSTM Hidden State Evolution (first {n_components} components)")
    axes[0].legend(loc="best", fontsize=8, ncol=2)
    axes[0].grid(True, alpha=0.3)
    
    # Plot mean and std over all components
    mean_state = np.mean(lstm_avg, axis=1)
    std_state = np.std(lstm_avg, axis=1)
    
    axes[1].plot(time_steps, mean_state, "b-", label="Mean", linewidth=2)
    axes[1].fill_between(time_steps, mean_state - std_state, mean_state + std_state, alpha=0.3, label="±1 Std")
    axes[1].set_xlabel("Time Step")
    axes[1].set_ylabel("Hidden State Value")
    axes[1].set_title("LSTM Hidden State Statistics")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved hidden state evolution plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_prediction_vs_actual(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 6),
):
    """
    Plot predictions vs actual values (scatter plots).
    
    Args:
        predictions: (n_samples, 2) - model predictions
        targets: (n_samples, 2) - ground truth targets
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    target_names = ["Primary", "Secondary"]
    
    for idx, (ax, name) in enumerate(zip(axes, target_names)):
        pred = predictions[:, idx]
        targ = targets[:, idx]
        
        # Scatter plot
        ax.scatter(targ, pred, alpha=0.5, s=20)
        
        # Perfect prediction line
        min_val = min(np.min(targ), np.min(pred))
        max_val = max(np.max(targ), np.max(pred))
        ax.plot([min_val, max_val], [min_val, max_val], "r--", linewidth=2, label="Perfect Prediction")
        
        # Calculate metrics for title
        mae = np.mean(np.abs(pred - targ))
        ss_res = np.sum((targ - pred) ** 2)
        ss_tot = np.sum((targ - np.mean(targ)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 1e-8 else 0
        
        ax.set_xlabel(f"Actual {name} Bubble Count")
        ax.set_ylabel(f"Predicted {name} Bubble Count")
        ax.set_title(f"{name} (MAE={mae:.3f}, R²={r2:.3f})")
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved prediction vs actual plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def plot_residuals(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (14, 10),
):
    """
    Plot residual analysis (errors vs predictions).
    
    Args:
        predictions: (n_samples, 2) - model predictions
        targets: (n_samples, 2) - ground truth targets
        output_path: Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    target_names = ["Primary", "Secondary"]
    
    for idx, (name, (ax1, ax2)) in enumerate(zip(target_names, axes)):
        pred = predictions[:, idx]
        targ = targets[:, idx]
        residuals = pred - targ
        
        # Residuals vs predictions
        ax1.scatter(pred, residuals, alpha=0.5, s=20)
        ax1.axhline(y=0, color="r", linestyle="--", linewidth=2)
        ax1.set_xlabel(f"Predicted {name} Bubble Count")
        ax1.set_ylabel(f"Residual (Predicted - Actual)")
        ax1.set_title(f"{name} Residuals vs Predictions")
        ax1.grid(True, alpha=0.3)
        
        # Residual distribution
        ax2.hist(residuals, bins=50, alpha=0.7, edgecolor="black")
        ax2.axvline(x=0, color="r", linestyle="--", linewidth=2)
        ax2.set_xlabel(f"Residual (Predicted - Actual)")
        ax2.set_ylabel("Frequency")
        ax2.set_title(f"{name} Residual Distribution")
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved residual plots to {output_path}")
    else:
        plt.show()
    
    plt.close()


def identify_failure_cases(
    predictions: np.ndarray,
    targets: np.ndarray,
    n_worst: int = 10,
) -> np.ndarray:
    """
    Identify worst prediction errors.
    
    Args:
        predictions: (n_samples, 2) - model predictions
        targets: (n_samples, 2) - ground truth targets
        n_worst: Number of worst cases to return
    
    Returns:
        indices: Array of sample indices with worst errors
    """
    # Calculate absolute errors per target
    errors_primary = np.abs(predictions[:, 0] - targets[:, 0])
    errors_secondary = np.abs(predictions[:, 1] - targets[:, 1])
    
    # Combined error (average)
    combined_errors = (errors_primary + errors_secondary) / 2
    
    # Get worst indices
    worst_indices = np.argsort(combined_errors)[-n_worst:][::-1]
    
    return worst_indices


def plot_metrics_comparison(
    metrics: Dict[str, float],
    output_path: Optional[Path] = None,
    figsize: Tuple[int, int] = (12, 8),
):
    """
    Plot comparison of metrics for primary vs secondary predictions.
    
    Args:
        metrics: Dictionary with metrics (should include mae_primary, mae_secondary, etc.)
        output_path: Path to save figure
        figsize: Figure size
    """
    metric_names = ["MAE", "RMSE", "MAPE", "R²"]
    metric_keys = {
        "MAE": ("mae_primary", "mae_secondary"),
        "RMSE": ("rmse_primary", "rmse_secondary"),
        "MAPE": ("mape_primary", "mape_secondary"),
        "R²": ("r2_primary", "r2_secondary"),
    }
    
    fig, axes = plt.subplots(2, 2, figsize=figsize)
    axes = axes.flatten()
    
    for metric_name, (ax, (key_primary, key_secondary)) in zip(metric_names, zip(axes, metric_keys.values())):
        primary_val = metrics.get(key_primary, 0)
        secondary_val = metrics.get(key_secondary, 0)
        
        # Bar plot
        x = np.arange(1)
        width = 0.35
        
        ax.bar(x - width/2, primary_val, width, label="Primary", alpha=0.8)
        ax.bar(x + width/2, secondary_val, width, label="Secondary", alpha=0.8)
        
        ax.set_ylabel(metric_name)
        ax.set_title(f"{metric_name} Comparison")
        ax.set_xticks(x)
        ax.set_xticklabels([""])
        ax.legend()
        ax.grid(True, alpha=0.3, axis="y")
    
    plt.tight_layout()
    
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=150, bbox_inches="tight")
        logger.info(f"Saved metrics comparison plot to {output_path}")
    else:
        plt.show()
    
    plt.close()


def analyze_errors(
    predictions: np.ndarray,
    targets: np.ndarray,
    output_dir: Optional[Path] = None,
    prefix: str = "error_analysis",
):
    """
    Comprehensive error analysis with multiple visualizations.
    
    Args:
        predictions: (n_samples, 2) - model predictions
        targets: (n_samples, 2) - ground truth targets
        output_dir: Directory to save all visualizations
        prefix: Prefix for output filenames
    """
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    # Prediction vs actual
    plot_prediction_vs_actual(
        predictions,
        targets,
        output_path=output_dir / f"{prefix}_prediction_vs_actual.png" if output_dir else None,
    )
    
    # Residual analysis
    plot_residuals(
        predictions,
        targets,
        output_path=output_dir / f"{prefix}_residuals.png" if output_dir else None,
    )
    
    # Identify worst cases
    worst_indices = identify_failure_cases(predictions, targets, n_worst=10)
    
    logger.info(f"Worst {len(worst_indices)} prediction errors:")
    for idx in worst_indices[:5]:  # Show top 5
        logger.info(
            f"  Sample {idx}: Primary pred={predictions[idx,0]:.2f} "
            f"actual={targets[idx,0]:.2f} (error={abs(predictions[idx,0]-targets[idx,0]):.2f}), "
            f"Secondary pred={predictions[idx,1]:.2f} actual={targets[idx,1]:.2f} "
            f"(error={abs(predictions[idx,1]-targets[idx,1]):.2f})"
        )
