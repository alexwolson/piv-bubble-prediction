"""
Data loading and preprocessing for PIV → bubble count prediction.

Creates sequences from PIV velocity fields aligned to bubble count timestamps.
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import zarr

from src.zarr_reader import (
    open_zarr_archive,
    find_all_experiments,
    load_piv_data,
    load_bubble_counts,
    load_alignment_indices,
)

logger = logging.getLogger(__name__)


def create_sequences(
    piv_frames: np.ndarray,
    bubble_counts: np.ndarray,
    alignment_indices: np.ndarray,
    sequence_length: int = 20,
    stride: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create sequences from PIV frames aligned to bubble counts.
    
    Args:
        piv_frames: PIV velocity frames (n_frames, height, width, channels)
        bubble_counts: Bubble counts (n_sensor_rows, 2) - primary and secondary
        alignment_indices: Maps each PIV frame to sensor row index (n_frames,)
        sequence_length: Number of frames per sequence
        stride: Stride for sliding window (1 = overlapping, sequence_length = non-overlapping)
    
    Returns:
        sequences: (n_sequences, sequence_length, height, width, channels)
        targets: (n_sequences, 2) - bubble_count_primary, bubble_count_secondary
    """
    n_frames, height, width, channels = piv_frames.shape
    n_sensor_rows = bubble_counts.shape[0]
    
    sequences = []
    targets = []
    
    # Create sliding windows
    for start_idx in range(0, n_frames - sequence_length + 1, stride):
        end_idx = start_idx + sequence_length
        
        # Get sequence of frames
        sequence = piv_frames[start_idx:end_idx]  # (sequence_length, height, width, channels)
        
        # Get target bubble count for the last frame in sequence
        last_frame_idx = end_idx - 1
        sensor_row_idx = alignment_indices[last_frame_idx]
        
        # Validate sensor row index
        if sensor_row_idx < 0 or sensor_row_idx >= n_sensor_rows:
            continue
        
        target = bubble_counts[sensor_row_idx]  # (2,) - primary and secondary
        
        sequences.append(sequence)
        targets.append(target)
    
    if len(sequences) == 0:
        raise ValueError("No valid sequences created")
    
    sequences = np.array(sequences)  # (n_sequences, sequence_length, height, width, channels)
    targets = np.array(targets)  # (n_sequences, 2)
    
    return sequences, targets


def normalize_velocity_fields(
    u: np.ndarray,
    v: np.ndarray,
    method: str = "per_experiment",
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Normalize velocity fields.
    
    Args:
        u: U velocity component (frames, height, width)
        v: V velocity component (frames, height, width)
        method: 'per_experiment' or 'global'
    
    Returns:
        u_norm, v_norm: Normalized velocity fields
        stats: Dictionary with normalization statistics
    """
    if method == "per_experiment":
        u_mean = np.mean(u)
        u_std = np.std(u)
        v_mean = np.mean(v)
        v_std = np.std(v)
    else:
        # Global normalization would require all experiments
        # For now, use per_experiment
        u_mean = np.mean(u)
        u_std = np.std(u)
        v_mean = np.mean(v)
        v_std = np.std(v)
    
    u_norm = (u - u_mean) / (u_std + 1e-8)
    v_norm = (v - v_mean) / (v_std + 1e-8)
    
    stats = {
        "u_mean": u_mean,
        "u_std": u_std,
        "v_mean": v_mean,
        "v_std": v_std,
    }
    
    return u_norm, v_norm, stats


def load_experiment_data(
    experiment_group: zarr.Group,
    sequence_length: int = 20,
    stride: int = 1,
    normalize: bool = True,
) -> Tuple[np.ndarray, np.ndarray, Dict]:
    """
    Load and prepare data for a single experiment.
    
    Args:
        experiment_group: Zarr experiment group
        sequence_length: Number of frames per sequence
        stride: Stride for sliding window
        normalize: Whether to normalize velocity fields
    
    Returns:
        sequences: (n_sequences, sequence_length, height, width, channels)
        targets: (n_sequences, 2)
        metadata: Experiment metadata
    """
    # Load PIV data
    u, v, piv_time_s, x_mm, y_mm = load_piv_data(experiment_group)
    
    # Load bubble counts
    bubble_count_primary, bubble_count_secondary = load_bubble_counts(experiment_group)
    bubble_counts = np.column_stack([bubble_count_primary, bubble_count_secondary])
    
    # Load alignment indices
    alignment_indices = load_alignment_indices(experiment_group)
    if alignment_indices is None:
        raise ValueError("Alignment indices not found in experiment group")
    
    # Normalize velocity fields
    if normalize:
        u, v, norm_stats = normalize_velocity_fields(u, v)
    else:
        norm_stats = {}
    
    # Stack u and v into channels
    # Shape: (frames, height, width, channels=2)
    piv_frames = np.stack([u, v], axis=-1)
    
    # Create sequences
    sequences, targets = create_sequences(
        piv_frames,
        bubble_counts,
        alignment_indices,
        sequence_length=sequence_length,
        stride=stride,
    )
    
    # Extract metadata
    from src.zarr_reader import extract_experiment_metadata
    metadata = extract_experiment_metadata(experiment_group)
    metadata["norm_stats"] = norm_stats
    metadata["n_sequences"] = len(sequences)
    
    return sequences, targets, metadata


def load_all_experiments(
    zarr_path: str,
    sequence_length: int = 20,
    stride: int = 1,
    normalize: bool = True,
    limit: Optional[int] = None,
    return_per_experiment: bool = False,
) -> Tuple[np.ndarray, np.ndarray, List[Dict], Optional[np.ndarray]]:
    """
    Load data from all experiments in Zarr archive.
    
    Args:
        zarr_path: Path to Zarr archive
        sequence_length: Number of frames per sequence
        stride: Stride for sliding window
        normalize: Whether to normalize velocity fields
        limit: Limit number of experiments (for testing)
        return_per_experiment: If True, also return experiment IDs per sequence
    
    Returns:
        all_sequences: (total_sequences, sequence_length, height, width, channels)
        all_targets: (total_sequences, 2)
        all_metadata: List of metadata dictionaries
        experiment_ids: (Optional) Array of experiment IDs for each sequence
    """
    zarr_root = open_zarr_archive(zarr_path)
    experiments = find_all_experiments(zarr_root)
    
    if limit:
        experiments = experiments[:limit]
    
    logger.info(f"Loading data from {len(experiments)} experiments")
    
    all_sequences = []
    all_targets = []
    all_metadata = []
    experiment_ids = [] if return_per_experiment else None
    
    for exp_idx, (exp_group, metadata) in enumerate(experiments):
        try:
            sequences, targets, exp_metadata = load_experiment_data(
                exp_group,
                sequence_length=sequence_length,
                stride=stride,
                normalize=normalize,
            )
            all_sequences.append(sequences)
            all_targets.append(targets)
            all_metadata.append(exp_metadata)
            
            if return_per_experiment:
                # Track which experiment each sequence belongs to
                experiment_ids.extend([exp_idx] * len(sequences))
            
            logger.info(
                f"Loaded {len(sequences)} sequences from {metadata['sen']} "
                f"{metadata['variant']} PIV{metadata['piv_run']:02d}"
            )
        except Exception as e:
            logger.warning(f"Failed to load experiment {metadata.get('sen', '?')}: {e}")
            continue
    
    if len(all_sequences) == 0:
        raise ValueError("No experiments loaded successfully")
    
    # Concatenate all sequences
    all_sequences = np.concatenate(all_sequences, axis=0)
    all_targets = np.concatenate(all_targets, axis=0)
    
    if return_per_experiment:
        experiment_ids = np.array(experiment_ids, dtype=np.int32)
    else:
        experiment_ids = None
    
    logger.info(f"Total sequences: {len(all_sequences)}")
    logger.info(f"Sequence shape: {all_sequences.shape}")
    logger.info(f"Target shape: {all_targets.shape}")
    
    return all_sequences, all_targets, all_metadata, experiment_ids
