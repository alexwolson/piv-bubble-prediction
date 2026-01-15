"""
Zarr archive reader for PIV and sensor data.

Loads experimental data from the Zarr archive created by sen-piv-sensor-archive.
"""

import logging
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import zarr

logger = logging.getLogger(__name__)


def open_zarr_archive(zarr_path: str) -> zarr.Group:
    """Open the zarr archive and return the root group."""
    zarr_path = Path(zarr_path).resolve()
    if not zarr_path.exists():
        raise FileNotFoundError(f"Zarr archive not found at {zarr_path}")
    
    # Suppress warnings about .DS_Store and other non-Zarr files in the directory
    # These are harmless macOS system files that shouldn't affect Zarr operations
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not recognized as a component.*",
            category=UserWarning  # ZarrUserWarning is a subclass of UserWarning
        )
        return zarr.open(str(zarr_path), mode="r")


def extract_experiment_metadata(experiment_group: zarr.Group) -> Dict:
    """Extract required metadata from an experiment group's attributes."""
    attrs = experiment_group.attrs.asdict()
    return {
        "sen": attrs["sen"],
        "variant": attrs.get("variant", "baseline"),
        "strand_speed_slug": attrs["strand_speed_slug"],
        "gas_flow_lpm": attrs["gas_flow_lpm"],
        "piv_run": attrs["piv_run"],
        "grid_width": attrs.get("grid_width"),
        "grid_height": attrs.get("grid_height"),
        "n_frames": attrs.get("n_frames"),
    }


def load_piv_data(experiment_group: zarr.Group) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load PIV velocity data from an experiment group.
    
    Returns:
        u: Velocity U component (frames × height × width)
        v: Velocity V component (frames × height × width)
        time_s: Timestamps for each frame
        x_mm: X coordinates (width,)
        y_mm: Y coordinates (height,)
    """
    piv_group = experiment_group["piv"]
    
    u = np.array(piv_group["u"])
    v = np.array(piv_group["v"])
    time_s = np.array(piv_group["time_s"])
    x_mm = np.array(piv_group["x_mm"])
    y_mm = np.array(piv_group["y_mm"])
    
    return u, v, time_s, x_mm, y_mm


def load_bubble_counts(experiment_group: zarr.Group) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load bubble counts from an experiment group.
    
    Returns:
        bubble_count_primary: Primary bubble counts (n_sensor_rows,)
        bubble_count_secondary: Secondary bubble counts (n_sensor_rows,)
    """
    preds_group = experiment_group["predictions"]
    bubble_counts = np.array(preds_group["bubble_counts"])
    
    if bubble_counts.shape[1] < 2:
        raise ValueError("bubble_counts must have at least 2 columns")
    
    return bubble_counts[:, 0], bubble_counts[:, 1]


def load_alignment_indices(experiment_group: zarr.Group) -> Optional[np.ndarray]:
    """
    Load alignment indices if available.
    
    Returns:
        sensor_row_index_per_frame: Array mapping each PIV frame to sensor row index (frames,)
        or None if not available
    """
    if "aligned" not in experiment_group:
        return None
    
    aligned_group = experiment_group["aligned"]
    if "sensor_row_index_per_frame" not in aligned_group:
        return None
    
    return np.array(aligned_group["sensor_row_index_per_frame"])


def find_all_experiments(zarr_root: zarr.Group) -> List[Tuple[zarr.Group, Dict]]:
    """
    Recursively find all experiment groups containing PIV data.
    
    Returns:
        List of (experiment_group, metadata) tuples
    """
    experiments: List[Tuple[zarr.Group, Dict]] = []

    def traverse(group: zarr.Group):
        if "piv" in group and "u" in group["piv"]:
            try:
                metadata = extract_experiment_metadata(group)
                experiments.append((group, metadata))
            except Exception as e:
                logger.warning(f"Failed to extract metadata from group: {e}")
                return
        for key in group.keys():
            # Skip macOS .DS_Store files and other hidden/system files
            if key.startswith('.'):
                continue
            subgroup = group[key]
            if isinstance(subgroup, zarr.Group):
                traverse(subgroup)

    traverse(zarr_root)
    return experiments
