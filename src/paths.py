"""
Environment-aware path configuration for PIV bubble prediction.

Automatically detects if running on HPC cluster (e.g., nibi) and adjusts paths accordingly.
"""

import os
from pathlib import Path
from typing import Optional


def is_on_cluster() -> bool:
    """
    Detect if running on an HPC cluster (e.g., nibi).
    
    Checks SLURM environment variables and hostname.
    
    Returns:
        True if running on a cluster, False otherwise
    """
    # Check SLURM environment variables
    slurm_partition = os.environ.get("SLURM_JOB_PARTITION", "").lower()
    slurm_nodelist = os.environ.get("SLURM_JOB_NODELIST", "").lower()
    hostname = os.environ.get("HOSTNAME", "").lower()
    
    # Check if any indicate cluster environment
    cluster_indicators = ["nibi", "login", "compute", "gpu"]
    environment_str = f"{slurm_partition} {slurm_nodelist} {hostname}"
    
    return any(indicator in environment_str for indicator in cluster_indicators)


def get_zarr_path(default_cluster_path: Optional[str] = None) -> str:
    """
    Get Zarr archive path based on environment.
    
    On cluster: uses PIV_DATA_PATH environment variable or default cluster path.
    Locally: uses default local path.
    
    Args:
        default_cluster_path: Default path to use on cluster if PIV_DATA_PATH not set.
                             If None, uses /project/<group>/data/raw/all_experiments.zarr/
                             (user should replace <group> with their group name)
    
    Returns:
        Path to Zarr archive
    """
    # Check if path is explicitly set
    explicit_path = os.environ.get("PIV_DATA_PATH")
    if explicit_path:
        return explicit_path
    
    if is_on_cluster():
        # On cluster: use project space (not home due to size limits)
        if default_cluster_path:
            return default_cluster_path
        else:
            # Default cluster path (user should replace <group> with their group)
            return "/project/<group>/data/raw/all_experiments.zarr/"
    else:
        # Local development: use relative path
        return "data/raw/all_experiments.zarr/"


def get_output_dir(default_dir: str = "models") -> str:
    """
    Get output directory for models and results.
    
    On cluster: uses PIV_OUTPUT_DIR if set, otherwise uses default_dir.
    Locally: uses default_dir.
    
    Args:
        default_dir: Default output directory name
    
    Returns:
        Path to output directory
    """
    # Check if output directory is explicitly set
    explicit_dir = os.environ.get("PIV_OUTPUT_DIR")
    if explicit_dir:
        return explicit_dir
    
    # Use default directory
    return default_dir


def get_log_dir(default_dir: str = "logs") -> str:
    """
    Get log directory.
    
    Args:
        default_dir: Default log directory name
    
    Returns:
        Path to log directory
    """
    # Check if log directory is explicitly set
    explicit_dir = os.environ.get("PIV_LOG_DIR")
    if explicit_dir:
        return explicit_dir
    
    # Use default directory
    return default_dir


def ensure_dir_exists(path: str) -> Path:
    """
    Ensure directory exists, creating it if necessary.
    
    Args:
        path: Directory path
    
    Returns:
        Path object pointing to the directory
    """
    dir_path = Path(path)
    dir_path.mkdir(parents=True, exist_ok=True)
    return dir_path


# Convenience constants (can be imported and used directly)
ON_NIBI = is_on_cluster()
ZARR_PATH = get_zarr_path()
