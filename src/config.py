"""
Configuration utilities for device selection and memory optimization.

Optimized for Apple Silicon (M4 Pro) with 24GB RAM and MPS support,
with HPC cluster support for CUDA GPUs (e.g., Nibi H100).
"""

import logging
import os
from typing import Optional

import torch

logger = logging.getLogger(__name__)


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


def get_device(force_cpu: bool = False) -> torch.device:
    """
    Get best available device, prioritizing CUDA on HPC clusters, MPS locally.
    
    Priority order on HPC clusters: CUDA > CPU
    Priority order locally: CUDA > MPS > CPU
    
    Args:
        force_cpu: If True, force CPU usage regardless of availability
        
    Returns:
        torch.device: Best available device
    """
    if force_cpu:
        logger.info("Forcing CPU usage")
        return torch.device("cpu")
    
    on_cluster = is_on_cluster()
    
    # On HPC clusters (like nibi), prioritize CUDA for H100 GPUs
    if on_cluster:
        if torch.cuda.is_available():
            device_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "CUDA"
            logger.info(f"Using CUDA device on cluster: {device_name}")
            return torch.device("cuda")
        else:
            logger.info("Using CPU device on cluster (CUDA not available)")
            return torch.device("cpu")
    else:
        # Local development: prioritize CUDA if available (e.g., Linux with GPU)
        # then MPS for Apple Silicon, then CPU
        if torch.cuda.is_available():
            logger.info("Using CUDA device")
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            logger.info("Using MPS (Metal Performance Shaders) device")
            return torch.device("mps")
        else:
            logger.info("Using CPU device")
            return torch.device("cpu")


def get_recommended_batch_size(
    device: torch.device,
    ram_gb: Optional[int] = None,
    model_size_mb: Optional[int] = None,
    sequence_length: int = 20,
    height: int = 22,
    width: int = 30,
    channels: int = 2,
) -> int:
    """
    Get recommended batch size based on device and memory constraints.
    
    Args:
        device: torch.device to use
        ram_gb: Available RAM in GB (None = auto-detect based on device)
        model_size_mb: Model size in MB (estimated if None)
        sequence_length: Length of input sequences
        height: Height of PIV frames
        width: Width of PIV frames
        channels: Number of input channels (u, v)
        
    Returns:
        Recommended batch size
    """
    if model_size_mb is None:
        # Rough estimate: CNN-LSTM model ~20-50MB
        model_size_mb = 35
    
    # Auto-detect available memory based on device
    if ram_gb is None:
        if device.type == "cuda" and torch.cuda.is_available():
            # Use GPU memory for CUDA devices
            # H100 has 80GB, A100 has 40GB, etc.
            gpu_memory_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
            ram_gb = gpu_memory_gb
            logger.info(f"Detected GPU memory: {gpu_memory_gb:.1f} GB")
        else:
            # Default to local development memory (24GB M4 Pro)
            ram_gb = 24
    
    # Estimate memory per sample
    # Sequence data: sequence_length × height × width × channels × 4 bytes (float32)
    sample_size_mb = sequence_length * height * width * channels * 4 / (1024 * 1024)
    
    # Batch overhead (gradients, optimizer states, etc.)
    # Conservative estimate: 3-4x input size
    batch_overhead_factor = 3.5
    
    # System overhead (PyTorch, system, etc.)
    if device.type == "cuda":
        # GPU: higher overhead for CUDA operations
        system_overhead_gb = 2.0  # Less overhead on GPU nodes
    else:
        # CPU/MPS: standard overhead
        system_overhead_gb = 4.0
    
    # Available memory for batches
    available_memory_gb = ram_gb - system_overhead_gb - (model_size_mb / 1024)
    available_memory_mb = available_memory_gb * 1024
    
    # Calculate max batch size
    memory_per_sample_mb = sample_size_mb * batch_overhead_factor
    max_batch_size = int(available_memory_mb / memory_per_sample_mb)
    
    # Conservative recommendations based on device
    if device.type == "mps":
        # MPS works well with moderate batch sizes (Apple Silicon)
        recommended = min(32, max(8, max_batch_size // 2))
    elif device.type == "cuda":
        # CUDA/GPU: can handle larger batches, especially H100 (80GB)
        # For H100, we can be more aggressive with batch sizes
        if ram_gb >= 70:  # H100 or similar large GPU
            recommended = min(128, max(32, max_batch_size // 2))
        else:  # Smaller GPUs
            recommended = min(64, max(16, max_batch_size // 2))
    else:
        # CPU: smaller batches
        recommended = min(16, max(4, max_batch_size // 4))
    
    logger.info(
        f"Memory estimation: {sample_size_mb:.2f} MB/sample, "
        f"available ~{available_memory_gb:.1f} GB, "
        f"recommended batch size: {recommended}"
    )
    
    return recommended


def configure_dataloader(
    device: torch.device,
    num_workers: Optional[int] = None,
    pin_memory: Optional[bool] = None,
    cpus_per_task: Optional[int] = None,
) -> dict:
    """
    Configure DataLoader settings optimized for device.
    
    On HPC clusters, automatically detects available CPU cores from SLURM.
    
    Args:
        device: torch.device to use
        num_workers: Number of workers (None = auto)
        pin_memory: Whether to use pin_memory (None = auto)
        cpus_per_task: Number of CPUs per task (from SLURM, None = auto-detect)
        
    Returns:
        Dictionary of DataLoader kwargs
    """
    config = {}
    
    # Configure num_workers
    if num_workers is None:
        if device.type == "mps":
            # MPS doesn't benefit from multiprocessing
            config["num_workers"] = 0
        elif device.type == "cuda":
            # CUDA can benefit from workers, especially on compute nodes
            # On HPC clusters, use available CPUs from SLURM
            if cpus_per_task is not None:
                # Use SLURM-provided CPU count, but leave some for main process
                config["num_workers"] = max(2, min(cpus_per_task - 1, 8))
            else:
                # Try to detect from SLURM environment
                slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
                if slurm_cpus:
                    cpus = int(slurm_cpus)
                    config["num_workers"] = max(2, min(cpus - 1, 8))
                else:
                    # Default: conservative for local development
                    config["num_workers"] = 2
        else:
            # CPU: can use workers but keep it small
            config["num_workers"] = 2
    else:
        config["num_workers"] = num_workers
    
    # Configure pin_memory
    if pin_memory is None:
        if device.type == "mps":
            # MPS doesn't support pin_memory
            config["pin_memory"] = False
        elif device.type == "cuda":
            # CUDA benefits from pin_memory for faster CPU->GPU transfers
            config["pin_memory"] = True
        else:
            # CPU: no benefit from pin_memory
            config["pin_memory"] = False
    else:
        config["pin_memory"] = pin_memory
    
    return config


def estimate_memory_usage(
    batch_size: int,
    sequence_length: int = 20,
    height: int = 22,
    width: int = 30,
    channels: int = 2,
    model_size_mb: Optional[int] = None,
) -> dict:
    """
    Estimate memory usage for given configuration.
    
    Args:
        batch_size: Batch size
        sequence_length: Sequence length
        height: Frame height
        width: Frame width
        channels: Input channels
        model_size_mb: Model size in MB (estimated if None)
        
    Returns:
        Dictionary with memory estimates
    """
    if model_size_mb is None:
        model_size_mb = 35
    
    # Input data size
    sample_size_mb = sequence_length * height * width * channels * 4 / (1024 * 1024)
    batch_input_mb = sample_size_mb * batch_size
    
    # Gradient and optimizer states (roughly 3x input)
    batch_gradients_mb = batch_input_mb * 3
    
    # Total per batch
    total_batch_mb = batch_input_mb + batch_gradients_mb
    
    # System overhead
    system_overhead_gb = 4.0
    
    # Total estimate
    total_gb = (model_size_mb / 1024) + (total_batch_mb / 1024) + system_overhead_gb
    
    return {
        "sample_size_mb": sample_size_mb,
        "batch_input_mb": batch_input_mb,
        "batch_gradients_mb": batch_gradients_mb,
        "total_batch_mb": total_batch_mb,
        "model_size_mb": model_size_mb,
        "system_overhead_gb": system_overhead_gb,
        "total_estimate_gb": total_gb,
    }
