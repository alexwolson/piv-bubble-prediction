"""
Compute aggregate metrics from PIV velocity fields.

This module implements computation of flow metrics from PIV velocity data,
including velocity magnitude, flow asymmetry, jet penetration, and turbulent
kinetic energy.
"""

import logging
from typing import Dict, List, Tuple, Optional

import numpy as np

logger = logging.getLogger(__name__)


def compute_velocity_magnitude(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Compute velocity magnitude from U and V components.
    
    Args:
        u: Velocity U component (frames × height × width) or (height × width)
        v: Velocity V component (frames × height × width) or (height × width)
    
    Returns:
        Velocity magnitude array with same shape as input
    """
    return np.sqrt(u**2 + v**2)


def define_quadrants(x_mm: np.ndarray, y_mm: np.ndarray) -> Dict[str, np.ndarray]:
    """
    Define quadrant masks based on spatial coordinates.
    
    Quadrants are defined relative to the center:
    - LL: Lower Left (x < center, y < center)
    - LQ: Lower Quadrant left (x < center, y >= center)
    - RQ: Right Quadrant (x >= center, y >= center)
    - RR: Lower Right (x >= center, y < center)
    
    Args:
        x_mm: X coordinates (width,)
        y_mm: Y coordinates (height,)
    
    Returns:
        Dictionary mapping quadrant names to boolean masks (height × width)
    """
    # Create meshgrid for spatial coordinates
    X, Y = np.meshgrid(x_mm, y_mm)
    
    # Find center
    x_center = np.mean(x_mm)
    y_center = np.mean(y_mm)
    
    # Define quadrants
    # Note: "Lower" typically means closer to SEN (negative Y in many coordinate systems)
    # Adjust logic based on actual coordinate system orientation
    quadrants = {
        "LL": (X < x_center) & (Y < y_center),  # Lower Left
        "LQ": (X < x_center) & (Y >= y_center),  # Lower Quadrant (left)
        "RQ": (X >= x_center) & (Y >= y_center),  # Right Quadrant
        "RR": (X >= x_center) & (Y < y_center),   # Lower Right
    }
    
    return quadrants


def compute_mean_velocity_magnitude_per_quadrant(
    u: np.ndarray,
    v: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute mean velocity magnitude for each quadrant.
    
    Args:
        u: Velocity U component (frames × height × width)
        v: Velocity V component (frames × height × width)
        x_mm: X coordinates (width,)
        y_mm: Y coordinates (height,)
    
    Returns:
        Dictionary mapping quadrant names to mean velocity magnitude (frames,)
    """
    velocity_mag = compute_velocity_magnitude(u, v)
    quadrants = define_quadrants(x_mm, y_mm)
    
    results = {}
    for quadrant_name, mask in quadrants.items():
        # Apply mask to each frame
        quadrant_velocities = []
        for frame_idx in range(velocity_mag.shape[0]):
            frame_vel = velocity_mag[frame_idx]
            quadrant_vel = frame_vel[mask]
            if len(quadrant_vel) > 0:
                mean_vel = np.mean(quadrant_vel)
            else:
                mean_vel = 0.0
            quadrant_velocities.append(mean_vel)
        results[quadrant_name] = np.array(quadrant_velocities)
    
    return results


def compute_flow_asymmetry_index(
    u: np.ndarray,
    v: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> np.ndarray:
    """
    Compute flow asymmetry index (difference between left and right quadrants).
    
    Asymmetry = (mean velocity in left quadrants) - (mean velocity in right quadrants)
    Left: LL + LQ, Right: RQ + RR
    
    Args:
        u: Velocity U component (frames × height × width)
        v: Velocity V component (frames × height × width)
        x_mm: X coordinates (width,)
        y_mm: Y coordinates (height,)
    
    Returns:
        Flow asymmetry index (frames,)
    """
    quadrant_velocities = compute_mean_velocity_magnitude_per_quadrant(u, v, x_mm, y_mm)
    
    # Left quadrants: LL + LQ
    left_velocity = (quadrant_velocities["LL"] + quadrant_velocities["LQ"]) / 2.0
    
    # Right quadrants: RQ + RR
    right_velocity = (quadrant_velocities["RQ"] + quadrant_velocities["RR"]) / 2.0
    
    # Asymmetry index
    asymmetry = left_velocity - right_velocity
    
    return asymmetry


def compute_jet_penetration_depth(
    u: np.ndarray,
    v: np.ndarray,
    y_mm: np.ndarray,
    threshold_fraction: float = 0.5,
) -> np.ndarray:
    """
    Compute jet penetration depth.
    
    Penetration depth is the distance from the top (SEN) where velocity
    drops below a threshold (default: 50% of peak velocity).
    
    Args:
        u: Velocity U component (frames × height × width)
        v: Velocity V component (frames × height × width)
        y_mm: Y coordinates (height,) - typically Y=0 at SEN, increasing downward
        threshold_fraction: Fraction of peak velocity for threshold (default: 0.5)
    
    Returns:
        Jet penetration depth in mm (frames,)
    """
    velocity_mag = compute_velocity_magnitude(u, v)
    
    penetration_depths = []
    
    for frame_idx in range(velocity_mag.shape[0]):
        frame_vel = velocity_mag[frame_idx]
        
        # Find peak velocity in this frame
        peak_velocity = np.max(frame_vel)
        threshold = peak_velocity * threshold_fraction
        
        # Compute mean velocity along each Y row (averaging across X)
        mean_vel_per_row = np.mean(frame_vel, axis=1)
        
        # Find where velocity drops below threshold
        # Start from top (SEN) and find first row below threshold
        penetration = None
        for y_idx, mean_vel in enumerate(mean_vel_per_row):
            if mean_vel < threshold:
                penetration = y_mm[y_idx]
                break
        
        # If velocity never drops below threshold, use maximum Y
        if penetration is None:
            penetration = np.max(y_mm)
        
        penetration_depths.append(penetration)
    
    return np.array(penetration_depths)


def compute_turbulent_kinetic_energy(
    u: np.ndarray,
    v: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
    window_size: Optional[int] = None,
) -> Dict[str, np.ndarray]:
    """
    Compute turbulent kinetic energy (TKE) per quadrant.
    
    TKE = 0.5 * (u'² + v'²)
    where u', v' are velocity fluctuations (deviation from mean)
    
    Args:
        u: Velocity U component (frames × height × width)
        v: Velocity V component (frames × height × width)
        x_mm: X coordinates (width,)
        y_mm: Y coordinates (height,)
        window_size: Optional window size for computing mean velocity.
                     If None, uses mean over all frames.
    
    Returns:
        Dictionary mapping quadrant names to TKE (frames,)
    """
    quadrants = define_quadrants(x_mm, y_mm)
    
    # Compute mean velocity for TKE calculation
    if window_size is None:
        # Use mean over all frames
        u_mean = np.mean(u, axis=0, keepdims=True)
        v_mean = np.mean(v, axis=0, keepdims=True)
    else:
        # Use rolling window mean (simplified - could use proper rolling window)
        u_mean = np.mean(u, axis=0, keepdims=True)
        v_mean = np.mean(v, axis=0, keepdims=True)
    
    # Compute fluctuations
    u_prime = u - u_mean
    v_prime = v - v_mean
    
    # Compute TKE = 0.5 * (u'² + v'²)
    tke = 0.5 * (u_prime**2 + v_prime**2)
    
    # Average TKE per quadrant
    results = {}
    for quadrant_name, mask in quadrants.items():
        quadrant_tke = []
        for frame_idx in range(tke.shape[0]):
            frame_tke = tke[frame_idx]
            quadrant_tke_values = frame_tke[mask]
            if len(quadrant_tke_values) > 0:
                mean_tke = np.mean(quadrant_tke_values)
            else:
                mean_tke = 0.0
            quadrant_tke.append(mean_tke)
        results[quadrant_name] = np.array(quadrant_tke)
    
    return results


def compute_all_tier1_metrics(
    u: np.ndarray,
    v: np.ndarray,
    time_s: np.ndarray,
    x_mm: np.ndarray,
    y_mm: np.ndarray,
) -> Dict[str, np.ndarray]:
    """
    Compute all Tier 1 PIV metrics.
    
    Returns:
        Dictionary with keys:
        - 'mean_velocity_LL', 'mean_velocity_LQ', 'mean_velocity_RQ', 'mean_velocity_RR'
        - 'flow_asymmetry_index'
        - 'jet_penetration_depth'
        - 'tke_LL', 'tke_LQ', 'tke_RQ', 'tke_RR'
        All arrays have shape (frames,)
    """
    metrics = {}
    
    # Mean velocity magnitude per quadrant (4 metrics)
    quadrant_velocities = compute_mean_velocity_magnitude_per_quadrant(u, v, x_mm, y_mm)
    for quadrant_name in ["LL", "LQ", "RQ", "RR"]:
        metrics[f"mean_velocity_{quadrant_name}"] = quadrant_velocities[quadrant_name]
    
    # Flow asymmetry index (1 metric)
    metrics["flow_asymmetry_index"] = compute_flow_asymmetry_index(u, v, x_mm, y_mm)
    
    # Jet penetration depth (1 metric)
    metrics["jet_penetration_depth"] = compute_jet_penetration_depth(u, v, y_mm)
    
    # Turbulent kinetic energy per quadrant (4 metrics)
    quadrant_tke = compute_turbulent_kinetic_energy(u, v, x_mm, y_mm)
    for quadrant_name in ["LL", "LQ", "RQ", "RR"]:
        metrics[f"tke_{quadrant_name}"] = quadrant_tke[quadrant_name]
    
    return metrics


def aggregate_metrics_using_alignment(
    metrics: Dict[str, np.ndarray],
    sensor_row_indices: np.ndarray,
    n_sensor_rows: int,
    aggregation_method: str = "mean",
) -> Dict[str, np.ndarray]:
    """
    Aggregate PIV metrics to sensor rows using alignment indices.
    
    This function uses pre-computed alignment indices that map each PIV frame
    to a sensor row index. This is more reliable than timestamp matching.
    
    Args:
        metrics: Dictionary of metric arrays (frames,)
        sensor_row_indices: Array mapping each PIV frame to sensor row index (frames,)
        n_sensor_rows: Number of sensor rows
        aggregation_method: 'mean', 'median', or 'max'
    
    Returns:
        Dictionary of aggregated metrics (n_sensor_rows,)
    """
    aggregated = {}
    
    for metric_name, metric_values in metrics.items():
        aggregated_values = np.full(n_sensor_rows, np.nan)
        
        # Group PIV frames by sensor row index
        for sensor_idx in range(n_sensor_rows):
            mask = sensor_row_indices == sensor_idx
            
            if np.any(mask):
                window_values = metric_values[mask]
                if aggregation_method == "mean":
                    aggregated_values[sensor_idx] = np.mean(window_values)
                elif aggregation_method == "median":
                    aggregated_values[sensor_idx] = np.median(window_values)
                elif aggregation_method == "max":
                    aggregated_values[sensor_idx] = np.max(window_values)
                else:
                    raise ValueError(f"Unknown aggregation method: {aggregation_method}")
        
        aggregated[metric_name] = aggregated_values
    
    return aggregated
