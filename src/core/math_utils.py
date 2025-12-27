# Mathematical utilities
# FORBIDDEN: torch, logging, any I/O

import numpy as np
from typing import Tuple


def normalize_angle(angle: float) -> float:
    """Normalize angle to [-pi, pi] range.
    
    Args:
        angle: Angle in radians
        
    Returns:
        Normalized angle in [-pi, pi]
    """
    while angle > np.pi:
        angle -= 2 * np.pi
    while angle < -np.pi:
        angle += 2 * np.pi
    return angle


def interpolate_curvature(
    track_positions: np.ndarray,
    curvatures: np.ndarray,
    query_position: float,
    num_samples: int = 10,
    lookahead_distance: float = 100.0,
) -> np.ndarray:
    """Interpolate curvature values ahead of current position.
    
    Args:
        track_positions: Array of track positions [0, 1]
        curvatures: Array of curvature values at each position
        query_position: Current track position [0, 1]
        num_samples: Number of lookahead samples
        lookahead_distance: Distance to look ahead in meters
        
    Returns:
        Array of curvature samples ahead
    """
    # Generate query points
    track_length = 5793.0  # Monza default, should come from config
    lookahead_fraction = lookahead_distance / track_length
    
    query_points = np.linspace(
        query_position,
        query_position + lookahead_fraction,
        num_samples,
    )
    # Wrap around track
    query_points = query_points % 1.0
    
    # Linear interpolation
    result = np.interp(query_points, track_positions, curvatures, period=1.0)
    
    return result.astype(np.float32)


def rotate_2d(x: float, y: float, angle: float) -> Tuple[float, float]:
    """Rotate 2D point by angle.
    
    Args:
        x, y: Point coordinates
        angle: Rotation angle in radians
        
    Returns:
        Rotated (x, y) coordinates
    """
    cos_a = np.cos(angle)
    sin_a = np.sin(angle)
    return (
        x * cos_a - y * sin_a,
        x * sin_a + y * cos_a,
    )


def clamp(value: float, min_val: float, max_val: float) -> float:
    """Clamp value to range.
    
    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value
        
    Returns:
        Clamped value
    """
    return max(min_val, min(max_val, value))


def moving_average(values: np.ndarray, window: int) -> np.ndarray:
    """Compute moving average.
    
    Args:
        values: Input array
        window: Window size
        
    Returns:
        Moving average array (same length, padded at start)
    """
    if len(values) < window:
        return values
    
    cumsum = np.cumsum(values)
    cumsum[window:] = cumsum[window:] - cumsum[:-window]
    
    result = np.zeros_like(values)
    result[:window] = cumsum[:window] / np.arange(1, window + 1)
    result[window:] = cumsum[window:] / window
    
    return result
