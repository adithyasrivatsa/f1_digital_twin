# State normalization
# FORBIDDEN: torch, models.*, training.*

import numpy as np
from ..core.types import NormalizationParams


def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize state to approximately [-1, 1] range.
    
    Normalization is critical for:
    1. Gradient stability (no exploding gradients from large values)
    2. Fair weighting (all features contribute equally initially)
    3. Activation function effectiveness (tanh, sigmoid work in [-1, 1])
    
    Args:
        state: Raw state array, shape (34,)
        
    Returns:
        Normalized state array
    """
    params = NormalizationParams()
    normalized = state.copy()
    
    # Velocity: divide by max
    normalized[0:3] /= params.velocity_max
    
    # Acceleration: divide by max
    normalized[3:6] /= params.acceleration_max
    
    # Angular velocity: divide by max
    normalized[6:9] /= params.angular_vel_max
    
    # Controls already in [0, 1] or [-1, 1]
    # normalized[9:12] unchanged
    
    # Tire temps: normalize to [0, 1]
    normalized[12:16] /= params.tire_temp_max
    
    # Tire wear already in [0, 1]
    # normalized[16:20] unchanged
    
    # Fuel already in [0, 1]
    # normalized[20] unchanged
    
    # Track position already in [0, 1]
    # normalized[21] unchanged
    
    # Lateral offset: divide by max
    normalized[22] /= params.lateral_max
    
    # Heading error: divide by π
    normalized[23] /= np.pi
    
    # Curvature: divide by max
    normalized[24:34] /= params.curvature_max
    
    return normalized.astype(np.float32)


def denormalize_state(normalized: np.ndarray) -> np.ndarray:
    """Reverse normalization for interpretation.
    
    Args:
        normalized: Normalized state array
        
    Returns:
        Denormalized state array with physical units
    """
    params = NormalizationParams()
    state = normalized.copy()
    
    state[0:3] *= params.velocity_max
    state[3:6] *= params.acceleration_max
    state[6:9] *= params.angular_vel_max
    state[12:16] *= params.tire_temp_max
    state[22] *= params.lateral_max
    state[23] *= np.pi
    state[24:34] *= params.curvature_max
    
    return state.astype(np.float32)


class RunningNormalizer:
    """Online normalization with running statistics.
    
    Computes running mean and std for normalization.
    Useful when true statistics are unknown.
    """
    
    def __init__(self, shape: tuple, epsilon: float = 1e-8):
        """Initialize normalizer.
        
        Args:
            shape: Shape of data to normalize
            epsilon: Small constant for numerical stability
        """
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 0
        self.epsilon = epsilon
    
    def update(self, x: np.ndarray) -> None:
        """Update running statistics with new data.
        
        Args:
            x: New data, shape (batch, *shape) or (*shape,)
        """
        if x.ndim == len(self.mean.shape):
            x = x[np.newaxis, ...]
        
        batch_mean = x.mean(axis=0)
        batch_var = x.var(axis=0)
        batch_count = x.shape[0]
        
        self._update_from_moments(batch_mean, batch_var, batch_count)
    
    def _update_from_moments(
        self,
        batch_mean: np.ndarray,
        batch_var: np.ndarray,
        batch_count: int,
    ) -> None:
        """Update from batch moments using Welford's algorithm."""
        delta = batch_mean - self.mean
        total_count = self.count + batch_count
        
        new_mean = self.mean + delta * batch_count / total_count
        
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        m2 = m_a + m_b + delta ** 2 * self.count * batch_count / total_count
        
        new_var = m2 / total_count
        
        self.mean = new_mean
        self.var = new_var
        self.count = total_count
    
    def normalize(self, x: np.ndarray) -> np.ndarray:
        """Normalize data using running statistics.
        
        Args:
            x: Data to normalize
            
        Returns:
            Normalized data
        """
        return (x - self.mean) / np.sqrt(self.var + self.epsilon)
    
    def denormalize(self, x: np.ndarray) -> np.ndarray:
        """Denormalize data.
        
        Args:
            x: Normalized data
            
        Returns:
            Denormalized data
        """
        return x * np.sqrt(self.var + self.epsilon) + self.mean
