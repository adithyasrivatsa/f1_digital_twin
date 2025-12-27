# Core type definitions
# FORBIDDEN: torch, logging, any I/O

from dataclasses import dataclass
from typing import Tuple
import numpy as np


@dataclass(frozen=True)
class ActionBounds:
    """Immutable action space bounds."""
    steering_min: float = -1.0
    steering_max: float = 1.0
    throttle_min: float = 0.0
    throttle_max: float = 1.0
    brake_min: float = 0.0
    brake_max: float = 1.0
    
    @property
    def low(self) -> np.ndarray:
        return np.array([self.steering_min, self.throttle_min, self.brake_min], dtype=np.float32)
    
    @property
    def high(self) -> np.ndarray:
        return np.array([self.steering_max, self.throttle_max, self.brake_max], dtype=np.float32)
    
    @property
    def dimension(self) -> int:
        return 3


@dataclass
class StateDefinition:
    """F1 racing state vector specification.
    
    Total dimensions: 34
    All values normalized to approximately [-1, 1] or [0, 1]
    """
    velocity: np.ndarray          # [vx, vy, vz] m/s
    acceleration: np.ndarray      # [ax, ay, az] m/s²
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s
    steering_angle: float         # [-1, 1] normalized
    throttle: float               # [0, 1]
    brake: float                  # [0, 1]
    tire_temps: np.ndarray        # [FL, FR, RL, RR] °C
    tire_wear: np.ndarray         # [FL, FR, RL, RR] [0, 1] remaining
    fuel_mass: float              # [0, 1] normalized
    track_position: float         # [0, 1] progress around lap
    lateral_offset: float         # meters from racing line
    heading_error: float          # radians from racing line tangent
    curvature_ahead: np.ndarray   # Next 10 curvature samples
    
    @property
    def dimension(self) -> int:
        return 34
    
    def to_array(self) -> np.ndarray:
        """Flatten to numpy array."""
        return np.concatenate([
            self.velocity,
            self.acceleration,
            self.angular_velocity,
            np.array([self.steering_angle, self.throttle, self.brake]),
            self.tire_temps,
            self.tire_wear,
            np.array([self.fuel_mass]),
            np.array([self.track_position, self.lateral_offset, self.heading_error]),
            self.curvature_ahead,
        ]).astype(np.float32)
    
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "StateDefinition":
        """Reconstruct from numpy array."""
        assert arr.shape == (34,), f"Expected shape (34,), got {arr.shape}"
        return cls(
            velocity=arr[0:3].copy(),
            acceleration=arr[3:6].copy(),
            angular_velocity=arr[6:9].copy(),
            steering_angle=float(arr[9]),
            throttle=float(arr[10]),
            brake=float(arr[11]),
            tire_temps=arr[12:16].copy(),
            tire_wear=arr[16:20].copy(),
            fuel_mass=float(arr[20]),
            track_position=float(arr[21]),
            lateral_offset=float(arr[22]),
            heading_error=float(arr[23]),
            curvature_ahead=arr[24:34].copy(),
        )
    
    @classmethod
    def zeros(cls) -> "StateDefinition":
        """Create zero-initialized state."""
        return cls(
            velocity=np.zeros(3, dtype=np.float32),
            acceleration=np.zeros(3, dtype=np.float32),
            angular_velocity=np.zeros(3, dtype=np.float32),
            steering_angle=0.0,
            throttle=0.0,
            brake=0.0,
            tire_temps=np.full(4, 80.0, dtype=np.float32),
            tire_wear=np.ones(4, dtype=np.float32),
            fuel_mass=1.0,
            track_position=0.0,
            lateral_offset=0.0,
            heading_error=0.0,
            curvature_ahead=np.zeros(10, dtype=np.float32),
        )


@dataclass(frozen=True)
class NormalizationParams:
    """Physical limits for state normalization."""
    velocity_max: float = 100.0       # m/s (~360 km/h)
    acceleration_max: float = 50.0    # m/s² (~5g)
    angular_vel_max: float = 5.0      # rad/s
    tire_temp_max: float = 120.0      # °C
    lateral_max: float = 5.0          # meters
    curvature_max: float = 0.1        # 1/m (10m radius minimum)
