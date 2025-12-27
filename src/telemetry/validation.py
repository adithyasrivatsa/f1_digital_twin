# State validation
# FORBIDDEN: torch, models.*, training.*

import numpy as np
from typing import Tuple, List


class StateValidator:
    """Validate state values are physically plausible."""
    
    # Physical bounds (before normalization)
    BOUNDS = {
        "velocity": (-150.0, 150.0),      # m/s
        "acceleration": (-100.0, 100.0),  # m/s²
        "angular_velocity": (-10.0, 10.0),  # rad/s
        "steering": (-1.0, 1.0),
        "throttle": (0.0, 1.0),
        "brake": (0.0, 1.0),
        "tire_temp": (0.0, 200.0),        # °C
        "tire_wear": (0.0, 1.0),
        "fuel": (0.0, 1.0),
        "track_position": (0.0, 1.0),
        "lateral_offset": (-20.0, 20.0),  # meters
        "heading_error": (-np.pi, np.pi),
        "curvature": (-0.5, 0.5),         # 1/m
    }
    
    @classmethod
    def validate(cls, state: np.ndarray) -> Tuple[bool, List[str]]:
        """Check state is within physical bounds.
        
        Args:
            state: State array, shape (34,)
            
        Returns:
            (is_valid, list of violations)
        """
        violations = []
        
        # Check for NaN/Inf first
        if np.any(np.isnan(state)):
            violations.append("State contains NaN")
            return False, violations
        if np.any(np.isinf(state)):
            violations.append("State contains Inf")
            return False, violations
        
        # Check velocity
        vel = state[0:3]
        low, high = cls.BOUNDS["velocity"]
        if np.any(vel < low) or np.any(vel > high):
            violations.append(f"Velocity out of bounds: {vel}")
        
        # Check acceleration
        acc = state[3:6]
        low, high = cls.BOUNDS["acceleration"]
        if np.any(acc < low) or np.any(acc > high):
            violations.append(f"Acceleration out of bounds: {acc}")
        
        # Check angular velocity
        ang_vel = state[6:9]
        low, high = cls.BOUNDS["angular_velocity"]
        if np.any(ang_vel < low) or np.any(ang_vel > high):
            violations.append(f"Angular velocity out of bounds: {ang_vel}")
        
        # Check steering
        steering = state[9]
        low, high = cls.BOUNDS["steering"]
        if steering < low or steering > high:
            violations.append(f"Steering out of bounds: {steering}")
        
        # Check throttle
        throttle = state[10]
        low, high = cls.BOUNDS["throttle"]
        if throttle < low or throttle > high:
            violations.append(f"Throttle out of bounds: {throttle}")
        
        # Check brake
        brake = state[11]
        low, high = cls.BOUNDS["brake"]
        if brake < low or brake > high:
            violations.append(f"Brake out of bounds: {brake}")
        
        # Check tire temps
        tire_temps = state[12:16]
        low, high = cls.BOUNDS["tire_temp"]
        if np.any(tire_temps < low) or np.any(tire_temps > high):
            violations.append(f"Tire temps out of bounds: {tire_temps}")
        
        # Check tire wear
        tire_wear = state[16:20]
        low, high = cls.BOUNDS["tire_wear"]
        if np.any(tire_wear < low) or np.any(tire_wear > high):
            violations.append(f"Tire wear out of bounds: {tire_wear}")
        
        # Check fuel
        fuel = state[20]
        low, high = cls.BOUNDS["fuel"]
        if fuel < low or fuel > high:
            violations.append(f"Fuel out of bounds: {fuel}")
        
        # Check track position
        track_pos = state[21]
        low, high = cls.BOUNDS["track_position"]
        if track_pos < low or track_pos > high:
            violations.append(f"Track position out of bounds: {track_pos}")
        
        # Check lateral offset
        lateral = state[22]
        low, high = cls.BOUNDS["lateral_offset"]
        if lateral < low or lateral > high:
            violations.append(f"Lateral offset out of bounds: {lateral}")
        
        # Check heading error
        heading = state[23]
        low, high = cls.BOUNDS["heading_error"]
        if heading < low or heading > high:
            violations.append(f"Heading error out of bounds: {heading}")
        
        # Check curvature
        curvature = state[24:34]
        low, high = cls.BOUNDS["curvature"]
        if np.any(curvature < low) or np.any(curvature > high):
            violations.append(f"Curvature out of bounds: {curvature}")
        
        return len(violations) == 0, violations
    
    @classmethod
    def clip(cls, state: np.ndarray) -> np.ndarray:
        """Clip state to physical bounds.
        
        Args:
            state: State array
            
        Returns:
            Clipped state array
        """
        clipped = state.copy()
        
        # Clip each component
        low, high = cls.BOUNDS["velocity"]
        clipped[0:3] = np.clip(clipped[0:3], low, high)
        
        low, high = cls.BOUNDS["acceleration"]
        clipped[3:6] = np.clip(clipped[3:6], low, high)
        
        low, high = cls.BOUNDS["angular_velocity"]
        clipped[6:9] = np.clip(clipped[6:9], low, high)
        
        low, high = cls.BOUNDS["steering"]
        clipped[9] = np.clip(clipped[9], low, high)
        
        low, high = cls.BOUNDS["throttle"]
        clipped[10] = np.clip(clipped[10], low, high)
        
        low, high = cls.BOUNDS["brake"]
        clipped[11] = np.clip(clipped[11], low, high)
        
        low, high = cls.BOUNDS["tire_temp"]
        clipped[12:16] = np.clip(clipped[12:16], low, high)
        
        low, high = cls.BOUNDS["tire_wear"]
        clipped[16:20] = np.clip(clipped[16:20], low, high)
        
        low, high = cls.BOUNDS["fuel"]
        clipped[20] = np.clip(clipped[20], low, high)
        
        low, high = cls.BOUNDS["track_position"]
        clipped[21] = np.clip(clipped[21], low, high)
        
        low, high = cls.BOUNDS["lateral_offset"]
        clipped[22] = np.clip(clipped[22], low, high)
        
        low, high = cls.BOUNDS["heading_error"]
        clipped[23] = np.clip(clipped[23], low, high)
        
        low, high = cls.BOUNDS["curvature"]
        clipped[24:34] = np.clip(clipped[24:34], low, high)
        
        return clipped
    
    @classmethod
    def check_nan_inf(cls, state: np.ndarray) -> bool:
        """Quick check for NaN or Inf values.
        
        Args:
            state: State array
            
        Returns:
            True if state is clean (no NaN/Inf)
        """
        return not (np.any(np.isnan(state)) or np.any(np.isinf(state)))
