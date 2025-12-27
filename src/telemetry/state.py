# State construction from raw telemetry
# FORBIDDEN: torch, models.*, training.*

import numpy as np
from typing import Dict, Any
from ..core.types import StateDefinition


class StateBuilder:
    """Build state vectors from raw telemetry data."""
    
    def __init__(self, state_dim: int = 34):
        self.state_dim = state_dim
    
    def build(self, telemetry: Dict[str, Any]) -> np.ndarray:
        """Convert raw telemetry to state array.
        
        Args:
            telemetry: Dict with telemetry values
            
        Returns:
            State array, shape (state_dim,)
        """
        state = StateDefinition(
            velocity=np.array(telemetry.get("velocity", [0, 0, 0]), dtype=np.float32),
            acceleration=np.array(telemetry.get("acceleration", [0, 0, 0]), dtype=np.float32),
            angular_velocity=np.array(telemetry.get("angular_velocity", [0, 0, 0]), dtype=np.float32),
            steering_angle=float(telemetry.get("steering", 0)),
            throttle=float(telemetry.get("throttle", 0)),
            brake=float(telemetry.get("brake", 0)),
            tire_temps=np.array(telemetry.get("tire_temps", [80, 80, 80, 80]), dtype=np.float32),
            tire_wear=np.array(telemetry.get("tire_wear", [1, 1, 1, 1]), dtype=np.float32),
            fuel_mass=float(telemetry.get("fuel", 1.0)),
            track_position=float(telemetry.get("track_position", 0)),
            lateral_offset=float(telemetry.get("lateral_offset", 0)),
            heading_error=float(telemetry.get("heading_error", 0)),
            curvature_ahead=np.array(
                telemetry.get("curvature_ahead", [0] * 10),
                dtype=np.float32,
            ),
        )
        
        return state.to_array()
    
    def parse(self, state_array: np.ndarray) -> StateDefinition:
        """Parse state array back to StateDefinition.
        
        Args:
            state_array: State array, shape (state_dim,)
            
        Returns:
            StateDefinition object
        """
        return StateDefinition.from_array(state_array)
    
    def get_component(self, state_array: np.ndarray, component: str) -> np.ndarray:
        """Extract specific component from state array.
        
        Args:
            state_array: State array
            component: Component name
            
        Returns:
            Component values
        """
        indices = {
            "velocity": (0, 3),
            "acceleration": (3, 6),
            "angular_velocity": (6, 9),
            "steering": (9, 10),
            "throttle": (10, 11),
            "brake": (11, 12),
            "tire_temps": (12, 16),
            "tire_wear": (16, 20),
            "fuel": (20, 21),
            "track_position": (21, 22),
            "lateral_offset": (22, 23),
            "heading_error": (23, 24),
            "curvature_ahead": (24, 34),
        }
        
        if component not in indices:
            raise ValueError(f"Unknown component: {component}")
        
        start, end = indices[component]
        return state_array[start:end]
