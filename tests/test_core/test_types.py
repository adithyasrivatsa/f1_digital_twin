# Tests for core types

import pytest
import numpy as np
from src.core.types import StateDefinition, ActionBounds, NormalizationParams


class TestStateDefinition:
    
    def test_dimension(self):
        """State dimension should be 34."""
        state = StateDefinition.zeros()
        assert state.dimension == 34
    
    def test_to_array_shape(self):
        """to_array should return correct shape."""
        state = StateDefinition.zeros()
        arr = state.to_array()
        assert arr.shape == (34,)
        assert arr.dtype == np.float32
    
    def test_from_array_roundtrip(self):
        """from_array should reconstruct state."""
        original = StateDefinition.zeros()
        original.velocity[0] = 50.0
        original.track_position = 0.5
        
        arr = original.to_array()
        reconstructed = StateDefinition.from_array(arr)
        
        assert np.allclose(reconstructed.velocity, original.velocity)
        assert reconstructed.track_position == original.track_position
    
    def test_from_array_invalid_shape(self):
        """from_array should reject wrong shape."""
        with pytest.raises(AssertionError):
            StateDefinition.from_array(np.zeros(10))
    
    def test_zeros_initialization(self):
        """zeros should create valid initial state."""
        state = StateDefinition.zeros()
        
        assert np.all(state.velocity == 0)
        assert np.all(state.tire_wear == 1.0)
        assert state.fuel_mass == 1.0
        assert state.track_position == 0.0


class TestActionBounds:
    
    def test_dimension(self):
        """Action dimension should be 3."""
        bounds = ActionBounds()
        assert bounds.dimension == 3
    
    def test_low_high_shapes(self):
        """low and high should have correct shapes."""
        bounds = ActionBounds()
        assert bounds.low.shape == (3,)
        assert bounds.high.shape == (3,)
    
    def test_steering_bounds(self):
        """Steering should be [-1, 1]."""
        bounds = ActionBounds()
        assert bounds.low[0] == -1.0
        assert bounds.high[0] == 1.0
    
    def test_throttle_bounds(self):
        """Throttle should be [0, 1]."""
        bounds = ActionBounds()
        assert bounds.low[1] == 0.0
        assert bounds.high[1] == 1.0
    
    def test_brake_bounds(self):
        """Brake should be [0, 1]."""
        bounds = ActionBounds()
        assert bounds.low[2] == 0.0
        assert bounds.high[2] == 1.0


class TestNormalizationParams:
    
    def test_immutable(self):
        """NormalizationParams should be immutable."""
        params = NormalizationParams()
        with pytest.raises(Exception):  # frozen dataclass
            params.velocity_max = 200.0
    
    def test_default_values(self):
        """Default values should be physically reasonable."""
        params = NormalizationParams()
        
        assert params.velocity_max == 100.0  # ~360 km/h
        assert params.acceleration_max == 50.0  # ~5g
        assert params.tire_temp_max == 120.0  # °C
