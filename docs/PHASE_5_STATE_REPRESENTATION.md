# PHASE 5 — State Representation & Telemetry

## Core Principle

State is the minimal sufficient representation for decision-making. Not maximal. Not convenient. Minimal and sufficient.

Every dimension costs memory, compute, and sample complexity. Justify each one.

## State Vector Definition

```python
# src/telemetry/state.py
from dataclasses import dataclass
import numpy as np

@dataclass
class StateDefinition:
    """F1 racing state vector specification.
    
    Total dimensions: 34
    All values normalized to approximately [-1, 1] or [0, 1]
    """
    
    # Vehicle dynamics (9 dimensions)
    velocity: np.ndarray          # [vx, vy, vz] m/s, normalized by 100
    acceleration: np.ndarray      # [ax, ay, az] m/s², normalized by 50
    angular_velocity: np.ndarray  # [wx, wy, wz] rad/s, normalized by 5
    
    # Control state (3 dimensions)
    steering_angle: float   # [-1, 1] normalized
    throttle: float         # [0, 1]
    brake: float            # [0, 1]
    
    # Tire state (8 dimensions)
    tire_temps: np.ndarray  # [FL, FR, RL, RR] °C, normalized by 100
    tire_wear: np.ndarray   # [FL, FR, RL, RR] [0, 1] remaining
    
    # Vehicle state (1 dimension)
    fuel_mass: float        # [0, 1] normalized by max fuel
    
    # Track-relative state (3 dimensions)
    track_position: float   # [0, 1] progress around lap
    lateral_offset: float   # meters from racing line, normalized by 5
    heading_error: float    # radians from racing line tangent, normalized by π
    
    # Lookahead (10 dimensions)
    curvature_ahead: np.ndarray  # Next 10 curvature samples, normalized
    
    @property
    def dimension(self) -> int:
        return 34
        
    def to_array(self) -> np.ndarray:
        """Flatten to numpy array."""
        return np.concatenate([
            self.velocity,
            self.acceleration,
            self.angular_velocity,
            [self.steering_angle, self.throttle, self.brake],
            self.tire_temps,
            self.tire_wear,
            [self.fuel_mass],
            [self.track_position, self.lateral_offset, self.heading_error],
            self.curvature_ahead,
        ]).astype(np.float32)
        
    @classmethod
    def from_array(cls, arr: np.ndarray) -> "StateDefinition":
        """Reconstruct from numpy array."""
        assert arr.shape == (34,), f"Expected shape (34,), got {arr.shape}"
        return cls(
            velocity=arr[0:3],
            acceleration=arr[3:6],
            angular_velocity=arr[6:9],
            steering_angle=arr[9],
            throttle=arr[10],
            brake=arr[11],
            tire_temps=arr[12:16],
            tire_wear=arr[16:20],
            fuel_mass=arr[20],
            track_position=arr[21],
            lateral_offset=arr[22],
            heading_error=arr[23],
            curvature_ahead=arr[24:34],
        )
```

## State vs Observation vs Latent

| Concept | Definition | Example | Where Used |
|---------|------------|---------|------------|
| State | Ground truth from simulator | Exact velocity, exact position | Simulator internal |
| Observation | What agent perceives | Noisy velocity, discretized position | Policy input |
| Latent | Compressed representation | 64-dim encoding | World model internal |

```
Simulator State (hidden) → Observation (34-dim) → Encoder → Latent (64-dim)
```

The agent never sees true state. It sees observations, which may be:
- Noisy (sensor noise)
- Delayed (processing latency)
- Partial (occluded information)

For initial development, observation equals state. Noise is added later.

## Forbidden State Components

| Component | Why Forbidden |
|-----------|---------------|
| Raw RGB images | 1920x1080x3 = 6.2M dimensions. Unlearnable without massive compute. |
| LIDAR point clouds | 100K+ points. Requires specialized architectures. |
| Full track mesh | Megabytes of geometry. Not needed for control. |
| Other car positions (raw) | Variable number of cars. Use relative encoding instead. |
| Historical frames | Increases state 10x. Use recurrent model instead. |
| Pit crew status | Discrete, sparse. Handle separately. |
| Weather radar | Image data. Use scalar summaries. |

## Dimensionality Limits

| Constraint | Limit | Reason |
|------------|-------|--------|
| State dimension | < 100 | Sample complexity scales with dimension |
| Latent dimension | < 128 | VRAM constraint for world model |
| Action dimension | < 10 | Continuous control curse of dimensionality |
| Lookahead samples | < 20 | Diminishing returns beyond ~1 second |

## Normalization Rules

```python
# src/telemetry/normalization.py
import numpy as np
from dataclasses import dataclass

@dataclass
class NormalizationParams:
    """Normalization parameters for state components."""
    
    # Physical limits for normalization
    VELOCITY_MAX = 100.0      # m/s (~360 km/h)
    ACCELERATION_MAX = 50.0   # m/s² (~5g)
    ANGULAR_VEL_MAX = 5.0     # rad/s
    TIRE_TEMP_MAX = 120.0     # °C
    LATERAL_MAX = 5.0         # meters
    CURVATURE_MAX = 0.1       # 1/m (10m radius minimum)
    
def normalize_state(state: np.ndarray) -> np.ndarray:
    """Normalize state to approximately [-1, 1] range.
    
    Normalization is critical for:
    1. Gradient stability (no exploding gradients from large values)
    2. Fair weighting (all features contribute equally initially)
    3. Activation function effectiveness (tanh, sigmoid work in [-1, 1])
    """
    params = NormalizationParams()
    normalized = state.copy()
    
    # Velocity: divide by max
    normalized[0:3] /= params.VELOCITY_MAX
    
    # Acceleration: divide by max
    normalized[3:6] /= params.ACCELERATION_MAX
    
    # Angular velocity: divide by max
    normalized[6:9] /= params.ANGULAR_VEL_MAX
    
    # Controls already in [0, 1] or [-1, 1]
    # normalized[9:12] unchanged
    
    # Tire temps: normalize to [0, 1]
    normalized[12:16] /= params.TIRE_TEMP_MAX
    
    # Tire wear already in [0, 1]
    # normalized[16:20] unchanged
    
    # Fuel already in [0, 1]
    # normalized[20] unchanged
    
    # Track position already in [0, 1]
    # normalized[21] unchanged
    
    # Lateral offset: divide by max
    normalized[22] /= params.LATERAL_MAX
    
    # Heading error: divide by π
    normalized[23] /= np.pi
    
    # Curvature: divide by max
    normalized[24:34] /= params.CURVATURE_MAX
    
    return normalized

def denormalize_state(normalized: np.ndarray) -> np.ndarray:
    """Reverse normalization for interpretation."""
    params = NormalizationParams()
    state = normalized.copy()
    
    state[0:3] *= params.VELOCITY_MAX
    state[3:6] *= params.ACCELERATION_MAX
    state[6:9] *= params.ANGULAR_VEL_MAX
    state[12:16] *= params.TIRE_TEMP_MAX
    state[22] *= params.LATERAL_MAX
    state[23] *= np.pi
    state[24:34] *= params.CURVATURE_MAX
    
    return state
```

## Why Telemetry Beats Pixels Initially

| Aspect | Telemetry (34-dim) | Pixels (6.2M-dim) |
|--------|-------------------|-------------------|
| Sample complexity | ~100K samples | ~10M samples |
| Training time | Hours | Days/weeks |
| Interpretability | Direct physical meaning | Black box |
| Debugging | Print values | Visualize activations |
| Sim-to-real gap | Calibrate sensors | Domain randomization |
| VRAM usage | Negligible | Dominates |

Pixels are useful for:
- Detecting other cars (vision)
- Reading track conditions (wet patches)
- Handling novel situations

Pixels are not useful for:
- Basic vehicle control
- Racing line following
- Tire management

Start with telemetry. Add vision later if needed.

## State Validation

```python
# src/telemetry/validation.py
import numpy as np

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
    def validate(cls, state: np.ndarray) -> tuple[bool, list[str]]:
        """Check state is within physical bounds.
        
        Returns:
            (is_valid, list of violations)
        """
        violations = []
        
        # Check velocity
        vel = state[0:3]
        if np.any(np.abs(vel) > 150.0):
            violations.append(f"Velocity out of bounds: {vel}")
            
        # Check acceleration
        acc = state[3:6]
        if np.any(np.abs(acc) > 100.0):
            violations.append(f"Acceleration out of bounds: {acc}")
            
        # Check for NaN/Inf
        if np.any(np.isnan(state)):
            violations.append("State contains NaN")
        if np.any(np.isinf(state)):
            violations.append("State contains Inf")
            
        return len(violations) == 0, violations
        
    @classmethod
    def clip(cls, state: np.ndarray) -> np.ndarray:
        """Clip state to physical bounds."""
        clipped = state.copy()
        
        # Clip each component
        clipped[0:3] = np.clip(clipped[0:3], -150.0, 150.0)
        clipped[3:6] = np.clip(clipped[3:6], -100.0, 100.0)
        clipped[6:9] = np.clip(clipped[6:9], -10.0, 10.0)
        clipped[9] = np.clip(clipped[9], -1.0, 1.0)
        clipped[10:12] = np.clip(clipped[10:12], 0.0, 1.0)
        clipped[12:16] = np.clip(clipped[12:16], 0.0, 200.0)
        clipped[16:21] = np.clip(clipped[16:21], 0.0, 1.0)
        clipped[22] = np.clip(clipped[22], -20.0, 20.0)
        clipped[23] = np.clip(clipped[23], -np.pi, np.pi)
        clipped[24:34] = np.clip(clipped[24:34], -0.5, 0.5)
        
        return clipped
```

## Observation Construction

```python
# src/telemetry/observation.py
import numpy as np
from .state import StateDefinition
from .normalization import normalize_state
from .validation import StateValidator

class ObservationBuilder:
    """Construct observations from raw telemetry."""
    
    def __init__(self, add_noise: bool = False, noise_std: float = 0.01):
        self.add_noise = add_noise
        self.noise_std = noise_std
        self._rng = np.random.default_rng()
        
    def build(self, raw_telemetry: dict) -> np.ndarray:
        """Convert raw telemetry dict to observation array.
        
        Args:
            raw_telemetry: Dict with keys matching StateDefinition fields
            
        Returns:
            Normalized observation array, shape (34,)
        """
        state = StateDefinition(
            velocity=np.array(raw_telemetry["velocity"]),
            acceleration=np.array(raw_telemetry["acceleration"]),
            angular_velocity=np.array(raw_telemetry["angular_velocity"]),
            steering_angle=raw_telemetry["steering"],
            throttle=raw_telemetry["throttle"],
            brake=raw_telemetry["brake"],
            tire_temps=np.array(raw_telemetry["tire_temps"]),
            tire_wear=np.array(raw_telemetry["tire_wear"]),
            fuel_mass=raw_telemetry["fuel"],
            track_position=raw_telemetry["track_position"],
            lateral_offset=raw_telemetry["lateral_offset"],
            heading_error=raw_telemetry["heading_error"],
            curvature_ahead=np.array(raw_telemetry["curvature_ahead"]),
        )
        
        # Convert to array
        obs = state.to_array()
        
        # Validate
        is_valid, violations = StateValidator.validate(obs)
        if not is_valid:
            obs = StateValidator.clip(obs)
            
        # Normalize
        obs = normalize_state(obs)
        
        # Add noise if enabled
        if self.add_noise:
            obs += self._rng.normal(0, self.noise_std, obs.shape)
            
        return obs.astype(np.float32)
```
