# PHASE 4 — Simulator Abstraction (NO SIMULATION YET)

## Core Principle

The ML system does not know what simulator exists. It knows only the interface.

This is not abstraction for elegance. This is abstraction for survival.

## Interface Definition

```python
# src/env/interface.py
from abc import ABC, abstractmethod
from dataclasses import dataclass
import numpy as np

@dataclass
class StepResult:
    """Result of environment step."""
    observation: np.ndarray  # Shape: (obs_dim,)
    reward: float
    terminated: bool  # Episode ended (crash, finish)
    truncated: bool   # Episode cut off (max steps)
    info: dict        # Diagnostic information only

class RacingEnvInterface(ABC):
    """Abstract interface for racing environments.
    
    All racing environments (stub, Assetto Corsa, rFactor, custom)
    must implement this interface exactly.
    """
    
    @property
    @abstractmethod
    def observation_space(self) -> tuple[int, ...]:
        """Return observation shape."""
        pass
        
    @property
    @abstractmethod
    def action_space(self) -> tuple[int, ...]:
        """Return action shape."""
        pass
        
    @property
    @abstractmethod
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (low, high) action bounds."""
        pass
        
    @abstractmethod
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        """Reset environment to initial state.
        
        Args:
            seed: Random seed for reproducibility
            
        Returns:
            observation: Initial observation
            info: Diagnostic information
        """
        pass
        
    @abstractmethod
    def step(self, action: np.ndarray) -> StepResult:
        """Execute action in environment.
        
        Args:
            action: Action array, shape (action_dim,)
            
        Returns:
            StepResult with observation, reward, termination flags, info
        """
        pass
        
    @abstractmethod
    def close(self) -> None:
        """Clean up environment resources."""
        pass
        
    def render(self) -> None:
        """Optional rendering. Default is no-op."""
        pass
```

## Stub Environment (CPU Testing)

```python
# src/env/stub_env.py
import numpy as np
from .interface import RacingEnvInterface, StepResult

class StubRacingEnv(RacingEnvInterface):
    """Minimal racing environment for testing ML pipeline.
    
    This environment has NO physics. It exists to:
    1. Verify observation/action shapes
    2. Test training loop mechanics
    3. Validate reward computation
    4. Enable CI without simulator
    
    The dynamics are intentionally trivial:
    - Position advances based on throttle
    - Lateral position drifts based on steering
    - Episode ends when lap complete or off-track
    """
    
    def __init__(
        self,
        obs_dim: int = 34,
        action_dim: int = 3,
        max_steps: int = 2000,
        track_length: float = 5793.0,  # Monza in meters
    ):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self._max_steps = max_steps
        self._track_length = track_length
        
        # State
        self._position = 0.0  # Track position in meters
        self._lateral = 0.0   # Lateral offset from center
        self._velocity = 0.0  # Forward velocity m/s
        self._step_count = 0
        self._rng = np.random.default_rng()
        
    @property
    def observation_space(self) -> tuple[int, ...]:
        return (self._obs_dim,)
        
    @property
    def action_space(self) -> tuple[int, ...]:
        return (self._action_dim,)
        
    @property
    def action_bounds(self) -> tuple[np.ndarray, np.ndarray]:
        # [steering, throttle, brake]
        low = np.array([-1.0, 0.0, 0.0], dtype=np.float32)
        high = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        return low, high
        
    def reset(self, seed: int | None = None) -> tuple[np.ndarray, dict]:
        if seed is not None:
            self._rng = np.random.default_rng(seed)
            
        self._position = 0.0
        self._lateral = 0.0
        self._velocity = 50.0  # Start at 50 m/s (~180 km/h)
        self._step_count = 0
        
        obs = self._get_observation()
        info = {"lap_progress": 0.0}
        return obs, info
        
    def step(self, action: np.ndarray) -> StepResult:
        assert action.shape == (self._action_dim,), f"Expected shape {(self._action_dim,)}, got {action.shape}"
        
        steering, throttle, brake = action
        
        # Trivial dynamics (NOT physics)
        self._velocity += (throttle - brake) * 5.0  # Acceleration
        self._velocity = np.clip(self._velocity, 0.0, 100.0)  # Max 360 km/h
        self._position += self._velocity * 0.02  # 50Hz timestep
        self._lateral += steering * 0.5  # Lateral drift
        
        self._step_count += 1
        
        # Termination conditions
        lap_complete = self._position >= self._track_length
        off_track = abs(self._lateral) > 5.0  # 5m track width
        max_steps = self._step_count >= self._max_steps
        
        terminated = lap_complete or off_track
        truncated = max_steps and not terminated
        
        # Reward: progress minus penalties
        reward = self._velocity * 0.01  # Reward speed
        if off_track:
            reward = -10.0
        if lap_complete:
            reward = 100.0
            
        obs = self._get_observation()
        info = {
            "lap_progress": self._position / self._track_length,
            "velocity": self._velocity,
            "lateral_offset": self._lateral,
            "off_track": off_track,
            "lap_complete": lap_complete,
        }
        
        return StepResult(obs, reward, terminated, truncated, info)
        
    def _get_observation(self) -> np.ndarray:
        """Construct observation vector."""
        obs = np.zeros(self._obs_dim, dtype=np.float32)
        
        # Fill with meaningful values where possible
        obs[0:3] = [self._velocity, 0.0, 0.0]  # Velocity
        obs[9] = self._position / self._track_length  # Track position
        obs[10] = self._lateral  # Lateral offset
        
        # Rest is noise (simulates sensor noise)
        obs[11:] = self._rng.normal(0, 0.01, self._obs_dim - 11)
        
        return obs
        
    def close(self) -> None:
        pass  # No resources to clean up
```

## What Must Be Stubbed

| Component | Stub Behavior | Why Stub |
|-----------|--------------|----------|
| Physics engine | Linear velocity update | Physics is simulator's job |
| Tire model | Constant grip | Tire physics is complex |
| Aero model | Ignored | Aero is track-specific |
| Collision detection | Simple bounds check | Collision is simulator's job |
| Track geometry | Straight line | Track data comes from simulator |
| Rendering | No-op | Rendering is optional |
| Telemetry | Synthetic values | Real telemetry from simulator |

## What Must Be Postponed

| Component | When to Add | Why Postpone |
|-----------|-------------|--------------|
| Assetto Corsa integration | After GPU training works | Simulator adds complexity |
| Real track data | After stub training converges | Track data is large |
| Multi-car racing | After single-car works | Multi-agent is hard |
| Weather effects | After dry conditions work | Weather adds state dimensions |
| Pit strategy | After lap completion works | Pit adds discrete decisions |

## Why Placeholders Are Strategically Correct

1. **Validate pipeline first:** A training loop that crashes on stub will crash on simulator. Fix the easy bugs first.

2. **Isolate failures:** When training fails, is it the ML code or the simulator? With stub, you know it is ML code.

3. **Fast iteration:** Stub runs at 100,000 steps/second. Simulator runs at 1,000 steps/second. Debug on stub.

4. **CI compatibility:** CI has no simulator license. Stub enables automated testing.

5. **Reproducibility:** Stub is deterministic. Simulator has race conditions, network latency, frame drops.

## Gymnasium Wrapper

```python
# src/env/gym_wrapper.py
import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .interface import RacingEnvInterface

class GymWrapper(gym.Env):
    """Wrap RacingEnvInterface as Gymnasium environment.
    
    This enables use of standard RL libraries (SB3, CleanRL)
    while maintaining our interface abstraction.
    """
    
    def __init__(self, env: RacingEnvInterface):
        self._env = env
        
        obs_shape = env.observation_space
        act_shape = env.action_space
        low, high = env.action_bounds
        
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs_shape, dtype=np.float32
        )
        self.action_space = spaces.Box(
            low=low, high=high, shape=act_shape, dtype=np.float32
        )
        
    def reset(self, seed=None, options=None):
        obs, info = self._env.reset(seed=seed)
        return obs, info
        
    def step(self, action):
        result = self._env.step(action)
        return result.observation, result.reward, result.terminated, result.truncated, result.info
        
    def close(self):
        self._env.close()
```

## Environment Factory

```python
# src/env/__init__.py
from .interface import RacingEnvInterface, StepResult
from .stub_env import StubRacingEnv
from .gym_wrapper import GymWrapper

def make_env(config) -> RacingEnvInterface:
    """Factory function for creating environments.
    
    Args:
        config: Environment configuration
        
    Returns:
        Environment implementing RacingEnvInterface
    """
    if config.env.name == "stub":
        return StubRacingEnv(
            obs_dim=config.state.dimension,
            action_dim=config.action.dimension,
            max_steps=config.env.max_episode_steps,
        )
    elif config.env.name == "assetto":
        # POSTPONED: Assetto Corsa integration
        raise NotImplementedError("Assetto Corsa integration not yet implemented")
    elif config.env.name == "rfactor":
        # POSTPONED: rFactor integration
        raise NotImplementedError("rFactor integration not yet implemented")
    else:
        raise ValueError(f"Unknown environment: {config.env.name}")
```

## Strict Decoupling Rules

```python
# FORBIDDEN in src/models/:
from src.env import StubRacingEnv  # Models do not know about environments
from src.env.interface import RacingEnvInterface  # Not even the interface

# FORBIDDEN in src/env/:
import torch  # Environment is numpy-only
from src.models import Policy  # Environment does not know about models

# ALLOWED in src/training/:
from src.env import make_env, RacingEnvInterface  # Training orchestrates
from src.models import Policy, WorldModel  # Training uses models
```

The training module is the only place where environment and models meet.
