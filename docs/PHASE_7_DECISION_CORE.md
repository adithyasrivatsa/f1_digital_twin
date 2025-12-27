# PHASE 7 — Decision Core (Where Intelligence Lives)

## Why Pure Prediction Is Insufficient

A world model predicts: "If I do X, Y will happen."

This is necessary but not sufficient for racing. The agent must also:
1. Evaluate outcomes: "Is Y good or bad?"
2. Compare alternatives: "Is Y better than Z?"
3. Plan ahead: "What sequence of actions leads to best outcome?"
4. Handle uncertainty: "What if prediction is wrong?"

Prediction without evaluation is a physics simulator. We already have one.

## Control + RL Interaction

```
                    ┌─────────────────┐
                    │   World Model   │
                    │  (Prediction)   │
                    └────────┬────────┘
                             │ predicted states
                             ▼
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│   Policy    │────▶│  Value Function │────▶│   Action    │
│   (Actor)   │     │    (Critic)     │     │  Selection  │
└─────────────┘     └─────────────────┘     └─────────────┘
       ▲                     ▲                     │
       │                     │                     │
       └─────────────────────┴─────────────────────┘
                    TD / Policy Gradient
```

The policy proposes actions. The value function evaluates them. The world model enables imagination-based evaluation.

## Long-Horizon Objectives

### Lap Time Consistency

```python
# Not just "minimize lap time" but "minimize variance of lap times"
def lap_consistency_reward(lap_times: list[float]) -> float:
    """Reward consistent lap times.
    
    Why: A driver who does 1:20, 1:25, 1:18 is worse than
    one who does 1:21, 1:21, 1:21 for race strategy.
    """
    if len(lap_times) < 2:
        return 0.0
    mean_time = np.mean(lap_times)
    std_time = np.std(lap_times)
    # Penalize variance relative to mean
    return -std_time / mean_time
```

### Tire Preservation

```python
def tire_preservation_reward(
    tire_wear: np.ndarray,
    lap_progress: float,
    target_stint_laps: int = 20,
) -> float:
    """Reward preserving tires for target stint length.
    
    Why: Aggressive driving destroys tires. A 1:19 lap that
    costs 5% tire wear is worse than 1:20 that costs 2%.
    """
    # Expected wear at this point in stint
    expected_wear = lap_progress / target_stint_laps
    actual_wear = 1.0 - tire_wear.mean()
    
    # Penalize wearing tires faster than expected
    excess_wear = max(0, actual_wear - expected_wear)
    return -excess_wear * 10.0
```

### Fuel Efficiency

```python
def fuel_efficiency_reward(
    fuel_used: float,
    distance_covered: float,
) -> float:
    """Reward fuel-efficient driving.
    
    Why: Running out of fuel = DNF. Lifting and coasting
    in non-critical zones saves fuel for overtakes.
    """
    if distance_covered < 1.0:
        return 0.0
    consumption_rate = fuel_used / distance_covered
    target_rate = 1.0 / 305.0  # ~305 km on full tank
    
    # Penalize excess consumption
    excess = max(0, consumption_rate - target_rate)
    return -excess * 100.0
```

## PPO vs SAC Choice

### PPO (Proximal Policy Optimization)

Advantages:
- Stable training (clipped objective)
- Works well with continuous actions
- Sample efficient for on-policy
- Easier to tune

Disadvantages:
- On-policy (cannot reuse old data)
- Requires many parallel environments

### SAC (Soft Actor-Critic)

Advantages:
- Off-policy (reuses replay buffer)
- Maximum entropy encourages exploration
- Often better final performance

Disadvantages:
- More hyperparameters
- Can be unstable early in training
- Entropy tuning is tricky

### Decision: Start with PPO

Rationale:
1. Stability is more important than sample efficiency initially
2. Debugging is easier with on-policy
3. PPO failure modes are well-understood
4. Can switch to SAC later if needed

## Action Space Design

```python
# src/models/policy.py
from dataclasses import dataclass
import torch
import torch.nn as nn
from torch.distributions import Normal

@dataclass
class ActionSpace:
    """Continuous action space for F1 control.
    
    Actions:
    - steering: [-1, 1] full left to full right
    - throttle: [0, 1] no throttle to full throttle
    - brake: [0, 1] no brake to full brake
    
    Note: throttle and brake are NOT mutually exclusive.
    Real drivers sometimes trail-brake (brake + partial throttle).
    """
    steering_range: tuple[float, float] = (-1.0, 1.0)
    throttle_range: tuple[float, float] = (0.0, 1.0)
    brake_range: tuple[float, float] = (0.0, 1.0)
    
    @property
    def dimension(self) -> int:
        return 3
        
    @property
    def low(self) -> torch.Tensor:
        return torch.tensor([-1.0, 0.0, 0.0])
        
    @property
    def high(self) -> torch.Tensor:
        return torch.tensor([1.0, 1.0, 1.0])


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous control.
    
    Outputs mean and log_std for each action dimension.
    Actions sampled from Normal(mean, exp(log_std)).
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 3,
        hidden_dims: list[int] = [256, 256],
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.Tanh(),  # Tanh for policy (bounded activations)
            ])
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        
        # Mean head
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Log std head (state-dependent)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
                
    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters.
        
        Args:
            state: Observation, shape (batch, state_dim)
            
        Returns:
            mean: Action mean, shape (batch, action_dim)
            log_std: Action log std, shape (batch, action_dim)
        """
        features = self.features(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
        
    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Observation
            deterministic: If True, return mean (no sampling)
            
        Returns:
            action: Sampled action, shape (batch, action_dim)
            log_prob: Log probability of action, shape (batch,)
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            action = mean
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            action = dist.rsample()  # Reparameterized sample
            log_prob = dist.log_prob(action).sum(dim=-1)
            
        # Squash to action bounds
        action = torch.tanh(action)
        
        # Correct log_prob for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return action, log_prob
        
    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of action.
        
        Args:
            state: Observation
            action: Action (already squashed)
            
        Returns:
            log_prob: Log probability, shape (batch,)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Unsquash action for log_prob computation
        action_unsquashed = torch.atanh(action.clamp(-0.999, 0.999))
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(action_unsquashed).sum(dim=-1)
        
        # Correct for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return log_prob
```

## Value Function

```python
# src/models/value.py
import torch
import torch.nn as nn

class ValueFunction(nn.Module):
    """State value function V(s).
    
    Estimates expected return from state s.
    Used for advantage estimation in PPO.
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        hidden_dims: list[int] = [256, 256],
    ):
        super().__init__()
        
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                nn.ReLU(),  # ReLU for value (unbounded output)
            ])
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
        
    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
                
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value.
        
        Args:
            state: Observation, shape (batch, state_dim)
            
        Returns:
            value: Estimated value, shape (batch, 1)
        """
        return self.network(state)
```

## CPU vs GPU Responsibility Split

| Component | CPU | GPU |
|-----------|-----|-----|
| Environment step | Yes | No |
| Observation preprocessing | Yes | No |
| Action postprocessing | Yes | No |
| Policy forward pass | No | Yes |
| Value forward pass | No | Yes |
| World model forward | No | Yes |
| Gradient computation | No | Yes |
| Replay buffer storage | Yes | No |
| Logging | Yes | No |

```python
# In training loop
def collect_rollout(env, policy, device, num_steps):
    """Collect experience with minimal GPU usage."""
    observations = []
    actions = []
    rewards = []
    
    obs, _ = env.reset()
    
    for _ in range(num_steps):
        # Move to GPU only for inference
        obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
        
        with torch.no_grad():
            action, _ = policy.sample(obs_tensor)
            
        # Move back to CPU immediately
        action_np = action.cpu().numpy().squeeze()
        
        # Environment step on CPU
        result = env.step(action_np)
        
        observations.append(obs)
        actions.append(action_np)
        rewards.append(result.reward)
        
        obs = result.observation
        
    return observations, actions, rewards
```

Data lives on CPU. Computation happens on GPU. Transfer is minimized.
