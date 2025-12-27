# PHASE 3 — Configuration-First Design

## Core Principle

Configuration controls everything. Code is generic. Behavior is specified externally.

If you must edit Python code to change a hyperparameter, the design is wrong.

## Configuration Schema

```yaml
# configs/base.yaml

# Experiment metadata
experiment:
  name: "baseline"
  seed: 42
  deterministic: true  # torch.use_deterministic_algorithms

# Hardware routing
device:
  type: "cpu"  # "cpu" or "cuda"
  cuda_device: 0  # GPU index if type=cuda
  num_workers: 4  # DataLoader workers
  pin_memory: false  # Only true for GPU

# Environment
env:
  name: "stub"  # "stub", "assetto", "rfactor"
  max_episode_steps: 2000
  action_repeat: 4  # Physics steps per action
  
  # Track specification
  track:
    name: "monza"
    length_km: 5.793
    
  # Normalization
  normalize_obs: true
  normalize_reward: true
  clip_obs: 10.0
  clip_reward: 10.0

# State representation
state:
  # Included state components
  include:
    - velocity  # 3D velocity vector
    - acceleration  # 3D acceleration
    - angular_velocity  # 3D angular velocity
    - steering_angle  # Current steering
    - throttle  # Current throttle
    - brake  # Current brake
    - tire_temps  # 4 tire temperatures
    - tire_wear  # 4 tire wear percentages
    - fuel_mass  # Remaining fuel
    - track_position  # Progress around track [0, 1]
    - lateral_offset  # Distance from racing line
    - heading_error  # Angle to racing line tangent
    - curvature_ahead  # Next 10 curvature samples
    
  # Dimensions (computed from include list)
  # velocity: 3, acceleration: 3, angular_velocity: 3
  # steering: 1, throttle: 1, brake: 1
  # tire_temps: 4, tire_wear: 4, fuel: 1
  # track_position: 1, lateral_offset: 1, heading_error: 1
  # curvature_ahead: 10
  # Total: 34 dimensions

# Action space
action:
  type: "continuous"
  
  # Action components
  steering:
    min: -1.0
    max: 1.0
  throttle:
    min: 0.0
    max: 1.0
  brake:
    min: 0.0
    max: 1.0
    
  # Action dimension: 3

# World model
world_model:
  enabled: true
  architecture: "mlp"  # "mlp", "gru", "ssm" (NO transformers)
  
  # Network architecture
  encoder:
    hidden_dims: [256, 256]
    latent_dim: 64
    activation: "elu"
    
  # Dynamics model
  dynamics:
    hidden_dims: [256, 256]
    activation: "elu"
    
  # Prediction horizon
  horizon: 16  # Steps to predict
  
  # Training
  learning_rate: 3.0e-4
  batch_size: 256
  gradient_clip: 1.0

# Policy (Actor-Critic)
policy:
  algorithm: "ppo"  # "ppo" or "sac"
  
  # Network architecture
  actor:
    hidden_dims: [256, 256]
    activation: "tanh"
    log_std_min: -20.0
    log_std_max: 2.0
    
  critic:
    hidden_dims: [256, 256]
    activation: "relu"
    
  # PPO-specific
  ppo:
    clip_ratio: 0.2
    value_clip: 0.2
    entropy_coef: 0.01
    value_coef: 0.5
    max_grad_norm: 0.5
    target_kl: 0.01  # Early stopping threshold
    
  # SAC-specific (if algorithm=sac)
  sac:
    tau: 0.005
    alpha: 0.2
    auto_alpha: true
    target_entropy: -3.0  # -dim(action)

# Training loop
training:
  total_timesteps: 1_000_000
  
  # Rollout
  rollout:
    num_envs: 8  # Parallel environments
    steps_per_env: 2048  # Steps before update
    
  # Updates
  update:
    epochs: 10  # PPO epochs per rollout
    minibatch_size: 64
    
  # Learning rate schedule
  lr_schedule:
    type: "linear"  # "constant", "linear", "cosine"
    initial: 3.0e-4
    final: 1.0e-5
    
  # Evaluation
  eval:
    frequency: 10_000  # Steps between evaluations
    num_episodes: 5
    deterministic: true

# Reward shaping
reward:
  # Primary objective
  lap_time_weight: 1.0
  
  # Safety penalties
  off_track_penalty: -10.0
  collision_penalty: -100.0
  
  # Smoothness rewards
  steering_smoothness_weight: 0.1
  throttle_smoothness_weight: 0.1
  
  # Long-horizon objectives
  tire_preservation_weight: 0.05
  fuel_efficiency_weight: 0.01
  
  # Reward scaling
  scale: 1.0
  clip: 10.0

# Logging
logging:
  level: "INFO"  # DEBUG, INFO, WARNING, ERROR
  
  # What to log
  log_frequency: 100  # Steps between logs
  
  metrics:
    - episode_return
    - episode_length
    - lap_time
    - policy_loss
    - value_loss
    - entropy
    - kl_divergence
    - explained_variance
    
  # TensorBoard
  tensorboard:
    enabled: true
    log_histograms: false  # Expensive, disable by default
    
  # Checkpointing
  checkpoint:
    frequency: 50_000  # Steps between checkpoints
    keep_last: 5
    save_best: true

# Explainability
analysis:
  # Decision tracing
  trace_decisions: true
  trace_frequency: 1000  # Steps between traces
  
  # What to trace
  trace_components:
    - action_distribution
    - value_estimate
    - state_encoding
    - world_model_prediction
```

## What Must Never Be Hardcoded

| Parameter | Why |
|-----------|-----|
| Learning rate | Tuned per experiment |
| Batch size | Hardware dependent |
| Network dimensions | Architecture search |
| Reward weights | Domain tuning |
| Clip values | Stability tuning |
| Seeds | Reproducibility |
| Paths | Machine dependent |
| Device | Hardware dependent |
| Logging level | Debug vs production |

## CPU to GPU Switch

Single change in config:

```yaml
# CPU
device:
  type: "cpu"
  pin_memory: false

# GPU
device:
  type: "cuda"
  cuda_device: 0
  pin_memory: true
```

Code handles this automatically:

```python
# In training/trainer.py
device = torch.device(config.device.type)
if config.device.type == "cuda":
    device = torch.device(f"cuda:{config.device.cuda_device}")
    
model = model.to(device)
```

## Seed Handling

```python
# In scripts/train.py (entry point only)
def set_global_seed(seed: int, deterministic: bool = False):
    """Set all random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
    # Return seed for logging
    return seed
```

Seed is set once at startup, never again.

## Config Validation

```python
# In src/core/config.py
from jsonschema import validate, ValidationError

def load_config(path: Path) -> Config:
    """Load and validate configuration."""
    with open(path) as f:
        raw = yaml.safe_load(f)
        
    # Validate against schema
    schema_path = Path(__file__).parent.parent.parent / "configs/schemas/config_schema.json"
    with open(schema_path) as f:
        schema = json.load(f)
        
    try:
        validate(raw, schema)
    except ValidationError as e:
        raise ConfigError(f"Invalid config: {e.message}") from e
        
    # Convert to typed dataclass
    return Config.from_dict(raw)
```

## Config Immutability

```python
@dataclass(frozen=True)  # frozen=True makes it immutable
class Config:
    experiment: ExperimentConfig
    device: DeviceConfig
    env: EnvConfig
    # ...
    
    def to_dict(self) -> dict:
        """Serialize for saving."""
        return asdict(self)
        
    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        """Deserialize from dict."""
        return cls(
            experiment=ExperimentConfig(**d["experiment"]),
            device=DeviceConfig(**d["device"]),
            # ...
        )
```

Config cannot be modified after creation. This prevents accidental mutation during training.

## Experiment Override Pattern

```bash
# Base config with experiment-specific overrides
python scripts/train.py \
    --config configs/base.yaml \
    --override experiment.name=high_lr \
    --override training.lr_schedule.initial=1e-3
```

```python
# In scripts/train.py
def apply_overrides(config: dict, overrides: list[str]) -> dict:
    """Apply command-line overrides to config."""
    for override in overrides:
        key, value = override.split("=")
        keys = key.split(".")
        
        # Navigate to nested key
        d = config
        for k in keys[:-1]:
            d = d[k]
            
        # Set value with type inference
        d[keys[-1]] = infer_type(value)
        
    return config
```
