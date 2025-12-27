# Pytest configuration and fixtures

import pytest
import numpy as np
import torch
from pathlib import Path
import tempfile
import yaml


@pytest.fixture
def seed():
    """Fixed seed for reproducibility."""
    return 42


@pytest.fixture
def set_seed(seed):
    """Set all random seeds."""
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)


@pytest.fixture
def device():
    """Get test device (CPU for CI)."""
    return torch.device("cpu")


@pytest.fixture
def state_dim():
    """Standard state dimension."""
    return 34


@pytest.fixture
def action_dim():
    """Standard action dimension."""
    return 3


@pytest.fixture
def batch_size():
    """Standard batch size for tests."""
    return 32


@pytest.fixture
def sample_state(state_dim, set_seed):
    """Generate sample state array."""
    return np.random.randn(state_dim).astype(np.float32)


@pytest.fixture
def sample_action(action_dim, set_seed):
    """Generate sample action array."""
    action = np.random.randn(action_dim).astype(np.float32)
    # Clip to valid range
    action[0] = np.clip(action[0], -1, 1)  # steering
    action[1] = np.clip(action[1], 0, 1)   # throttle
    action[2] = np.clip(action[2], 0, 1)   # brake
    return action


@pytest.fixture
def sample_batch_states(batch_size, state_dim, set_seed):
    """Generate batch of states."""
    return torch.randn(batch_size, state_dim)


@pytest.fixture
def sample_batch_actions(batch_size, action_dim, set_seed):
    """Generate batch of actions."""
    actions = torch.randn(batch_size, action_dim)
    actions[:, 0] = torch.clamp(actions[:, 0], -1, 1)
    actions[:, 1] = torch.clamp(actions[:, 1], 0, 1)
    actions[:, 2] = torch.clamp(actions[:, 2], 0, 1)
    return actions


@pytest.fixture
def config():
    """Standard test configuration."""
    return {
        "experiment": {
            "name": "test",
            "seed": 42,
            "deterministic": True,
        },
        "device": {
            "type": "cpu",
            "cuda_device": 0,
            "num_workers": 0,
            "pin_memory": False,
        },
        "env": {
            "name": "stub",
            "max_episode_steps": 100,
            "action_repeat": 1,
            "track": {
                "name": "test_track",
                "length_km": 5.0,
            },
            "normalize_obs": True,
            "normalize_reward": True,
            "clip_obs": 10.0,
            "clip_reward": 10.0,
        },
        "state": {
            "dimension": 34,
        },
        "action": {
            "dimension": 3,
        },
        "world_model": {
            "enabled": False,
            "architecture": "mlp",
            "encoder": {
                "hidden_dims": [64, 64],
                "latent_dim": 32,
                "activation": "elu",
            },
            "dynamics": {
                "hidden_dims": [64, 64],
                "activation": "elu",
            },
            "horizon": 8,
            "learning_rate": 0.001,
            "batch_size": 32,
            "gradient_clip": 1.0,
        },
        "policy": {
            "algorithm": "ppo",
            "actor": {
                "hidden_dims": [64, 64],
                "activation": "tanh",
                "log_std_min": -20.0,
                "log_std_max": 2.0,
            },
            "critic": {
                "hidden_dims": [64, 64],
                "activation": "relu",
            },
            "ppo": {
                "clip_ratio": 0.2,
                "value_clip": 0.2,
                "entropy_coef": 0.01,
                "value_coef": 0.5,
                "max_grad_norm": 0.5,
                "target_kl": 0.01,
            },
        },
        "training": {
            "total_timesteps": 1000,
            "rollout": {
                "num_envs": 2,
                "steps_per_env": 64,
            },
            "update": {
                "epochs": 2,
                "minibatch_size": 32,
            },
            "lr_schedule": {
                "type": "constant",
                "initial": 0.001,
                "final": 0.0001,
            },
            "eval": {
                "frequency": 500,
                "num_episodes": 2,
                "deterministic": True,
            },
        },
        "reward": {
            "progress_weight": 1.0,
            "speed_weight": 0.1,
            "off_track_penalty": -10.0,
            "collision_penalty": -100.0,
            "smoothness_penalty": 0.1,
            "scale": 1.0,
            "clip": 10.0,
        },
        "logging": {
            "level": "WARNING",
            "log_frequency": 100,
        },
    }


@pytest.fixture
def temp_dir():
    """Create temporary directory for test outputs."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


@pytest.fixture
def config_file(config, temp_dir):
    """Create temporary config file."""
    config_path = temp_dir / "config.yaml"
    with open(config_path, "w") as f:
        yaml.dump(config, f)
    return config_path
