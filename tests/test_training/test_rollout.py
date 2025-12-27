# Tests for rollout collection

import pytest
import numpy as np
import torch
from src.training.rollout import RolloutBuffer


class TestRolloutBuffer:
    
    @pytest.fixture
    def buffer(self):
        num_envs = 4
        num_steps = 64
        obs_dim = 34
        action_dim = 3
        
        return RolloutBuffer(
            observations=np.random.randn(num_envs, num_steps, obs_dim).astype(np.float32),
            actions=np.random.randn(num_envs, num_steps, action_dim).astype(np.float32),
            rewards=np.random.randn(num_envs, num_steps).astype(np.float32),
            values=np.random.randn(num_envs, num_steps).astype(np.float32),
            log_probs=np.random.randn(num_envs, num_steps).astype(np.float32),
            dones=np.zeros((num_envs, num_steps), dtype=np.float32),
        )
    
    def test_compute_gae(self, buffer):
        """GAE computation should produce valid advantages."""
        last_values = np.random.randn(4).astype(np.float32)
        buffer.compute_gae(last_values)
        
        assert buffer.advantages is not None
        assert buffer.returns is not None
        assert buffer.advantages.shape == buffer.rewards.shape
        assert buffer.returns.shape == buffer.rewards.shape
    
    def test_gae_no_nan(self, buffer):
        """GAE should not produce NaN values."""
        last_values = np.random.randn(4).astype(np.float32)
        buffer.compute_gae(last_values)
        
        assert not np.isnan(buffer.advantages).any()
        assert not np.isnan(buffer.returns).any()
    
    def test_flatten(self, buffer):
        """Flatten should produce correct shapes."""
        last_values = np.random.randn(4).astype(np.float32)
        buffer.compute_gae(last_values)
        
        flat = buffer.flatten()
        
        expected_batch = 4 * 64  # num_envs * num_steps
        assert flat["observations"].shape == (expected_batch, 34)
        assert flat["actions"].shape == (expected_batch, 3)
        assert flat["log_probs"].shape == (expected_batch,)
        assert flat["advantages"].shape == (expected_batch,)
        assert flat["returns"].shape == (expected_batch,)
    
    def test_to_tensor(self, buffer, device):
        """to_tensor should convert to torch tensors."""
        last_values = np.random.randn(4).astype(np.float32)
        buffer.compute_gae(last_values)
        
        tensors = buffer.to_tensor(device)
        
        assert isinstance(tensors["observations"], torch.Tensor)
        assert tensors["observations"].device == device
        assert tensors["observations"].dtype == torch.float32
    
    def test_gae_with_dones(self):
        """GAE should handle episode boundaries correctly."""
        num_envs = 2
        num_steps = 10
        
        buffer = RolloutBuffer(
            observations=np.zeros((num_envs, num_steps, 34), dtype=np.float32),
            actions=np.zeros((num_envs, num_steps, 3), dtype=np.float32),
            rewards=np.ones((num_envs, num_steps), dtype=np.float32),
            values=np.zeros((num_envs, num_steps), dtype=np.float32),
            log_probs=np.zeros((num_envs, num_steps), dtype=np.float32),
            dones=np.zeros((num_envs, num_steps), dtype=np.float32),
        )
        
        # Mark episode end at step 5
        buffer.dones[:, 5] = 1.0
        
        last_values = np.zeros(num_envs, dtype=np.float32)
        buffer.compute_gae(last_values, gamma=0.99, gae_lambda=0.95)
        
        # Advantages should be computed correctly
        assert buffer.advantages is not None
        # After done, advantage should reset
        assert not np.allclose(buffer.advantages[:, 4], buffer.advantages[:, 6])
