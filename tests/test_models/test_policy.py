# Tests for policy network

import pytest
import torch
from src.models import GaussianPolicy


class TestGaussianPolicy:
    
    @pytest.fixture
    def policy(self, state_dim, action_dim):
        return GaussianPolicy(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[64, 64],
        )
    
    def test_forward_shapes(self, policy, sample_batch_states, batch_size, action_dim):
        """Forward should return mean and log_std."""
        mean, log_std = policy(sample_batch_states)
        
        assert mean.shape == (batch_size, action_dim)
        assert log_std.shape == (batch_size, action_dim)
    
    def test_sample_shapes(self, policy, sample_batch_states, batch_size, action_dim):
        """Sample should return action and log_prob."""
        action, log_prob = policy.sample(sample_batch_states)
        
        assert action.shape == (batch_size, action_dim)
        assert log_prob.shape == (batch_size,)
    
    def test_sample_deterministic(self, policy, sample_batch_states, batch_size, action_dim):
        """Deterministic sample should return mean."""
        policy.eval()
        
        action1, _ = policy.sample(sample_batch_states, deterministic=True)
        action2, _ = policy.sample(sample_batch_states, deterministic=True)
        
        assert torch.allclose(action1, action2)
    
    def test_action_bounds(self, policy, sample_batch_states):
        """Actions should be bounded by tanh."""
        for _ in range(10):
            action, _ = policy.sample(sample_batch_states)
            assert (action >= -1).all()
            assert (action <= 1).all()
    
    def test_log_prob_shape(self, policy, sample_batch_states, sample_batch_actions, batch_size):
        """log_prob should return correct shape."""
        # Clamp actions to valid range for tanh
        actions = torch.clamp(sample_batch_actions, -0.99, 0.99)
        log_prob = policy.log_prob(sample_batch_states, actions)
        
        assert log_prob.shape == (batch_size,)
    
    def test_entropy_shape(self, policy, sample_batch_states, batch_size):
        """Entropy should return correct shape."""
        entropy = policy.entropy(sample_batch_states)
        
        assert entropy.shape == (batch_size,)
    
    def test_gradient_flow(self, policy, state_dim, action_dim):
        """Gradients should flow through policy."""
        state = torch.randn(8, state_dim, requires_grad=True)
        
        action, log_prob = policy.sample(state)
        loss = log_prob.sum()
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
    
    def test_log_std_bounds(self, policy, sample_batch_states):
        """Log std should be bounded."""
        _, log_std = policy(sample_batch_states)
        
        assert (log_std >= policy.log_std_min).all()
        assert (log_std <= policy.log_std_max).all()
    
    def test_stochastic_sampling(self, policy, sample_batch_states):
        """Stochastic sampling should produce different actions."""
        policy.train()
        
        action1, _ = policy.sample(sample_batch_states, deterministic=False)
        action2, _ = policy.sample(sample_batch_states, deterministic=False)
        
        # Actions should differ (with high probability)
        assert not torch.allclose(action1, action2)
