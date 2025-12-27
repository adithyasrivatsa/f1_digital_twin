# Tests for world model

import pytest
import torch
from src.models import WorldModel


class TestWorldModel:
    
    @pytest.fixture
    def model(self, state_dim, action_dim):
        return WorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[64, 64],
            latent_dim=32,
        )
    
    def test_forward_shapes(self, model, sample_batch_states, sample_batch_actions, batch_size, state_dim):
        """Forward pass should return correct shapes."""
        next_state, reward, done = model(sample_batch_states, sample_batch_actions)
        
        assert next_state.shape == (batch_size, state_dim)
        assert reward.shape == (batch_size, 1)
        assert done.shape == (batch_size, 1)
    
    def test_rollout_shapes(self, model, state_dim, action_dim, batch_size):
        """Rollout should return correct shapes."""
        horizon = 10
        initial_state = torch.randn(batch_size, state_dim)
        actions = torch.randn(batch_size, horizon, action_dim)
        
        states, rewards, dones = model.rollout(initial_state, actions)
        
        assert states.shape == (batch_size, horizon + 1, state_dim)
        assert rewards.shape == (batch_size, horizon)
        assert dones.shape == (batch_size, horizon)
    
    def test_deterministic(self, model, sample_batch_states, sample_batch_actions):
        """Model should be deterministic."""
        model.eval()
        
        out1 = model(sample_batch_states, sample_batch_actions)
        out2 = model(sample_batch_states, sample_batch_actions)
        
        assert torch.allclose(out1[0], out2[0])
        assert torch.allclose(out1[1], out2[1])
        assert torch.allclose(out1[2], out2[2])
    
    def test_gradient_flow(self, model, state_dim, action_dim):
        """Gradients should flow through model."""
        state = torch.randn(8, state_dim, requires_grad=True)
        action = torch.randn(8, action_dim)
        
        next_state, reward, done = model(state, action)
        loss = next_state.sum() + reward.sum() + done.sum()
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
    
    def test_parameter_count(self, model):
        """Model should have reasonable parameter count."""
        params = sum(p.numel() for p in model.parameters())
        # Should be < 1M parameters for small MLP
        assert params < 1_000_000
    
    def test_encode_decode(self, model, sample_batch_states, state_dim, batch_size):
        """Encode and decode should preserve shape."""
        latent = model.encode(sample_batch_states)
        assert latent.shape == (batch_size, model.latent_dim)
        
        decoded = model.decode(latent)
        assert decoded.shape == (batch_size, state_dim)
    
    def test_done_probability_range(self, model, sample_batch_states, sample_batch_actions):
        """Done probability should be in [0, 1]."""
        _, _, done = model(sample_batch_states, sample_batch_actions)
        
        assert (done >= 0).all()
        assert (done <= 1).all()
