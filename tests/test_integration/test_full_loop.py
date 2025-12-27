# Integration tests for full training loop

import pytest
import torch
import numpy as np
from src.env import make_env, make_vec_env, GymWrapper
from src.models import GaussianPolicy, ValueFunction, WorldModel
from src.training.rollout import collect_rollout
from src.training.ppo import PPOUpdate


class TestFullLoop:
    """Integration tests for complete training pipeline."""
    
    def test_env_policy_interaction(self, config, device, state_dim, action_dim):
        """Test environment and policy can interact."""
        env = GymWrapper(make_env(config))
        policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        
        obs, _ = env.reset()
        
        for _ in range(10):
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.sample(obs_tensor)
            action_np = action.cpu().numpy().squeeze()
            
            obs, reward, terminated, truncated, info = env.step(action_np)
            
            if terminated or truncated:
                obs, _ = env.reset()
        
        env.close()
    
    def test_vectorized_env(self, config, device, state_dim, action_dim):
        """Test vectorized environment works."""
        num_envs = 4
        envs = make_vec_env(config, num_envs)
        policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        
        obs, _ = envs.reset()
        assert obs.shape == (num_envs, state_dim)
        
        for _ in range(10):
            obs_tensor = torch.tensor(obs, device=device)
            with torch.no_grad():
                action, _ = policy.sample(obs_tensor)
            action_np = action.cpu().numpy()
            
            obs, reward, terminated, truncated, info = envs.step(action_np)
            assert obs.shape == (num_envs, state_dim)
        
        envs.close()
    
    def test_rollout_collection(self, config, device, state_dim, action_dim):
        """Test rollout collection works."""
        num_envs = 2
        num_steps = 32
        
        envs = make_vec_env(config, num_envs)
        policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        value_fn = ValueFunction(state_dim=state_dim).to(device)
        
        buffer = collect_rollout(
            envs=envs,
            policy=policy,
            value_fn=value_fn,
            num_steps=num_steps,
            device=device,
        )
        
        assert buffer.observations.shape == (num_envs, num_steps, state_dim)
        assert buffer.actions.shape == (num_envs, num_steps, action_dim)
        assert buffer.rewards.shape == (num_envs, num_steps)
        assert buffer.advantages is not None
        assert buffer.returns is not None
        
        envs.close()
    
    def test_ppo_update(self, config, device, state_dim, action_dim):
        """Test PPO update works."""
        num_envs = 2
        num_steps = 32
        
        envs = make_vec_env(config, num_envs)
        policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        value_fn = ValueFunction(state_dim=state_dim).to(device)
        
        ppo = PPOUpdate(
            policy=policy,
            value_fn=value_fn,
            lr=1e-3,
        )
        
        # Collect rollout
        buffer = collect_rollout(
            envs=envs,
            policy=policy,
            value_fn=value_fn,
            num_steps=num_steps,
            device=device,
        )
        
        # Convert to tensors
        buffer_tensors = buffer.to_tensor(device)
        
        # Run update
        metrics = ppo.update(
            buffer=buffer_tensors,
            num_epochs=2,
            minibatch_size=16,
        )
        
        assert "policy_loss" in metrics
        assert "value_loss" in metrics
        assert "entropy" in metrics
        assert not np.isnan(metrics["policy_loss"])
        assert not np.isnan(metrics["value_loss"])
        
        envs.close()
    
    def test_world_model_training(self, config, device, state_dim, action_dim):
        """Test world model can be trained."""
        world_model = WorldModel(
            state_dim=state_dim,
            action_dim=action_dim,
            hidden_dims=[64, 64],
            latent_dim=32,
        ).to(device)
        
        optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
        
        # Generate fake data
        batch_size = 32
        states = torch.randn(batch_size, state_dim, device=device)
        actions = torch.randn(batch_size, action_dim, device=device)
        next_states = torch.randn(batch_size, state_dim, device=device)
        rewards = torch.randn(batch_size, device=device)
        dones = torch.zeros(batch_size, device=device)
        
        # Training step
        pred_next, pred_reward, pred_done = world_model(states, actions)
        
        loss = (
            torch.nn.functional.mse_loss(pred_next, next_states)
            + torch.nn.functional.mse_loss(pred_reward.squeeze(), rewards)
            + torch.nn.functional.binary_cross_entropy(pred_done.squeeze(), dones)
        )
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        assert not torch.isnan(loss)
    
    def test_checkpoint_save_load(self, config, device, state_dim, action_dim, temp_dir):
        """Test checkpoint save and load."""
        from src.analysis.checkpointing import save_checkpoint, load_checkpoint
        
        policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        value_fn = ValueFunction(state_dim=state_dim).to(device)
        optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_fn.parameters()),
            lr=1e-3,
        )
        
        # Save checkpoint
        checkpoint_path = temp_dir / "test_checkpoint.pt"
        save_checkpoint(
            path=checkpoint_path,
            step=1000,
            policy_state=policy.state_dict(),
            value_state=value_fn.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
        )
        
        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, device)
        
        assert checkpoint["step"] == 1000
        assert "policy_state_dict" in checkpoint
        assert "value_state_dict" in checkpoint
        
        # Load into new models
        new_policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
        new_policy.load_state_dict(checkpoint["policy_state_dict"])
        
        # Verify weights match
        for p1, p2 in zip(policy.parameters(), new_policy.parameters()):
            assert torch.allclose(p1, p2)
    
    def test_deterministic_with_seed(self, config, device, state_dim, action_dim):
        """Test training is deterministic with fixed seed."""
        def run_training(seed):
            torch.manual_seed(seed)
            np.random.seed(seed)
            
            envs = make_vec_env(config, num_envs=2)
            policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim).to(device)
            value_fn = ValueFunction(state_dim=state_dim).to(device)
            
            buffer = collect_rollout(
                envs=envs,
                policy=policy,
                value_fn=value_fn,
                num_steps=32,
                device=device,
            )
            
            envs.close()
            return buffer.rewards.sum()
        
        result1 = run_training(42)
        result2 = run_training(42)
        
        assert np.isclose(result1, result2)
