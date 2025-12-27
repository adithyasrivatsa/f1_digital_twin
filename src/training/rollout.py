# Rollout collection
# Collects experience from environments

import numpy as np
import torch
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RolloutBuffer:
    """Storage for rollout data.
    
    Stores transitions from parallel environments.
    Data stays on CPU until needed for update.
    """
    observations: np.ndarray   # (num_envs, steps, obs_dim)
    actions: np.ndarray        # (num_envs, steps, action_dim)
    rewards: np.ndarray        # (num_envs, steps)
    values: np.ndarray         # (num_envs, steps)
    log_probs: np.ndarray      # (num_envs, steps)
    dones: np.ndarray          # (num_envs, steps)
    
    # Computed after rollout
    advantages: Optional[np.ndarray] = None
    returns: Optional[np.ndarray] = None
    
    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ) -> None:
        """Compute Generalized Advantage Estimation.
        
        GAE balances bias-variance tradeoff in advantage estimation.
        lambda=0: high bias, low variance (TD(0))
        lambda=1: low bias, high variance (Monte Carlo)
        lambda=0.95: good balance for most tasks
        
        Args:
            last_values: Value estimates for final states
            gamma: Discount factor
            gae_lambda: GAE lambda parameter
        """
        num_envs, num_steps = self.rewards.shape
        
        self.advantages = np.zeros_like(self.rewards)
        self.returns = np.zeros_like(self.rewards)
        
        last_gae = np.zeros(num_envs)
        
        for t in reversed(range(num_steps)):
            if t == num_steps - 1:
                next_values = last_values
                next_non_terminal = 1.0 - self.dones[:, t]
            else:
                next_values = self.values[:, t + 1]
                next_non_terminal = 1.0 - self.dones[:, t]
            
            delta = (
                self.rewards[:, t]
                + gamma * next_values * next_non_terminal
                - self.values[:, t]
            )
            
            last_gae = delta + gamma * gae_lambda * next_non_terminal * last_gae
            self.advantages[:, t] = last_gae
        
        self.returns = self.advantages + self.values
    
    def flatten(self) -> Dict[str, np.ndarray]:
        """Flatten for minibatch sampling.
        
        Returns:
            Dict with flattened arrays
        """
        batch_size = self.observations.shape[0] * self.observations.shape[1]
        
        return {
            "observations": self.observations.reshape(batch_size, -1),
            "actions": self.actions.reshape(batch_size, -1),
            "log_probs": self.log_probs.reshape(batch_size),
            "advantages": self.advantages.reshape(batch_size) if self.advantages is not None else None,
            "returns": self.returns.reshape(batch_size) if self.returns is not None else None,
            "values": self.values.reshape(batch_size),
        }
    
    def to_tensor(self, device: torch.device) -> Dict[str, torch.Tensor]:
        """Convert flattened buffer to tensors on device.
        
        Args:
            device: Target device
            
        Returns:
            Dict with tensors
        """
        flat = self.flatten()
        return {
            k: torch.tensor(v, device=device, dtype=torch.float32)
            for k, v in flat.items()
            if v is not None
        }


def collect_rollout(
    envs,  # Vectorized environments
    policy: torch.nn.Module,
    value_fn: torch.nn.Module,
    num_steps: int,
    device: torch.device,
) -> RolloutBuffer:
    """Collect rollout from parallel environments.
    
    Args:
        envs: Vectorized environment (num_envs parallel)
        policy: Policy network
        value_fn: Value network
        num_steps: Steps per environment
        device: Torch device
        
    Returns:
        RolloutBuffer with collected data
    """
    num_envs = envs.num_envs
    obs_dim = envs.single_observation_space.shape[0]
    action_dim = envs.single_action_space.shape[0]
    
    # Preallocate buffers on CPU
    observations = np.zeros((num_envs, num_steps, obs_dim), dtype=np.float32)
    actions = np.zeros((num_envs, num_steps, action_dim), dtype=np.float32)
    rewards = np.zeros((num_envs, num_steps), dtype=np.float32)
    values = np.zeros((num_envs, num_steps), dtype=np.float32)
    log_probs = np.zeros((num_envs, num_steps), dtype=np.float32)
    dones = np.zeros((num_envs, num_steps), dtype=np.float32)
    
    obs, _ = envs.reset()
    
    for step in range(num_steps):
        observations[:, step] = obs
        
        # Policy inference on device
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
            action, log_prob = policy.sample(obs_tensor)
            value = value_fn(obs_tensor)
        
        # Move to CPU immediately
        action_np = action.cpu().numpy()
        log_prob_np = log_prob.cpu().numpy()
        value_np = value.cpu().numpy().squeeze(-1)
        
        # Environment step (CPU)
        next_obs, reward, terminated, truncated, info = envs.step(action_np)
        done = np.logical_or(terminated, truncated)
        
        actions[:, step] = action_np
        rewards[:, step] = reward
        values[:, step] = value_np
        log_probs[:, step] = log_prob_np
        dones[:, step] = done.astype(np.float32)
        
        obs = next_obs
    
    # Get final values for GAE
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, device=device, dtype=torch.float32)
        last_values = value_fn(obs_tensor).cpu().numpy().squeeze(-1)
    
    buffer = RolloutBuffer(
        observations=observations,
        actions=actions,
        rewards=rewards,
        values=values,
        log_probs=log_probs,
        dones=dones,
    )
    buffer.compute_gae(last_values)
    
    return buffer
