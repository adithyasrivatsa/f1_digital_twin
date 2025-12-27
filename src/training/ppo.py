# PPO algorithm implementation

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict
import numpy as np


class PPOUpdate:
    """Proximal Policy Optimization update.
    
    Implements clipped surrogate objective with value function clipping.
    """
    
    def __init__(
        self,
        policy: nn.Module,
        value_fn: nn.Module,
        lr: float = 3e-4,
        clip_ratio: float = 0.2,
        value_clip: float = 0.2,
        entropy_coef: float = 0.01,
        value_coef: float = 0.5,
        max_grad_norm: float = 0.5,
        target_kl: float = 0.01,
    ):
        """Initialize PPO updater.
        
        Args:
            policy: Policy network
            value_fn: Value network
            lr: Learning rate
            clip_ratio: PPO clip ratio
            value_clip: Value function clip range
            entropy_coef: Entropy bonus coefficient
            value_coef: Value loss coefficient
            max_grad_norm: Maximum gradient norm
            target_kl: Target KL divergence for early stopping
        """
        self.policy = policy
        self.value_fn = value_fn
        self.clip_ratio = clip_ratio
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
        # Single optimizer for both networks
        self.optimizer = torch.optim.Adam(
            list(policy.parameters()) + list(value_fn.parameters()),
            lr=lr,
        )
    
    def update(
        self,
        buffer: Dict[str, torch.Tensor],
        num_epochs: int = 10,
        minibatch_size: int = 64,
    ) -> Dict[str, float]:
        """Perform PPO update.
        
        Args:
            buffer: Dict with observations, actions, log_probs, advantages, returns, values
            num_epochs: Number of passes over data
            minibatch_size: Minibatch size
            
        Returns:
            Dict of metrics
        """
        obs = buffer["observations"]
        actions = buffer["actions"]
        old_log_probs = buffer["log_probs"]
        advantages = buffer["advantages"]
        returns = buffer["returns"]
        old_values = buffer["values"]
        
        # Normalize advantages
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        batch_size = obs.shape[0]
        metrics = {
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "entropy": 0.0,
            "kl": 0.0,
            "clip_fraction": 0.0,
        }
        num_updates = 0
        early_stop = False
        
        for epoch in range(num_epochs):
            if early_stop:
                break
            
            # Shuffle indices
            indices = torch.randperm(batch_size, device=obs.device)
            
            for start in range(0, batch_size, minibatch_size):
                end = min(start + minibatch_size, batch_size)
                mb_indices = indices[start:end]
                
                mb_obs = obs[mb_indices]
                mb_actions = actions[mb_indices]
                mb_old_log_probs = old_log_probs[mb_indices]
                mb_advantages = advantages[mb_indices]
                mb_returns = returns[mb_indices]
                mb_old_values = old_values[mb_indices]
                
                # Policy loss
                new_log_probs = self.policy.log_prob(mb_obs, mb_actions)
                ratio = torch.exp(new_log_probs - mb_old_log_probs)
                
                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(
                    ratio, 1 - self.clip_ratio, 1 + self.clip_ratio
                ) * mb_advantages
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # Value loss (clipped)
                new_values = self.value_fn(mb_obs).squeeze(-1)
                value_clipped = mb_old_values + torch.clamp(
                    new_values - mb_old_values,
                    -self.value_clip,
                    self.value_clip,
                )
                value_loss1 = F.mse_loss(new_values, mb_returns)
                value_loss2 = F.mse_loss(value_clipped, mb_returns)
                value_loss = torch.max(value_loss1, value_loss2)
                
                # Entropy bonus
                entropy = self.policy.entropy(mb_obs).mean()
                
                # Total loss
                loss = (
                    policy_loss
                    + self.value_coef * value_loss
                    - self.entropy_coef * entropy
                )
                
                # Gradient step
                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + list(self.value_fn.parameters()),
                    self.max_grad_norm,
                )
                self.optimizer.step()
                
                # Compute metrics
                with torch.no_grad():
                    kl = (mb_old_log_probs - new_log_probs).mean()
                    clip_fraction = (
                        (torch.abs(ratio - 1) > self.clip_ratio).float().mean()
                    )
                
                metrics["policy_loss"] += policy_loss.item()
                metrics["value_loss"] += value_loss.item()
                metrics["entropy"] += entropy.item()
                metrics["kl"] += kl.item()
                metrics["clip_fraction"] += clip_fraction.item()
                num_updates += 1
            
            # Early stopping on KL divergence
            avg_kl = metrics["kl"] / num_updates
            if avg_kl > self.target_kl:
                early_stop = True
        
        # Average metrics
        for k in metrics:
            metrics[k] /= max(num_updates, 1)
        
        metrics["num_updates"] = num_updates
        metrics["early_stopped"] = early_stop
        
        return metrics
    
    def set_learning_rate(self, lr: float) -> None:
        """Update learning rate.
        
        Args:
            lr: New learning rate
        """
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr


def compute_explained_variance(
    values: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Compute explained variance of value function.
    
    EV = 1 - Var(returns - values) / Var(returns)
    
    EV = 1: Perfect value function
    EV = 0: Value function is constant (predicts mean)
    EV < 0: Value function is worse than predicting mean
    
    Args:
        values: Value predictions
        returns: Actual returns
        
    Returns:
        Explained variance
    """
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 1.0  # Returns are constant
    return 1 - np.var(returns - values) / var_returns
