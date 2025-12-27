# PHASE 8 — Training Loop Design

## Rollout Strategy

```python
# src/training/rollout.py
import numpy as np
import torch
from dataclasses import dataclass
from typing import List

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
    advantages: np.ndarray = None
    returns: np.ndarray = None
    
    def compute_gae(
        self,
        last_values: np.ndarray,
        gamma: float = 0.99,
        gae_lambda: float = 0.95,
    ):
        """Compute Generalized Advantage Estimation.
        
        GAE balances bias-variance tradeoff in advantage estimation.
        lambda=0: high bias, low variance (TD(0))
        lambda=1: low bias, high variance (Monte Carlo)
        lambda=0.95: good balance for most tasks
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
        
    def flatten(self) -> dict:
        """Flatten for minibatch sampling."""
        batch_size = self.observations.shape[0] * self.observations.shape[1]
        
        return {
            "observations": self.observations.reshape(batch_size, -1),
            "actions": self.actions.reshape(batch_size, -1),
            "log_probs": self.log_probs.reshape(batch_size),
            "advantages": self.advantages.reshape(batch_size),
            "returns": self.returns.reshape(batch_size),
            "values": self.values.reshape(batch_size),
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
    
    # Preallocate buffers
    observations = np.zeros((num_envs, num_steps, obs_dim), dtype=np.float32)
    actions = np.zeros((num_envs, num_steps, action_dim), dtype=np.float32)
    rewards = np.zeros((num_envs, num_steps), dtype=np.float32)
    values = np.zeros((num_envs, num_steps), dtype=np.float32)
    log_probs = np.zeros((num_envs, num_steps), dtype=np.float32)
    dones = np.zeros((num_envs, num_steps), dtype=np.float32)
    
    obs, _ = envs.reset()
    
    for step in range(num_steps):
        observations[:, step] = obs
        
        # Policy inference on GPU
        with torch.no_grad():
            obs_tensor = torch.tensor(obs, device=device)
            action, log_prob = policy.sample(obs_tensor)
            value = value_fn(obs_tensor)
            
        # Move to CPU
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
        dones[:, step] = done
        
        obs = next_obs
        
    # Get final values for GAE
    with torch.no_grad():
        obs_tensor = torch.tensor(obs, device=device)
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
```

## Update Cadence

```python
# src/training/ppo.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class PPOUpdate:
    """PPO policy update."""
    
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
        self.policy = policy
        self.value_fn = value_fn
        self.clip_ratio = clip_ratio
        self.value_clip = value_clip
        self.entropy_coef = entropy_coef
        self.value_coef = value_coef
        self.max_grad_norm = max_grad_norm
        self.target_kl = target_kl
        
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
            buffer: Flattened rollout buffer (on device)
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
        
        for epoch in range(num_epochs):
            # Shuffle indices
            indices = torch.randperm(batch_size)
            
            for start in range(0, batch_size, minibatch_size):
                end = start + minibatch_size
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
                mean, log_std = self.policy(mb_obs)
                entropy = (log_std + 0.5 * np.log(2 * np.pi * np.e)).sum(-1).mean()
                
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
                
                # Metrics
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
            if metrics["kl"] / num_updates > self.target_kl:
                break
                
        # Average metrics
        for k in metrics:
            metrics[k] /= num_updates
            
        return metrics
```

## Evaluation Protocol

```python
# src/training/evaluation.py
import numpy as np
from typing import Dict, List

def evaluate_policy(
    env,
    policy,
    num_episodes: int = 5,
    deterministic: bool = True,
    device: str = "cpu",
) -> Dict[str, float]:
    """Evaluate policy performance.
    
    Args:
        env: Single environment (not vectorized)
        policy: Policy network
        num_episodes: Number of evaluation episodes
        deterministic: Use deterministic actions
        device: Torch device
        
    Returns:
        Dict of evaluation metrics
    """
    episode_returns = []
    episode_lengths = []
    lap_times = []
    off_track_counts = []
    
    for _ in range(num_episodes):
        obs, _ = env.reset()
        done = False
        episode_return = 0.0
        episode_length = 0
        off_track = 0
        
        while not done:
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.sample(obs_tensor, deterministic=deterministic)
            action_np = action.cpu().numpy().squeeze()
            
            result = env.step(action_np)
            obs = result.observation
            episode_return += result.reward
            episode_length += 1
            
            if result.info.get("off_track", False):
                off_track += 1
            if result.info.get("lap_complete", False):
                lap_times.append(result.info.get("lap_time", episode_length))
                
            done = result.terminated or result.truncated
            
        episode_returns.append(episode_return)
        episode_lengths.append(episode_length)
        off_track_counts.append(off_track)
        
    return {
        "eval/mean_return": np.mean(episode_returns),
        "eval/std_return": np.std(episode_returns),
        "eval/mean_length": np.mean(episode_lengths),
        "eval/mean_lap_time": np.mean(lap_times) if lap_times else 0.0,
        "eval/lap_completion_rate": len(lap_times) / num_episodes,
        "eval/mean_off_track": np.mean(off_track_counts),
    }
```

## Avoiding Reward Hacking

Reward hacking: Agent finds unintended way to maximize reward that violates intent.

Examples in racing:
- Spinning in circles (maximizes "distance traveled" if poorly defined)
- Driving backwards (if only forward progress is rewarded)
- Exploiting physics glitches

Prevention:

```python
# src/training/reward.py
def compute_reward(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    info: dict,
    config,
) -> float:
    """Compute shaped reward with hack prevention.
    
    Reward components are additive and bounded.
    Each component has clear physical meaning.
    """
    reward = 0.0
    
    # Progress reward (bounded by track length)
    progress = info["lap_progress"] - info.get("prev_lap_progress", 0)
    progress = np.clip(progress, -0.1, 0.1)  # Prevent teleportation exploits
    reward += progress * config.reward.progress_weight
    
    # Speed reward (only if on track and moving forward)
    if not info["off_track"] and progress > 0:
        speed = state[0]  # Forward velocity
        speed_reward = np.clip(speed / 100.0, 0, 1)  # Normalized
        reward += speed_reward * config.reward.speed_weight
        
    # Penalties (always negative)
    if info["off_track"]:
        reward += config.reward.off_track_penalty  # Negative
        
    if info.get("collision", False):
        reward += config.reward.collision_penalty  # Negative
        
    # Smoothness (penalize jerky control)
    if "prev_action" in info:
        action_diff = np.abs(action - info["prev_action"]).sum()
        reward -= action_diff * config.reward.smoothness_penalty
        
    # Clip total reward
    reward = np.clip(reward, -config.reward.clip, config.reward.clip)
    
    return reward
```

## Detecting Policy Collapse Early

Policy collapse: Policy converges to degenerate behavior (always same action, zero entropy).

Detection:

```python
# src/training/diagnostics.py
def check_policy_health(metrics: Dict[str, float]) -> List[str]:
    """Check for signs of policy collapse.
    
    Returns list of warnings (empty if healthy).
    """
    warnings = []
    
    # Entropy collapse
    if metrics["entropy"] < 0.1:
        warnings.append(f"LOW ENTROPY: {metrics['entropy']:.3f} - policy may be collapsing")
        
    # KL divergence spike
    if metrics["kl"] > 0.1:
        warnings.append(f"HIGH KL: {metrics['kl']:.3f} - policy changing too fast")
        
    # Value function divergence
    if metrics.get("explained_variance", 1.0) < 0.0:
        warnings.append(f"NEGATIVE EXPLAINED VARIANCE - value function is worse than mean")
        
    # Gradient issues
    if metrics.get("grad_norm", 0) > 10.0:
        warnings.append(f"HIGH GRADIENT NORM: {metrics['grad_norm']:.1f}")
        
    # Clip fraction too high
    if metrics["clip_fraction"] > 0.3:
        warnings.append(f"HIGH CLIP FRACTION: {metrics['clip_fraction']:.2f} - learning rate may be too high")
        
    return warnings
```

## Why Stability Metrics Matter More Than Reward

Reward is noisy. A single good episode can spike mean reward.

Stability metrics reveal true learning:
- Explained variance: Is value function learning?
- KL divergence: Is policy changing smoothly?
- Entropy: Is policy exploring?
- Clip fraction: Is learning rate appropriate?

```python
def compute_explained_variance(
    values: np.ndarray,
    returns: np.ndarray,
) -> float:
    """Compute explained variance of value function.
    
    EV = 1 - Var(returns - values) / Var(returns)
    
    EV = 1: Perfect value function
    EV = 0: Value function is constant (predicts mean)
    EV < 0: Value function is worse than predicting mean
    """
    var_returns = np.var(returns)
    if var_returns < 1e-8:
        return 1.0  # Returns are constant, any prediction is fine
    return 1 - np.var(returns - values) / var_returns
```

A run with increasing reward but decreasing explained variance is not learning — it is getting lucky.
