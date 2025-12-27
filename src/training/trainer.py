# Main trainer class

import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Any
import time
import logging

from ..env import make_vec_env
from ..models import GaussianPolicy, ValueFunction, WorldModel
from .rollout import collect_rollout
from .ppo import PPOUpdate, compute_explained_variance


logger = logging.getLogger(__name__)


class Trainer:
    """Main training orchestrator.
    
    Coordinates environment, models, and training loop.
    """
    
    def __init__(self, config: Dict[str, Any]):
        """Initialize trainer.
        
        Args:
            config: Configuration dictionary
        """
        self.config = config
        
        # Device setup
        device_type = config.get("device", {}).get("type", "cpu")
        if device_type == "cuda":
            cuda_device = config.get("device", {}).get("cuda_device", 0)
            self.device = torch.device(f"cuda:{cuda_device}")
        else:
            self.device = torch.device("cpu")
        
        logger.info(f"Using device: {self.device}")
        
        # Dimensions
        self.state_dim = config.get("state", {}).get("dimension", 34)
        self.action_dim = config.get("action", {}).get("dimension", 3)
        
        # Create environment
        num_envs = config.get("training", {}).get("rollout", {}).get("num_envs", 8)
        self.envs = make_vec_env(config, num_envs)
        
        # Create models
        self._create_models()
        
        # Create optimizer
        self._create_optimizer()
        
        # Training state
        self.global_step = 0
        self.episode_count = 0
        self.best_return = float("-inf")
        
        # Metrics
        self.metrics_history = []
    
    def _create_models(self) -> None:
        """Create neural network models."""
        policy_config = self.config.get("policy", {})
        
        # Policy network
        actor_config = policy_config.get("actor", {})
        self.policy = GaussianPolicy(
            state_dim=self.state_dim,
            action_dim=self.action_dim,
            hidden_dims=actor_config.get("hidden_dims", [256, 256]),
            activation=actor_config.get("activation", "tanh"),
            log_std_min=actor_config.get("log_std_min", -20.0),
            log_std_max=actor_config.get("log_std_max", 2.0),
        ).to(self.device)
        
        # Value network
        critic_config = policy_config.get("critic", {})
        self.value_fn = ValueFunction(
            state_dim=self.state_dim,
            hidden_dims=critic_config.get("hidden_dims", [256, 256]),
            activation=critic_config.get("activation", "relu"),
        ).to(self.device)
        
        # World model (optional)
        wm_config = self.config.get("world_model", {})
        if wm_config.get("enabled", False):
            encoder_config = wm_config.get("encoder", {})
            self.world_model = WorldModel(
                state_dim=self.state_dim,
                action_dim=self.action_dim,
                hidden_dims=encoder_config.get("hidden_dims", [256, 256]),
                latent_dim=encoder_config.get("latent_dim", 64),
                activation=encoder_config.get("activation", "elu"),
            ).to(self.device)
        else:
            self.world_model = None
        
        # Log parameter counts
        policy_params = sum(p.numel() for p in self.policy.parameters())
        value_params = sum(p.numel() for p in self.value_fn.parameters())
        logger.info(f"Policy parameters: {policy_params:,}")
        logger.info(f"Value parameters: {value_params:,}")
        
        if self.world_model is not None:
            wm_params = sum(p.numel() for p in self.world_model.parameters())
            logger.info(f"World model parameters: {wm_params:,}")
    
    def _create_optimizer(self) -> None:
        """Create PPO optimizer."""
        ppo_config = self.config.get("policy", {}).get("ppo", {})
        lr_config = self.config.get("training", {}).get("lr_schedule", {})
        
        self.ppo = PPOUpdate(
            policy=self.policy,
            value_fn=self.value_fn,
            lr=lr_config.get("initial", 3e-4),
            clip_ratio=ppo_config.get("clip_ratio", 0.2),
            value_clip=ppo_config.get("value_clip", 0.2),
            entropy_coef=ppo_config.get("entropy_coef", 0.01),
            value_coef=ppo_config.get("value_coef", 0.5),
            max_grad_norm=ppo_config.get("max_grad_norm", 0.5),
            target_kl=ppo_config.get("target_kl", 0.01),
        )
    
    def train(self, total_timesteps: Optional[int] = None) -> Dict[str, float]:
        """Run training loop.
        
        Args:
            total_timesteps: Total timesteps to train (overrides config)
            
        Returns:
            Final metrics
        """
        if total_timesteps is None:
            total_timesteps = self.config.get("training", {}).get("total_timesteps", 1_000_000)
        
        rollout_config = self.config.get("training", {}).get("rollout", {})
        steps_per_env = rollout_config.get("steps_per_env", 2048)
        num_envs = rollout_config.get("num_envs", 8)
        
        update_config = self.config.get("training", {}).get("update", {})
        num_epochs = update_config.get("epochs", 10)
        minibatch_size = update_config.get("minibatch_size", 64)
        
        eval_config = self.config.get("training", {}).get("eval", {})
        eval_frequency = eval_config.get("frequency", 10_000)
        
        steps_per_rollout = steps_per_env * num_envs
        num_rollouts = total_timesteps // steps_per_rollout
        
        logger.info(f"Starting training for {total_timesteps:,} timesteps")
        logger.info(f"Steps per rollout: {steps_per_rollout:,}")
        logger.info(f"Number of rollouts: {num_rollouts:,}")
        
        start_time = time.time()
        
        for rollout_idx in range(num_rollouts):
            # Collect rollout
            buffer = collect_rollout(
                envs=self.envs,
                policy=self.policy,
                value_fn=self.value_fn,
                num_steps=steps_per_env,
                device=self.device,
            )
            
            # Update learning rate
            self._update_learning_rate(rollout_idx, num_rollouts)
            
            # PPO update
            buffer_tensors = buffer.to_tensor(self.device)
            update_metrics = self.ppo.update(
                buffer=buffer_tensors,
                num_epochs=num_epochs,
                minibatch_size=minibatch_size,
            )
            
            # Compute additional metrics
            explained_var = compute_explained_variance(
                buffer.values.flatten(),
                buffer.returns.flatten(),
            )
            
            self.global_step += steps_per_rollout
            
            # Log metrics
            metrics = {
                "step": self.global_step,
                "rollout": rollout_idx,
                "mean_reward": buffer.rewards.mean(),
                "mean_return": buffer.returns.mean(),
                "explained_variance": explained_var,
                **update_metrics,
            }
            self.metrics_history.append(metrics)
            
            # Log progress
            if rollout_idx % 10 == 0:
                elapsed = time.time() - start_time
                fps = self.global_step / elapsed
                logger.info(
                    f"Step {self.global_step:,} | "
                    f"Return: {metrics['mean_return']:.2f} | "
                    f"Policy Loss: {metrics['policy_loss']:.4f} | "
                    f"Entropy: {metrics['entropy']:.4f} | "
                    f"FPS: {fps:.0f}"
                )
            
            # Evaluation
            if self.global_step % eval_frequency < steps_per_rollout:
                eval_metrics = self.evaluate()
                logger.info(f"Evaluation: {eval_metrics}")
        
        logger.info(f"Training complete. Total time: {time.time() - start_time:.1f}s")
        
        return metrics
    
    def _update_learning_rate(self, rollout_idx: int, num_rollouts: int) -> None:
        """Update learning rate according to schedule."""
        lr_config = self.config.get("training", {}).get("lr_schedule", {})
        schedule_type = lr_config.get("type", "linear")
        initial_lr = lr_config.get("initial", 3e-4)
        final_lr = lr_config.get("final", 1e-5)
        
        progress = rollout_idx / num_rollouts
        
        if schedule_type == "linear":
            lr = initial_lr + (final_lr - initial_lr) * progress
        elif schedule_type == "cosine":
            lr = final_lr + 0.5 * (initial_lr - final_lr) * (1 + np.cos(np.pi * progress))
        else:
            lr = initial_lr
        
        self.ppo.set_learning_rate(lr)
    
    def evaluate(self, num_episodes: int = 5) -> Dict[str, float]:
        """Evaluate current policy.
        
        Args:
            num_episodes: Number of evaluation episodes
            
        Returns:
            Evaluation metrics
        """
        from ..env import make_env, GymWrapper
        
        eval_env = GymWrapper(make_env(self.config))
        
        episode_returns = []
        episode_lengths = []
        
        for _ in range(num_episodes):
            obs, _ = eval_env.reset()
            done = False
            episode_return = 0.0
            episode_length = 0
            
            while not done:
                obs_tensor = torch.tensor(obs, device=self.device).unsqueeze(0)
                with torch.no_grad():
                    action, _ = self.policy.sample(obs_tensor, deterministic=True)
                action_np = action.cpu().numpy().squeeze()
                
                obs, reward, terminated, truncated, _ = eval_env.step(action_np)
                episode_return += reward
                episode_length += 1
                done = terminated or truncated
            
            episode_returns.append(episode_return)
            episode_lengths.append(episode_length)
        
        eval_env.close()
        
        return {
            "eval/mean_return": np.mean(episode_returns),
            "eval/std_return": np.std(episode_returns),
            "eval/mean_length": np.mean(episode_lengths),
        }
    
    def save_checkpoint(self, path: Path) -> None:
        """Save training checkpoint.
        
        Args:
            path: Checkpoint file path
        """
        checkpoint = {
            "global_step": self.global_step,
            "policy_state_dict": self.policy.state_dict(),
            "value_fn_state_dict": self.value_fn.state_dict(),
            "optimizer_state_dict": self.ppo.optimizer.state_dict(),
            "config": self.config,
        }
        
        if self.world_model is not None:
            checkpoint["world_model_state_dict"] = self.world_model.state_dict()
        
        torch.save(checkpoint, path)
        logger.info(f"Saved checkpoint to {path}")
    
    def load_checkpoint(self, path: Path) -> None:
        """Load training checkpoint.
        
        Args:
            path: Checkpoint file path
        """
        checkpoint = torch.load(path, map_location=self.device)
        
        self.global_step = checkpoint["global_step"]
        self.policy.load_state_dict(checkpoint["policy_state_dict"])
        self.value_fn.load_state_dict(checkpoint["value_fn_state_dict"])
        self.ppo.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        
        if self.world_model is not None and "world_model_state_dict" in checkpoint:
            self.world_model.load_state_dict(checkpoint["world_model_state_dict"])
        
        logger.info(f"Loaded checkpoint from {path} at step {self.global_step}")
    
    def close(self) -> None:
        """Clean up resources."""
        self.envs.close()
