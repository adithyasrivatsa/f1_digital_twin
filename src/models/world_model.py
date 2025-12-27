# World model for dynamics prediction
# FORBIDDEN: env.*, training.*, logging, pathlib

import torch
import torch.nn as nn
from typing import Tuple, List
from .blocks import MLP, get_activation


class WorldModel(nn.Module):
    """Deterministic world model for short-horizon prediction.
    
    Architecture: State + Action → MLP → Next State
    
    Why deterministic (not probabilistic):
    1. Simpler to train and debug
    2. Sufficient for short horizons (< 1 second)
    3. Uncertainty can be added later via ensemble
    
    Why MLP (not RNN/Transformer):
    1. Markov assumption holds for physics
    2. Faster training and inference
    3. Easier to interpret
    4. Sufficient for 16-step horizon
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 3,
        hidden_dims: List[int] = None,
        latent_dim: int = 64,
        activation: str = "elu",
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.latent_dim = latent_dim
        
        act_fn = get_activation(activation)
        
        # Encoder: state → latent
        encoder_layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                get_activation(activation),
            ])
            in_dim = h_dim
        encoder_layers.append(nn.Linear(in_dim, latent_dim))
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Dynamics: latent + action → next latent
        dynamics_layers = []
        in_dim = latent_dim + action_dim
        for h_dim in hidden_dims:
            dynamics_layers.extend([
                nn.Linear(in_dim, h_dim),
                get_activation(activation),
            ])
            in_dim = h_dim
        dynamics_layers.append(nn.Linear(in_dim, latent_dim))
        self.dynamics = nn.Sequential(*dynamics_layers)
        
        # Decoder: latent → state
        self.decoder = nn.Linear(latent_dim, state_dim)
        
        # Reward predictor: latent + action → reward
        self.reward_head = nn.Sequential(
            nn.Linear(latent_dim + action_dim, 128),
            get_activation(activation),
            nn.Linear(128, 1),
        )
        
        # Termination predictor: latent → done probability
        self.done_head = nn.Sequential(
            nn.Linear(latent_dim, 64),
            get_activation(activation),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def encode(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent representation.
        
        Args:
            state: State tensor, shape (batch, state_dim)
            
        Returns:
            Latent tensor, shape (batch, latent_dim)
        """
        return self.encoder(state)
    
    def decode(self, latent: torch.Tensor) -> torch.Tensor:
        """Decode latent to state.
        
        Args:
            latent: Latent tensor, shape (batch, latent_dim)
            
        Returns:
            State tensor, shape (batch, state_dim)
        """
        return self.decoder(latent)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict next state, reward, and termination.
        
        Args:
            state: Current state, shape (batch, state_dim)
            action: Action taken, shape (batch, action_dim)
            
        Returns:
            next_state: Predicted next state, shape (batch, state_dim)
            reward: Predicted reward, shape (batch, 1)
            done: Predicted termination probability, shape (batch, 1)
        """
        # Encode state
        latent = self.encode(state)
        
        # Predict dynamics
        latent_action = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(latent_action)
        
        # Decode to state
        next_state = self.decode(next_latent)
        
        # Predict reward and termination
        reward = self.reward_head(latent_action)
        done = self.done_head(next_latent)
        
        return next_state, reward, done
    
    def rollout(
        self,
        initial_state: torch.Tensor,
        actions: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Rollout world model for multiple steps.
        
        Args:
            initial_state: Starting state, shape (batch, state_dim)
            actions: Action sequence, shape (batch, horizon, action_dim)
            
        Returns:
            states: Predicted states, shape (batch, horizon+1, state_dim)
            rewards: Predicted rewards, shape (batch, horizon)
            dones: Predicted terminations, shape (batch, horizon)
        """
        batch_size, horizon, _ = actions.shape
        
        states = [initial_state]
        rewards = []
        dones = []
        
        state = initial_state
        for t in range(horizon):
            action = actions[:, t, :]
            next_state, reward, done = self.forward(state, action)
            
            states.append(next_state)
            rewards.append(reward.squeeze(-1))
            dones.append(done.squeeze(-1))
            
            state = next_state
        
        states = torch.stack(states, dim=1)  # (batch, horizon+1, state_dim)
        rewards = torch.stack(rewards, dim=1)  # (batch, horizon)
        dones = torch.stack(dones, dim=1)  # (batch, horizon)
        
        return states, rewards, dones
    
    def get_latent(self, state: torch.Tensor) -> torch.Tensor:
        """Get latent representation for analysis.
        
        Args:
            state: State tensor
            
        Returns:
            Latent tensor
        """
        return self.encode(state)
