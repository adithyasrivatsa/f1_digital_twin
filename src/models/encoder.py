# State encoder
# FORBIDDEN: env.*, training.*, logging, pathlib

import torch
import torch.nn as nn
from typing import List
from .blocks import MLP, get_activation


class StateEncoder(nn.Module):
    """Encode raw state to latent representation.
    
    Used to share representations between policy and value function.
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        latent_dim: int = 64,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.latent_dim = latent_dim
        
        self.encoder = MLP(
            input_dim=state_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Encode state to latent.
        
        Args:
            state: Raw state, shape (batch, state_dim)
            
        Returns:
            Latent representation, shape (batch, latent_dim)
        """
        return self.encoder(state)


class SharedEncoder(nn.Module):
    """Shared encoder for actor-critic architectures.
    
    Encodes state once, then separate heads for policy and value.
    Reduces computation and can improve learning.
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 3,
        latent_dim: int = 256,
        hidden_dims: List[int] = None,
        activation: str = "relu",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256]
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared encoder
        self.encoder = MLP(
            input_dim=state_dim,
            output_dim=latent_dim,
            hidden_dims=hidden_dims,
            activation=activation,
        )
        
        # Policy head
        self.policy_mean = nn.Linear(latent_dim, action_dim)
        self.policy_log_std = nn.Linear(latent_dim, action_dim)
        
        # Value head
        self.value_head = nn.Sequential(
            nn.Linear(latent_dim, 256),
            get_activation(activation),
            nn.Linear(256, 1),
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights."""
        nn.init.orthogonal_(self.policy_mean.weight, gain=0.01)
        nn.init.zeros_(self.policy_mean.bias)
        nn.init.orthogonal_(self.policy_log_std.weight, gain=0.01)
        nn.init.zeros_(self.policy_log_std.bias)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass through shared encoder.
        
        Args:
            state: Raw state, shape (batch, state_dim)
            
        Returns:
            mean: Policy mean, shape (batch, action_dim)
            log_std: Policy log std, shape (batch, action_dim)
            value: State value, shape (batch, 1)
        """
        latent = self.encoder(state)
        
        mean = self.policy_mean(latent)
        log_std = self.policy_log_std(latent)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        value = self.value_head(latent)
        
        return mean, log_std, value
    
    def get_latent(self, state: torch.Tensor) -> torch.Tensor:
        """Get latent representation for analysis."""
        return self.encoder(state)
