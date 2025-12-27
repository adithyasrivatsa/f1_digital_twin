# Policy network (Actor)
# FORBIDDEN: env.*, training.*, logging, pathlib

import torch
import torch.nn as nn
from torch.distributions import Normal
from typing import Tuple, List, Optional
import numpy as np
from .blocks import get_activation


class GaussianPolicy(nn.Module):
    """Gaussian policy for continuous control.
    
    Outputs mean and log_std for each action dimension.
    Actions sampled from Normal(mean, exp(log_std)).
    
    Uses tanh squashing to bound actions to [-1, 1].
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = "tanh",
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
        # Shared feature extractor
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                get_activation(activation),
            ])
            in_dim = h_dim
        self.features = nn.Sequential(*layers)
        
        # Mean head
        self.mean_head = nn.Linear(hidden_dims[-1], action_dim)
        
        # Log std head (state-dependent)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.01)
                nn.init.zeros_(m.bias)
        
        # Initialize mean head with smaller weights
        nn.init.orthogonal_(self.mean_head.weight, gain=0.01)
        nn.init.zeros_(self.mean_head.bias)
    
    def forward(
        self,
        state: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute action distribution parameters.
        
        Args:
            state: Observation, shape (batch, state_dim)
            
        Returns:
            mean: Action mean, shape (batch, action_dim)
            log_std: Action log std, shape (batch, action_dim)
        """
        features = self.features(state)
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        return mean, log_std
    
    def sample(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample action from policy.
        
        Args:
            state: Observation, shape (batch, state_dim)
            deterministic: If True, return mean (no sampling)
            
        Returns:
            action: Sampled action, shape (batch, action_dim)
            log_prob: Log probability of action, shape (batch,)
        """
        mean, log_std = self.forward(state)
        
        if deterministic:
            # Return tanh(mean) for deterministic action
            action = torch.tanh(mean)
            # Log prob is not meaningful for deterministic, return zeros
            log_prob = torch.zeros(state.shape[0], device=state.device)
        else:
            std = log_std.exp()
            dist = Normal(mean, std)
            
            # Reparameterized sample
            x = dist.rsample()
            
            # Squash to [-1, 1]
            action = torch.tanh(x)
            
            # Compute log probability with tanh correction
            log_prob = dist.log_prob(x).sum(dim=-1)
            # Correction for tanh squashing
            log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return action, log_prob
    
    def log_prob(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute log probability of action.
        
        Args:
            state: Observation, shape (batch, state_dim)
            action: Action (already squashed to [-1, 1]), shape (batch, action_dim)
            
        Returns:
            log_prob: Log probability, shape (batch,)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Unsquash action for log_prob computation
        # atanh is unstable at boundaries, so clamp
        action_clamped = action.clamp(-0.999, 0.999)
        x = torch.atanh(action_clamped)
        
        dist = Normal(mean, std)
        log_prob = dist.log_prob(x).sum(dim=-1)
        
        # Correction for tanh squashing
        log_prob -= torch.log(1 - action.pow(2) + 1e-6).sum(dim=-1)
        
        return log_prob
    
    def entropy(self, state: torch.Tensor) -> torch.Tensor:
        """Compute entropy of action distribution.
        
        Args:
            state: Observation, shape (batch, state_dim)
            
        Returns:
            entropy: Entropy, shape (batch,)
        """
        _, log_std = self.forward(state)
        # Entropy of Gaussian: 0.5 * log(2 * pi * e * sigma^2)
        # = 0.5 * (1 + log(2*pi) + 2*log_std)
        # = log_std + 0.5 * log(2*pi*e)
        entropy = log_std + 0.5 * np.log(2 * np.pi * np.e)
        return entropy.sum(dim=-1)
    
    def get_distribution(
        self,
        state: torch.Tensor,
    ) -> Normal:
        """Get action distribution for analysis.
        
        Args:
            state: Observation
            
        Returns:
            Normal distribution
        """
        mean, log_std = self.forward(state)
        return Normal(mean, log_std.exp())
