# Value function (Critic)
# FORBIDDEN: env.*, training.*, logging, pathlib

import torch
import torch.nn as nn
from typing import List
from .blocks import get_activation


class ValueFunction(nn.Module):
    """State value function V(s).
    
    Estimates expected return from state s.
    Used for advantage estimation in PPO.
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        
        layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                get_activation(activation),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Estimate state value.
        
        Args:
            state: Observation, shape (batch, state_dim)
            
        Returns:
            value: Estimated value, shape (batch, 1)
        """
        return self.network(state)


class QFunction(nn.Module):
    """State-action value function Q(s, a).
    
    Estimates expected return from state s taking action a.
    Used in SAC and other off-policy algorithms.
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        
        if hidden_dims is None:
            hidden_dims = [256, 256]
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        layers = []
        in_dim = state_dim + action_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h_dim),
                get_activation(activation),
            ])
            in_dim = h_dim
        layers.append(nn.Linear(hidden_dims[-1], 1))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Estimate state-action value.
        
        Args:
            state: Observation, shape (batch, state_dim)
            action: Action, shape (batch, action_dim)
            
        Returns:
            value: Estimated Q-value, shape (batch, 1)
        """
        x = torch.cat([state, action], dim=-1)
        return self.network(x)


class TwinQFunction(nn.Module):
    """Twin Q-functions for SAC.
    
    Uses two Q-functions and takes minimum to reduce overestimation.
    """
    
    def __init__(
        self,
        state_dim: int = 34,
        action_dim: int = 3,
        hidden_dims: List[int] = None,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.q1 = QFunction(state_dim, action_dim, hidden_dims, activation)
        self.q2 = QFunction(state_dim, action_dim, hidden_dims, activation)
    
    def forward(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute both Q-values.
        
        Args:
            state: Observation
            action: Action
            
        Returns:
            q1, q2: Q-values from both networks
        """
        return self.q1(state, action), self.q2(state, action)
    
    def min_q(
        self,
        state: torch.Tensor,
        action: torch.Tensor,
    ) -> torch.Tensor:
        """Compute minimum Q-value.
        
        Args:
            state: Observation
            action: Action
            
        Returns:
            Minimum of q1 and q2
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)
