# Reusable network building blocks
# FORBIDDEN: env.*, training.*, logging, pathlib

import torch
import torch.nn as nn
from typing import List, Optional


def get_activation(name: str) -> nn.Module:
    """Get activation function by name.
    
    Args:
        name: Activation name ("relu", "elu", "tanh", "gelu")
        
    Returns:
        Activation module
    """
    activations = {
        "relu": nn.ReLU,
        "elu": nn.ELU,
        "tanh": nn.Tanh,
        "gelu": nn.GELU,
        "silu": nn.SiLU,
    }
    if name not in activations:
        raise ValueError(f"Unknown activation: {name}. Available: {list(activations.keys())}")
    return activations[name]()


class MLP(nn.Module):
    """Multi-layer perceptron.
    
    Simple feedforward network with configurable depth and width.
    """
    
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        activation: str = "relu",
        output_activation: Optional[str] = None,
        dropout: float = 0.0,
    ):
        super().__init__()
        
        layers = []
        in_dim = input_dim
        
        for h_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, h_dim))
            layers.append(get_activation(activation))
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
            in_dim = h_dim
        
        layers.append(nn.Linear(in_dim, output_dim))
        
        if output_activation is not None:
            layers.append(get_activation(output_activation))
        
        self.network = nn.Sequential(*layers)
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights with orthogonal initialization."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=1.0)
                nn.init.zeros_(m.bias)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


class ResidualBlock(nn.Module):
    """Residual block with skip connection.
    
    output = activation(linear(x)) + x
    """
    
    def __init__(
        self,
        dim: int,
        activation: str = "relu",
    ):
        super().__init__()
        
        self.linear1 = nn.Linear(dim, dim)
        self.linear2 = nn.Linear(dim, dim)
        self.activation = get_activation(activation)
        self.norm = nn.LayerNorm(dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.activation(self.linear1(x))
        x = self.linear2(x)
        x = self.norm(x + residual)
        return x


class GRUEncoder(nn.Module):
    """GRU-based sequence encoder.
    
    For processing temporal sequences of states.
    POSTPONED: Use MLP first, add GRU if needed.
    """
    
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_layers: int = 1,
    ):
        super().__init__()
        
        self.gru = nn.GRU(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
        )
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
    
    def forward(
        self,
        x: torch.Tensor,
        hidden: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Forward pass.
        
        Args:
            x: Input sequence, shape (batch, seq_len, input_dim)
            hidden: Initial hidden state, shape (num_layers, batch, hidden_dim)
            
        Returns:
            output: Output sequence, shape (batch, seq_len, hidden_dim)
            hidden: Final hidden state
        """
        output, hidden = self.gru(x, hidden)
        return output, hidden
    
    def init_hidden(self, batch_size: int, device: torch.device) -> torch.Tensor:
        """Initialize hidden state."""
        return torch.zeros(self.num_layers, batch_size, self.hidden_dim, device=device)
