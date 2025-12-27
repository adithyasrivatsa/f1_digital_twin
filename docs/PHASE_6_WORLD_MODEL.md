# PHASE 6 — World Model (Minimal, Interpretable)

## Purpose

The world model predicts future states given current state and action. It enables:
1. Planning without simulator queries
2. Imagination-based training (Dreamer-style)
3. Safety verification (predict crash before it happens)
4. Explainability (show predicted trajectory)

## VRAM Budget

RTX A4000: 16 GB VRAM

| Component | Budget | Actual |
|-----------|--------|--------|
| World model | 4 GB | ~2 GB |
| Policy network | 2 GB | ~500 MB |
| Value network | 2 GB | ~500 MB |
| Replay buffer | 4 GB | ~3 GB |
| Gradients + optimizer | 4 GB | ~3 GB |
| **Total** | 16 GB | ~9 GB |

Headroom is intentional. Memory spikes during training.

## Architecture: Deterministic MLP

```python
# src/models/world_model.py
import torch
import torch.nn as nn
from typing import Tuple

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
        hidden_dims: list[int] = [256, 256],
        activation: str = "elu",
    ):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Activation function
        act_fn = {"relu": nn.ReLU, "elu": nn.ELU, "tanh": nn.Tanh}[activation]
        
        # Encoder: state → latent
        encoder_layers = []
        in_dim = state_dim
        for h_dim in hidden_dims:
            encoder_layers.extend([
                nn.Linear(in_dim, h_dim),
                act_fn(),
            ])
            in_dim = h_dim
        self.encoder = nn.Sequential(*encoder_layers)
        
        # Dynamics: latent + action → next latent
        dynamics_layers = []
        in_dim = hidden_dims[-1] + action_dim
        for h_dim in hidden_dims:
            dynamics_layers.extend([
                nn.Linear(in_dim, h_dim),
                act_fn(),
            ])
            in_dim = h_dim
        self.dynamics = nn.Sequential(*dynamics_layers)
        
        # Decoder: latent → state
        self.decoder = nn.Linear(hidden_dims[-1], state_dim)
        
        # Reward predictor: latent + action → reward
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dims[-1] + action_dim, 128),
            act_fn(),
            nn.Linear(128, 1),
        )
        
        # Termination predictor: latent → done probability
        self.done_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 64),
            act_fn(),
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
        latent = self.encoder(state)
        
        # Predict dynamics
        latent_action = torch.cat([latent, action], dim=-1)
        next_latent = self.dynamics(latent_action)
        
        # Decode to state
        next_state = self.decoder(next_latent)
        
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
        device = initial_state.device
        
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
```

## Latent Dimension Bounds

| Constraint | Value | Reason |
|------------|-------|--------|
| Minimum latent dim | 32 | Must capture state complexity |
| Maximum latent dim | 128 | VRAM and overfitting |
| Recommended | 64 | Balance of capacity and efficiency |

## Loss Functions

### Allowed Losses

```python
# src/training/world_model_loss.py
import torch
import torch.nn.functional as F

def state_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """MSE loss for state prediction.
    
    Why MSE:
    1. Continuous state space
    2. Gaussian assumption is reasonable for physics
    3. Simple and stable
    """
    return F.mse_loss(predicted, target)

def reward_prediction_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Huber loss for reward prediction.
    
    Why Huber:
    1. Robust to outliers (crashes have extreme rewards)
    2. Smooth gradient near zero
    """
    return F.huber_loss(predicted, target, delta=1.0)

def termination_loss(
    predicted: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Binary cross-entropy for termination prediction.
    
    Why BCE:
    1. Binary classification (done or not)
    2. Handles class imbalance via weighting
    """
    # Weight positive class (terminations are rare)
    pos_weight = torch.tensor([10.0], device=predicted.device)
    return F.binary_cross_entropy(predicted, target, weight=pos_weight)

def world_model_loss(
    model: WorldModel,
    states: torch.Tensor,
    actions: torch.Tensor,
    next_states: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """Combined world model loss.
    
    Returns dict for logging individual components.
    """
    pred_next, pred_reward, pred_done = model(states, actions)
    
    state_loss = state_prediction_loss(pred_next, next_states)
    reward_loss = reward_prediction_loss(pred_reward.squeeze(), rewards)
    done_loss = termination_loss(pred_done.squeeze(), dones.float())
    
    # Weighted combination
    total_loss = state_loss + 0.5 * reward_loss + 0.1 * done_loss
    
    return {
        "total": total_loss,
        "state": state_loss,
        "reward": reward_loss,
        "done": done_loss,
    }
```

### Forbidden Losses

| Loss | Why Forbidden |
|------|---------------|
| Contrastive loss | Requires negative sampling, complex |
| VAE ELBO | Adds KL term, harder to tune |
| GAN loss | Training instability |
| Perceptual loss | Requires pretrained network |

## Why Transformers Are Postponed

| Reason | Explanation |
|--------|-------------|
| VRAM | Attention is O(n²) in sequence length |
| Sample efficiency | Transformers need more data |
| Debugging | Attention patterns are hard to interpret |
| Overkill | MLP sufficient for Markov dynamics |
| Complexity | Positional encoding, masking, etc. |

Transformers are appropriate when:
- Sequence modeling is essential (language, long-horizon planning)
- Data is abundant (millions of samples)
- Compute is abundant (multiple GPUs)

None of these apply to initial development.

## Model Validation

```python
# tests/test_models/test_world_model.py
import torch
import pytest
from src.models.world_model import WorldModel

class TestWorldModel:
    
    @pytest.fixture
    def model(self):
        return WorldModel(state_dim=34, action_dim=3)
        
    def test_forward_shapes(self, model):
        """Verify output shapes match specification."""
        batch = 32
        state = torch.randn(batch, 34)
        action = torch.randn(batch, 3)
        
        next_state, reward, done = model(state, action)
        
        assert next_state.shape == (batch, 34)
        assert reward.shape == (batch, 1)
        assert done.shape == (batch, 1)
        
    def test_rollout_shapes(self, model):
        """Verify rollout shapes."""
        batch = 16
        horizon = 10
        state = torch.randn(batch, 34)
        actions = torch.randn(batch, horizon, 3)
        
        states, rewards, dones = model.rollout(state, actions)
        
        assert states.shape == (batch, horizon + 1, 34)
        assert rewards.shape == (batch, horizon)
        assert dones.shape == (batch, horizon)
        
    def test_deterministic(self, model):
        """Verify model is deterministic."""
        model.eval()
        state = torch.randn(1, 34)
        action = torch.randn(1, 3)
        
        out1 = model(state, action)
        out2 = model(state, action)
        
        assert torch.allclose(out1[0], out2[0])
        
    def test_gradient_flow(self, model):
        """Verify gradients flow through model."""
        state = torch.randn(8, 34, requires_grad=True)
        action = torch.randn(8, 3)
        
        next_state, _, _ = model(state, action)
        loss = next_state.sum()
        loss.backward()
        
        assert state.grad is not None
        assert not torch.isnan(state.grad).any()
        
    def test_parameter_count(self, model):
        """Verify model fits VRAM budget."""
        params = sum(p.numel() for p in model.parameters())
        # Should be < 1M parameters for MLP
        assert params < 1_000_000
        
        # Estimate memory: params * 4 bytes * 2 (weights + gradients)
        memory_mb = params * 4 * 2 / 1e6
        assert memory_mb < 100  # Should be < 100 MB
```

## Ensemble for Uncertainty (Future)

```python
# POSTPONED: Add after single model works
class WorldModelEnsemble(nn.Module):
    """Ensemble of world models for uncertainty estimation.
    
    Uncertainty = disagreement between ensemble members.
    High uncertainty → don't trust prediction → be conservative.
    """
    
    def __init__(self, num_models: int = 5, **model_kwargs):
        super().__init__()
        self.models = nn.ModuleList([
            WorldModel(**model_kwargs) for _ in range(num_models)
        ])
        
    def forward(self, state, action):
        predictions = [m(state, action) for m in self.models]
        
        # Mean prediction
        mean_next = torch.stack([p[0] for p in predictions]).mean(0)
        
        # Uncertainty (std across ensemble)
        std_next = torch.stack([p[0] for p in predictions]).std(0)
        
        return mean_next, std_next
```
