# Replay buffer for off-policy algorithms

import numpy as np
import torch
from typing import Dict, Optional, Tuple


class ReplayBuffer:
    """Experience replay buffer for off-policy algorithms.
    
    Stores transitions (s, a, r, s', done) for sampling.
    Used by SAC and other off-policy methods.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
    ):
        """Initialize replay buffer.
        
        Args:
            capacity: Maximum number of transitions to store
            state_dim: State dimension
            action_dim: Action dimension
        """
        self.capacity = capacity
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Preallocate arrays
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.float32)
        
        self.ptr = 0  # Next write position
        self.size = 0  # Current size
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition to buffer.
        
        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Episode termination flag
        """
        self.states[self.ptr] = state
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.next_states[self.ptr] = next_state
        self.dones[self.ptr] = float(done)
        
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def add_batch(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        rewards: np.ndarray,
        next_states: np.ndarray,
        dones: np.ndarray,
    ) -> None:
        """Add batch of transitions.
        
        Args:
            states: Batch of states
            actions: Batch of actions
            rewards: Batch of rewards
            next_states: Batch of next states
            dones: Batch of done flags
        """
        batch_size = states.shape[0]
        
        # Handle wraparound
        if self.ptr + batch_size <= self.capacity:
            self.states[self.ptr:self.ptr + batch_size] = states
            self.actions[self.ptr:self.ptr + batch_size] = actions
            self.rewards[self.ptr:self.ptr + batch_size] = rewards
            self.next_states[self.ptr:self.ptr + batch_size] = next_states
            self.dones[self.ptr:self.ptr + batch_size] = dones
        else:
            # Split across boundary
            first_part = self.capacity - self.ptr
            self.states[self.ptr:] = states[:first_part]
            self.actions[self.ptr:] = actions[:first_part]
            self.rewards[self.ptr:] = rewards[:first_part]
            self.next_states[self.ptr:] = next_states[:first_part]
            self.dones[self.ptr:] = dones[:first_part]
            
            second_part = batch_size - first_part
            self.states[:second_part] = states[first_part:]
            self.actions[:second_part] = actions[first_part:]
            self.rewards[:second_part] = rewards[first_part:]
            self.next_states[:second_part] = next_states[first_part:]
            self.dones[:second_part] = dones[first_part:]
        
        self.ptr = (self.ptr + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)
    
    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Dict[str, torch.Tensor]:
        """Sample batch of transitions.
        
        Args:
            batch_size: Number of transitions to sample
            device: Target device for tensors
            
        Returns:
            Dict with states, actions, rewards, next_states, dones
        """
        indices = np.random.randint(0, self.size, size=batch_size)
        
        batch = {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }
        
        if device is not None:
            batch = {
                k: torch.tensor(v, device=device, dtype=torch.float32)
                for k, v in batch.items()
            }
        
        return batch
    
    def __len__(self) -> int:
        return self.size
    
    def is_ready(self, batch_size: int) -> bool:
        """Check if buffer has enough samples."""
        return self.size >= batch_size


class PrioritizedReplayBuffer(ReplayBuffer):
    """Prioritized experience replay buffer.
    
    Samples transitions with probability proportional to TD error.
    POSTPONED: Implement after basic training works.
    """
    
    def __init__(
        self,
        capacity: int,
        state_dim: int,
        action_dim: int,
        alpha: float = 0.6,
        beta: float = 0.4,
    ):
        super().__init__(capacity, state_dim, action_dim)
        
        self.alpha = alpha  # Priority exponent
        self.beta = beta    # Importance sampling exponent
        
        # Priority storage
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.max_priority = 1.0
    
    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        """Add transition with max priority."""
        self.priorities[self.ptr] = self.max_priority
        super().add(state, action, reward, next_state, done)
    
    def sample(
        self,
        batch_size: int,
        device: Optional[torch.device] = None,
    ) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """Sample batch with priorities.
        
        Returns:
            batch: Dict of transitions
            indices: Sampled indices (for priority update)
            weights: Importance sampling weights
        """
        # Compute sampling probabilities
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # Sample indices
        indices = np.random.choice(self.size, size=batch_size, p=probs)
        
        # Compute importance sampling weights
        weights = (self.size * probs[indices]) ** (-self.beta)
        weights /= weights.max()  # Normalize
        
        batch = {
            "states": self.states[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "next_states": self.next_states[indices],
            "dones": self.dones[indices],
        }
        
        if device is not None:
            batch = {
                k: torch.tensor(v, device=device, dtype=torch.float32)
                for k, v in batch.items()
            }
            weights = torch.tensor(weights, device=device, dtype=torch.float32)
        
        return batch, indices, weights
    
    def update_priorities(
        self,
        indices: np.ndarray,
        priorities: np.ndarray,
    ) -> None:
        """Update priorities for sampled transitions.
        
        Args:
            indices: Indices of transitions
            priorities: New priorities (typically |TD error| + epsilon)
        """
        self.priorities[indices] = priorities
        self.max_priority = max(self.max_priority, priorities.max())
