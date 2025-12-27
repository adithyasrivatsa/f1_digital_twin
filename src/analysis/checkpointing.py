# Checkpoint save/load utilities

import torch
from pathlib import Path
from typing import Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


def save_checkpoint(
    path: Path,
    step: int,
    policy_state: Dict[str, Any],
    value_state: Dict[str, Any],
    optimizer_state: Dict[str, Any],
    config: Dict[str, Any],
    world_model_state: Optional[Dict[str, Any]] = None,
    metrics: Optional[Dict[str, float]] = None,
) -> None:
    """Save training checkpoint.
    
    Args:
        path: Checkpoint file path
        step: Current training step
        policy_state: Policy network state dict
        value_state: Value network state dict
        optimizer_state: Optimizer state dict
        config: Training configuration
        world_model_state: Optional world model state dict
        metrics: Optional current metrics
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    checkpoint = {
        "step": step,
        "policy_state_dict": policy_state,
        "value_state_dict": value_state,
        "optimizer_state_dict": optimizer_state,
        "config": config,
    }
    
    if world_model_state is not None:
        checkpoint["world_model_state_dict"] = world_model_state
    
    if metrics is not None:
        checkpoint["metrics"] = metrics
    
    # Save to temporary file first, then rename (atomic)
    temp_path = path.with_suffix(".tmp")
    torch.save(checkpoint, temp_path)
    temp_path.rename(path)
    
    logger.info(f"Saved checkpoint to {path} at step {step}")


def load_checkpoint(
    path: Path,
    device: torch.device = torch.device("cpu"),
) -> Dict[str, Any]:
    """Load training checkpoint.
    
    Args:
        path: Checkpoint file path
        device: Device to load tensors to
        
    Returns:
        Checkpoint dict
    """
    path = Path(path)
    
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    
    checkpoint = torch.load(path, map_location=device)
    
    logger.info(f"Loaded checkpoint from {path} at step {checkpoint.get('step', 'unknown')}")
    
    return checkpoint


def get_latest_checkpoint(checkpoint_dir: Path) -> Optional[Path]:
    """Find latest checkpoint in directory.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        
    Returns:
        Path to latest checkpoint or None
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return None
    
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    
    if not checkpoints:
        return None
    
    # Sort by step number
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    
    return checkpoints[0]


def cleanup_old_checkpoints(
    checkpoint_dir: Path,
    keep_last: int = 5,
    keep_best: bool = True,
) -> None:
    """Remove old checkpoints, keeping only recent ones.
    
    Args:
        checkpoint_dir: Directory containing checkpoints
        keep_last: Number of recent checkpoints to keep
        keep_best: Whether to keep best.pt
    """
    checkpoint_dir = Path(checkpoint_dir)
    
    if not checkpoint_dir.exists():
        return
    
    checkpoints = list(checkpoint_dir.glob("step_*.pt"))
    
    # Sort by step number (descending)
    def get_step(p: Path) -> int:
        try:
            return int(p.stem.split("_")[1])
        except (IndexError, ValueError):
            return 0
    
    checkpoints.sort(key=get_step, reverse=True)
    
    # Keep recent checkpoints
    to_delete = checkpoints[keep_last:]
    
    for ckpt in to_delete:
        ckpt.unlink()
        logger.debug(f"Deleted old checkpoint: {ckpt}")


class CheckpointManager:
    """Manage checkpoint saving and loading."""
    
    def __init__(
        self,
        checkpoint_dir: Path,
        keep_last: int = 5,
        save_best: bool = True,
    ):
        """Initialize checkpoint manager.
        
        Args:
            checkpoint_dir: Directory for checkpoints
            keep_last: Number of recent checkpoints to keep
            save_best: Whether to save best checkpoint
        """
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last = keep_last
        self.save_best = save_best
        self.best_metric = float("-inf")
    
    def save(
        self,
        step: int,
        policy_state: Dict[str, Any],
        value_state: Dict[str, Any],
        optimizer_state: Dict[str, Any],
        config: Dict[str, Any],
        world_model_state: Optional[Dict[str, Any]] = None,
        metrics: Optional[Dict[str, float]] = None,
        metric_for_best: Optional[float] = None,
    ) -> None:
        """Save checkpoint.
        
        Args:
            step: Current step
            policy_state: Policy state dict
            value_state: Value state dict
            optimizer_state: Optimizer state dict
            config: Configuration
            world_model_state: Optional world model state
            metrics: Optional metrics
            metric_for_best: Metric to use for best checkpoint selection
        """
        # Save regular checkpoint
        path = self.checkpoint_dir / f"step_{step}.pt"
        save_checkpoint(
            path=path,
            step=step,
            policy_state=policy_state,
            value_state=value_state,
            optimizer_state=optimizer_state,
            config=config,
            world_model_state=world_model_state,
            metrics=metrics,
        )
        
        # Save best checkpoint
        if self.save_best and metric_for_best is not None:
            if metric_for_best > self.best_metric:
                self.best_metric = metric_for_best
                best_path = self.checkpoint_dir / "best.pt"
                save_checkpoint(
                    path=best_path,
                    step=step,
                    policy_state=policy_state,
                    value_state=value_state,
                    optimizer_state=optimizer_state,
                    config=config,
                    world_model_state=world_model_state,
                    metrics=metrics,
                )
                logger.info(f"New best checkpoint at step {step} with metric {metric_for_best:.4f}")
        
        # Cleanup old checkpoints
        cleanup_old_checkpoints(self.checkpoint_dir, self.keep_last)
    
    def load_latest(self, device: torch.device = torch.device("cpu")) -> Optional[Dict[str, Any]]:
        """Load latest checkpoint.
        
        Args:
            device: Device to load to
            
        Returns:
            Checkpoint dict or None
        """
        latest = get_latest_checkpoint(self.checkpoint_dir)
        if latest is None:
            return None
        return load_checkpoint(latest, device)
    
    def load_best(self, device: torch.device = torch.device("cpu")) -> Optional[Dict[str, Any]]:
        """Load best checkpoint.
        
        Args:
            device: Device to load to
            
        Returns:
            Checkpoint dict or None
        """
        best_path = self.checkpoint_dir / "best.pt"
        if not best_path.exists():
            return None
        return load_checkpoint(best_path, device)
