#!/usr/bin/env python3
"""Training entry point for F1 Digital Twin."""

import argparse
import logging
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.trainer import Trainer
from src.analysis.logger import ExperimentLogger


def set_global_seed(seed: int, deterministic: bool = False) -> int:
    """Set all random seeds for reproducibility.
    
    Args:
        seed: Random seed
        deterministic: If True, use deterministic algorithms
        
    Returns:
        The seed used
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    if deterministic:
        torch.use_deterministic_algorithms(True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    
    return seed


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to config file
        
    Returns:
        Configuration dictionary
    """
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def apply_overrides(config: dict, overrides: list) -> dict:
    """Apply command-line overrides to config.
    
    Args:
        config: Base configuration
        overrides: List of "key.subkey=value" strings
        
    Returns:
        Modified configuration
    """
    for override in overrides:
        if "=" not in override:
            raise ValueError(f"Invalid override format: {override}. Expected key=value")
        
        key, value = override.split("=", 1)
        keys = key.split(".")
        
        # Navigate to nested key
        d = config
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        
        # Infer type and set value
        try:
            # Try int
            d[keys[-1]] = int(value)
        except ValueError:
            try:
                # Try float
                d[keys[-1]] = float(value)
            except ValueError:
                # Try bool
                if value.lower() in ("true", "false"):
                    d[keys[-1]] = value.lower() == "true"
                else:
                    # Keep as string
                    d[keys[-1]] = value
    
    return config


def main():
    parser = argparse.ArgumentParser(description="Train F1 Digital Twin")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("configs/base.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--device",
        type=str,
        choices=["cpu", "cuda"],
        default=None,
        help="Device to use (overrides config)",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=None,
        help="Maximum training steps (overrides config)",
    )
    parser.add_argument(
        "--override",
        action="append",
        default=[],
        help="Config overrides in format key.subkey=value",
    )
    parser.add_argument(
        "--resume",
        type=Path,
        default=None,
        help="Path to checkpoint to resume from",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Experiment name (overrides config)",
    )
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Apply overrides
    if args.override:
        config = apply_overrides(config, args.override)
    
    if args.device:
        config["device"]["type"] = args.device
    
    if args.max_steps:
        config["training"]["total_timesteps"] = args.max_steps
    
    if args.experiment_name:
        config["experiment"]["name"] = args.experiment_name
    
    # Set seed
    seed = config.get("experiment", {}).get("seed", 42)
    deterministic = config.get("experiment", {}).get("deterministic", False)
    set_global_seed(seed, deterministic)
    
    # Setup logging
    experiment_name = config.get("experiment", {}).get("name", "default")
    exp_logger = ExperimentLogger(experiment_name)
    exp_logger.save_config(config)
    exp_logger.save_git_info()
    
    logger = logging.getLogger("f1_digital_twin")
    logger.info(f"Starting experiment: {experiment_name}")
    logger.info(f"Device: {config['device']['type']}")
    logger.info(f"Seed: {seed}")
    
    # Create trainer
    trainer = Trainer(config)
    
    # Resume from checkpoint if specified
    if args.resume:
        trainer.load_checkpoint(args.resume)
    
    # Train
    try:
        final_metrics = trainer.train()
        logger.info(f"Training complete. Final metrics: {final_metrics}")
    except KeyboardInterrupt:
        logger.info("Training interrupted by user")
    finally:
        # Save final checkpoint
        checkpoint_path = exp_logger.checkpoints_dir / f"step_{trainer.global_step}.pt"
        trainer.save_checkpoint(checkpoint_path)
        
        # Save metrics
        exp_logger.metrics.save_summary()
        
        # Cleanup
        trainer.close()
    
    logger.info("Done")


if __name__ == "__main__":
    main()
