#!/usr/bin/env python3
"""Validate configuration file."""

import argparse
import sys
from pathlib import Path

import yaml


def validate_config(config: dict) -> list:
    """Validate configuration.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        List of validation errors (empty if valid)
    """
    errors = []
    
    # Required sections
    required_sections = ["experiment", "device", "env", "state", "action", "policy", "training"]
    for section in required_sections:
        if section not in config:
            errors.append(f"Missing required section: {section}")
    
    # Validate experiment
    if "experiment" in config:
        if "seed" not in config["experiment"]:
            errors.append("experiment.seed is required")
    
    # Validate device
    if "device" in config:
        device_type = config["device"].get("type", "")
        if device_type not in ["cpu", "cuda"]:
            errors.append(f"device.type must be 'cpu' or 'cuda', got '{device_type}'")
    
    # Validate env
    if "env" in config:
        env_name = config["env"].get("name", "")
        if env_name not in ["stub", "assetto", "rfactor"]:
            errors.append(f"env.name must be 'stub', 'assetto', or 'rfactor', got '{env_name}'")
        
        max_steps = config["env"].get("max_episode_steps", 0)
        if max_steps <= 0:
            errors.append(f"env.max_episode_steps must be positive, got {max_steps}")
    
    # Validate state
    if "state" in config:
        dim = config["state"].get("dimension", 0)
        if dim != 34:
            errors.append(f"state.dimension must be 34, got {dim}")
    
    # Validate action
    if "action" in config:
        dim = config["action"].get("dimension", 0)
        if dim != 3:
            errors.append(f"action.dimension must be 3, got {dim}")
    
    # Validate policy
    if "policy" in config:
        algorithm = config["policy"].get("algorithm", "")
        if algorithm not in ["ppo", "sac"]:
            errors.append(f"policy.algorithm must be 'ppo' or 'sac', got '{algorithm}'")
    
    # Validate training
    if "training" in config:
        total_steps = config["training"].get("total_timesteps", 0)
        if total_steps <= 0:
            errors.append(f"training.total_timesteps must be positive, got {total_steps}")
        
        if "rollout" in config["training"]:
            num_envs = config["training"]["rollout"].get("num_envs", 0)
            if num_envs <= 0:
                errors.append(f"training.rollout.num_envs must be positive, got {num_envs}")
            
            steps_per_env = config["training"]["rollout"].get("steps_per_env", 0)
            if steps_per_env <= 0:
                errors.append(f"training.rollout.steps_per_env must be positive, got {steps_per_env}")
        
        if "lr_schedule" in config["training"]:
            initial_lr = config["training"]["lr_schedule"].get("initial", 0)
            if initial_lr <= 0:
                errors.append(f"training.lr_schedule.initial must be positive, got {initial_lr}")
    
    return errors


def main():
    parser = argparse.ArgumentParser(description="Validate configuration file")
    parser.add_argument(
        "config",
        type=Path,
        help="Path to configuration file",
    )
    
    args = parser.parse_args()
    
    if not args.config.exists():
        print(f"Error: Config file not found: {args.config}")
        sys.exit(1)
    
    with open(args.config) as f:
        config = yaml.safe_load(f)
    
    errors = validate_config(config)
    
    if errors:
        print("Configuration validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)
    else:
        print("Configuration is valid")
        sys.exit(0)


if __name__ == "__main__":
    main()
