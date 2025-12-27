#!/usr/bin/env python3
"""Observe agent behavior without modifying anything.

Runs episodes with current policy (or random) and records telemetry.
Useful for understanding what the car is actually doing.

Usage:
    # Random policy (baseline behavior)
    python scripts/observe.py --config configs/bicycle.yaml --episodes 3
    
    # With trained policy
    python scripts/observe.py --config configs/bicycle.yaml --checkpoint experiments/.../best.pt --episodes 5
    
    # Save telemetry to file
    python scripts/observe.py --config configs/bicycle.yaml --output telemetry.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.env import make_env, GymWrapper
from src.env.telemetry_recorder import TelemetryRecorder
from src.models import GaussianPolicy


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_episode(
    env,
    policy,
    recorder: TelemetryRecorder,
    device: torch.device,
    deterministic: bool = True,
    max_steps: int = 2000,
    verbose: bool = False,
) -> dict:
    """Run single episode and record telemetry.
    
    Returns:
        Episode statistics
    """
    obs, info = env.reset()
    recorder.on_reset(info)
    
    episode_return = 0.0
    episode_length = 0
    
    for step in range(max_steps):
        # Get action
        if policy is not None:
            obs_tensor = torch.tensor(obs, device=device).unsqueeze(0)
            with torch.no_grad():
                action, _ = policy.sample(obs_tensor, deterministic=deterministic)
            action = action.cpu().numpy().squeeze()
        else:
            # Random policy
            action = env.action_space.sample()
        
        # Step environment
        obs, reward, terminated, truncated, info = env.step(action)
        
        # Record
        recorder.on_step(step, action, reward, terminated, info)
        
        episode_return += reward
        episode_length += 1
        
        # Periodic status
        if verbose and step % 200 == 0:
            print(f"  Step {step:4d}: speed={info.get('velocity', 0):.1f} m/s, "
                  f"lateral={info.get('lateral_offset', 0):.2f} m, "
                  f"progress={info.get('lap_progress', 0)*100:.1f}%")
        
        if terminated or truncated:
            break
    
    return {
        "return": episode_return,
        "length": episode_length,
        "terminated": terminated,
        "final_progress": info.get("lap_progress", 0),
    }


def main():
    parser = argparse.ArgumentParser(description="Observe agent behavior")
    parser.add_argument("--config", type=Path, default=Path("configs/bicycle.yaml"))
    parser.add_argument("--checkpoint", type=Path, default=None, help="Policy checkpoint (random if not provided)")
    parser.add_argument("--episodes", type=int, default=3, help="Number of episodes to run")
    parser.add_argument("--output", type=Path, default=None, help="Save telemetry to CSV")
    parser.add_argument("--deterministic", action="store_true", help="Use deterministic actions")
    parser.add_argument("--verbose", action="store_true", help="Print step-by-step info")
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Load config
    config = load_config(args.config)
    
    # Set seed
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    device = torch.device("cpu")
    
    # Create environment
    env = GymWrapper(make_env(config))
    
    # Load policy if provided
    policy = None
    if args.checkpoint is not None:
        print(f"Loading policy from {args.checkpoint}")
        checkpoint = torch.load(args.checkpoint, map_location=device)
        
        state_dim = config.get("state", {}).get("dimension", 34)
        action_dim = config.get("action", {}).get("dimension", 3)
        
        policy = GaussianPolicy(state_dim=state_dim, action_dim=action_dim)
        policy.load_state_dict(checkpoint["policy_state_dict"])
        policy.eval()
    else:
        print("No checkpoint provided - using random policy")
    
    # Create recorder
    recorder = TelemetryRecorder()
    
    # Run episodes
    print(f"\nRunning {args.episodes} episodes...")
    print("-" * 40)
    
    episode_stats = []
    for ep in range(args.episodes):
        print(f"\nEpisode {ep + 1}/{args.episodes}")
        
        stats = run_episode(
            env=env,
            policy=policy,
            recorder=recorder,
            device=device,
            deterministic=args.deterministic,
            verbose=args.verbose,
        )
        
        episode_stats.append(stats)
        print(f"  Return: {stats['return']:.1f}, Length: {stats['length']}, "
              f"Progress: {stats['final_progress']*100:.1f}%, "
              f"Terminated: {stats['terminated']}")
    
    env.close()
    
    # Print summary
    print("\n" + "=" * 40)
    print("EPISODE SUMMARY")
    print("=" * 40)
    returns = [s["return"] for s in episode_stats]
    lengths = [s["length"] for s in episode_stats]
    progress = [s["final_progress"] for s in episode_stats]
    
    print(f"Return:   mean={np.mean(returns):.1f}, std={np.std(returns):.1f}")
    print(f"Length:   mean={np.mean(lengths):.0f}, std={np.std(lengths):.0f}")
    print(f"Progress: mean={np.mean(progress)*100:.1f}%, max={max(progress)*100:.1f}%")
    
    # Print telemetry summary
    recorder.print_summary()
    
    # Save if requested
    if args.output:
        recorder.save(args.output)
        print(f"Telemetry saved to {args.output}")


if __name__ == "__main__":
    main()
