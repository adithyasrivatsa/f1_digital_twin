# Metrics computation

import numpy as np
from typing import Dict, List, Any


def compute_metrics(
    episode_returns: List[float],
    episode_lengths: List[int],
    lap_times: List[float] = None,
) -> Dict[str, float]:
    """Compute summary metrics from episodes.
    
    Args:
        episode_returns: List of episode returns
        episode_lengths: List of episode lengths
        lap_times: Optional list of lap times
        
    Returns:
        Dict of computed metrics
    """
    metrics = {}
    
    if episode_returns:
        metrics["mean_return"] = float(np.mean(episode_returns))
        metrics["std_return"] = float(np.std(episode_returns))
        metrics["min_return"] = float(np.min(episode_returns))
        metrics["max_return"] = float(np.max(episode_returns))
    
    if episode_lengths:
        metrics["mean_length"] = float(np.mean(episode_lengths))
        metrics["std_length"] = float(np.std(episode_lengths))
    
    if lap_times:
        metrics["mean_lap_time"] = float(np.mean(lap_times))
        metrics["std_lap_time"] = float(np.std(lap_times))
        metrics["best_lap_time"] = float(np.min(lap_times))
        metrics["lap_completion_rate"] = len(lap_times) / max(len(episode_returns), 1)
    
    return metrics


def compute_training_metrics(
    policy_losses: List[float],
    value_losses: List[float],
    entropies: List[float],
    kl_divergences: List[float],
) -> Dict[str, float]:
    """Compute training metrics.
    
    Args:
        policy_losses: List of policy losses
        value_losses: List of value losses
        entropies: List of entropy values
        kl_divergences: List of KL divergences
        
    Returns:
        Dict of training metrics
    """
    metrics = {}
    
    if policy_losses:
        metrics["mean_policy_loss"] = float(np.mean(policy_losses))
    
    if value_losses:
        metrics["mean_value_loss"] = float(np.mean(value_losses))
    
    if entropies:
        metrics["mean_entropy"] = float(np.mean(entropies))
    
    if kl_divergences:
        metrics["mean_kl"] = float(np.mean(kl_divergences))
    
    return metrics


def check_policy_health(metrics: Dict[str, float]) -> List[str]:
    """Check for signs of policy collapse or training issues.
    
    Args:
        metrics: Current training metrics
        
    Returns:
        List of warnings (empty if healthy)
    """
    warnings = []
    
    # Entropy collapse
    entropy = metrics.get("entropy", metrics.get("mean_entropy", 1.0))
    if entropy < 0.1:
        warnings.append(f"LOW ENTROPY: {entropy:.3f} - policy may be collapsing")
    
    # KL divergence spike
    kl = metrics.get("kl", metrics.get("mean_kl", 0.0))
    if kl > 0.1:
        warnings.append(f"HIGH KL: {kl:.3f} - policy changing too fast")
    
    # Value function divergence
    explained_var = metrics.get("explained_variance", 1.0)
    if explained_var < 0.0:
        warnings.append(f"NEGATIVE EXPLAINED VARIANCE: {explained_var:.3f}")
    
    # Clip fraction too high
    clip_fraction = metrics.get("clip_fraction", 0.0)
    if clip_fraction > 0.3:
        warnings.append(f"HIGH CLIP FRACTION: {clip_fraction:.2f}")
    
    # Loss explosion
    policy_loss = metrics.get("policy_loss", 0.0)
    if abs(policy_loss) > 100:
        warnings.append(f"POLICY LOSS EXPLOSION: {policy_loss:.2f}")
    
    value_loss = metrics.get("value_loss", 0.0)
    if value_loss > 1000:
        warnings.append(f"VALUE LOSS EXPLOSION: {value_loss:.2f}")
    
    return warnings


def compute_racing_metrics(
    velocities: List[float],
    lateral_offsets: List[float],
    tire_wears: List[float],
    fuel_levels: List[float],
) -> Dict[str, float]:
    """Compute racing-specific metrics.
    
    Args:
        velocities: List of velocities (m/s)
        lateral_offsets: List of lateral offsets (m)
        tire_wears: List of tire wear values [0, 1]
        fuel_levels: List of fuel levels [0, 1]
        
    Returns:
        Dict of racing metrics
    """
    metrics = {}
    
    if velocities:
        metrics["mean_velocity_kmh"] = float(np.mean(velocities) * 3.6)
        metrics["max_velocity_kmh"] = float(np.max(velocities) * 3.6)
    
    if lateral_offsets:
        metrics["mean_lateral_offset"] = float(np.mean(np.abs(lateral_offsets)))
        metrics["max_lateral_offset"] = float(np.max(np.abs(lateral_offsets)))
    
    if tire_wears:
        metrics["final_tire_wear"] = float(np.mean(tire_wears[-1]) if isinstance(tire_wears[-1], (list, np.ndarray)) else tire_wears[-1])
        metrics["tire_wear_rate"] = float((1.0 - metrics["final_tire_wear"]) / max(len(tire_wears), 1))
    
    if fuel_levels:
        metrics["final_fuel"] = float(fuel_levels[-1])
        metrics["fuel_consumption_rate"] = float((1.0 - metrics["final_fuel"]) / max(len(fuel_levels), 1))
    
    return metrics
