# PHASE 9 — Explainability & Analysis

## What Decisions Must Be Traceable

Every non-trivial decision the agent makes must be explainable:

| Decision | Question | Required Data |
|----------|----------|---------------|
| Braking point | "Why did it brake here?" | Speed, distance to corner, predicted trajectory |
| Throttle application | "Why partial throttle?" | Tire grip estimate, lateral acceleration |
| Racing line deviation | "Why did it go wide?" | Predicted vs actual trajectory, obstacle avoidance |
| Lift and coast | "Why did it lift?" | Fuel state, tire state, lap time delta |

## Decision Trace Structure

```python
# src/analysis/explainability.py
from dataclasses import dataclass, field
from typing import List, Dict, Optional
import numpy as np
import json

@dataclass
class DecisionTrace:
    """Complete trace of a single decision.
    
    Contains all information needed to explain why
    the agent took a particular action.
    """
    # Timing
    timestamp: float
    step: int
    lap: int
    track_position: float
    
    # Input state (denormalized for readability)
    state: Dict[str, float] = field(default_factory=dict)
    
    # Policy output
    action_mean: np.ndarray = None
    action_std: np.ndarray = None
    action_taken: np.ndarray = None
    
    # Value estimate
    value_estimate: float = 0.0
    
    # World model prediction (if used)
    predicted_next_state: Optional[np.ndarray] = None
    predicted_reward: Optional[float] = None
    
    # Feature importance (gradient-based)
    state_importance: Optional[Dict[str, float]] = None
    
    # Outcome (filled in after step)
    actual_next_state: Optional[Dict[str, float]] = None
    actual_reward: Optional[float] = None
    prediction_error: Optional[float] = None
    
    def to_dict(self) -> dict:
        """Serialize for logging."""
        return {
            "timestamp": self.timestamp,
            "step": self.step,
            "lap": self.lap,
            "track_position": self.track_position,
            "state": self.state,
            "action_mean": self.action_mean.tolist() if self.action_mean is not None else None,
            "action_std": self.action_std.tolist() if self.action_std is not None else None,
            "action_taken": self.action_taken.tolist() if self.action_taken is not None else None,
            "value_estimate": self.value_estimate,
            "state_importance": self.state_importance,
            "actual_reward": self.actual_reward,
            "prediction_error": self.prediction_error,
        }


class DecisionTracer:
    """Collect and analyze decision traces."""
    
    def __init__(self, trace_frequency: int = 100):
        self.trace_frequency = trace_frequency
        self.traces: List[DecisionTrace] = []
        self._step_counter = 0
        
    def should_trace(self) -> bool:
        """Check if current step should be traced."""
        return self._step_counter % self.trace_frequency == 0
        
    def trace_decision(
        self,
        state: np.ndarray,
        policy,
        value_fn,
        world_model=None,
        device="cpu",
    ) -> DecisionTrace:
        """Create trace for current decision.
        
        Args:
            state: Current observation (normalized)
            policy: Policy network
            value_fn: Value network
            world_model: Optional world model
            device: Torch device
            
        Returns:
            DecisionTrace with policy outputs
        """
        import torch
        
        state_tensor = torch.tensor(state, device=device).unsqueeze(0)
        state_tensor.requires_grad_(True)
        
        # Get policy output
        with torch.enable_grad():
            mean, log_std = policy(state_tensor)
            value = value_fn(state_tensor)
            
        # Compute state importance via gradient
        value.backward()
        importance = state_tensor.grad.abs().squeeze().cpu().numpy()
        
        # Denormalize state for readability
        from src.telemetry.normalization import denormalize_state
        from src.telemetry.state import StateDefinition
        
        denorm_state = denormalize_state(state)
        state_def = StateDefinition.from_array(denorm_state)
        
        # Create importance dict
        importance_dict = {
            "velocity": float(importance[0:3].mean()),
            "acceleration": float(importance[3:6].mean()),
            "angular_velocity": float(importance[6:9].mean()),
            "steering": float(importance[9]),
            "throttle": float(importance[10]),
            "brake": float(importance[11]),
            "tire_temps": float(importance[12:16].mean()),
            "tire_wear": float(importance[16:20].mean()),
            "fuel": float(importance[20]),
            "track_position": float(importance[21]),
            "lateral_offset": float(importance[22]),
            "heading_error": float(importance[23]),
            "curvature_ahead": float(importance[24:34].mean()),
        }
        
        trace = DecisionTrace(
            timestamp=0.0,  # Filled by caller
            step=self._step_counter,
            lap=0,  # Filled by caller
            track_position=float(state_def.track_position),
            state={
                "velocity_kmh": float(state_def.velocity[0] * 3.6),
                "lateral_offset_m": float(state_def.lateral_offset),
                "heading_error_deg": float(np.degrees(state_def.heading_error)),
                "tire_wear_avg": float(state_def.tire_wear.mean()),
                "fuel_pct": float(state_def.fuel_mass * 100),
            },
            action_mean=mean.detach().cpu().numpy().squeeze(),
            action_std=log_std.exp().detach().cpu().numpy().squeeze(),
            value_estimate=float(value.detach().cpu().numpy()),
            state_importance=importance_dict,
        )
        
        self._step_counter += 1
        self.traces.append(trace)
        
        return trace
        
    def update_trace_outcome(
        self,
        trace: DecisionTrace,
        action_taken: np.ndarray,
        next_state: np.ndarray,
        reward: float,
    ):
        """Update trace with actual outcome."""
        from src.telemetry.normalization import denormalize_state
        from src.telemetry.state import StateDefinition
        
        trace.action_taken = action_taken
        trace.actual_reward = reward
        
        denorm_next = denormalize_state(next_state)
        next_def = StateDefinition.from_array(denorm_next)
        
        trace.actual_next_state = {
            "velocity_kmh": float(next_def.velocity[0] * 3.6),
            "lateral_offset_m": float(next_def.lateral_offset),
        }
        
        if trace.predicted_next_state is not None:
            trace.prediction_error = float(
                np.mean((trace.predicted_next_state - next_state) ** 2)
            )
            
    def save_traces(self, path: str):
        """Save traces to JSON file."""
        with open(path, "w") as f:
            json.dump([t.to_dict() for t in self.traces], f, indent=2)
            
    def get_summary(self) -> Dict[str, float]:
        """Summarize traced decisions."""
        if not self.traces:
            return {}
            
        return {
            "num_traces": len(self.traces),
            "mean_value_estimate": np.mean([t.value_estimate for t in self.traces]),
            "mean_action_std": np.mean([t.action_std.mean() for t in self.traces if t.action_std is not None]),
            "top_importance_feature": max(
                self.traces[-1].state_importance.items(),
                key=lambda x: x[1]
            )[0] if self.traces[-1].state_importance else None,
        }
```

## Explaining Specific Behaviors

### "Why did the car slow down?"

```python
def explain_slowdown(
    traces: List[DecisionTrace],
    threshold_decel: float = 5.0,  # m/s² deceleration
) -> List[Dict]:
    """Find and explain slowdown events.
    
    Returns list of explanations for each slowdown.
    """
    explanations = []
    
    for i in range(1, len(traces)):
        prev = traces[i-1]
        curr = traces[i]
        
        prev_vel = prev.state.get("velocity_kmh", 0) / 3.6  # to m/s
        curr_vel = curr.state.get("velocity_kmh", 0) / 3.6
        
        decel = (prev_vel - curr_vel) / 0.02  # Assuming 50Hz
        
        if decel > threshold_decel:
            # Determine cause
            importance = curr.state_importance
            
            if importance["curvature_ahead"] > 0.3:
                cause = "Upcoming corner detected"
            elif importance["lateral_offset"] > 0.3:
                cause = "Correcting track position"
            elif importance["tire_wear"] > 0.2:
                cause = "Tire preservation"
            elif curr.action_taken[2] > 0.5:  # Brake > 50%
                cause = "Heavy braking for corner"
            else:
                cause = "Unknown"
                
            explanations.append({
                "step": curr.step,
                "track_position": curr.track_position,
                "deceleration_ms2": decel,
                "cause": cause,
                "brake_input": float(curr.action_taken[2]) if curr.action_taken is not None else None,
                "top_importance": max(importance.items(), key=lambda x: x[1]),
            })
            
    return explanations
```

### "Why did the car push wide?"

```python
def explain_wide_exit(
    traces: List[DecisionTrace],
    threshold_offset: float = 2.0,  # meters from racing line
) -> List[Dict]:
    """Find and explain wide corner exits."""
    explanations = []
    
    for trace in traces:
        offset = abs(trace.state.get("lateral_offset_m", 0))
        
        if offset > threshold_offset:
            importance = trace.state_importance
            
            if importance["tire_wear"] > 0.2:
                cause = "Reduced grip from tire wear"
            elif trace.action_taken is not None and trace.action_taken[1] > 0.8:
                cause = "Aggressive throttle application"
            elif importance["velocity"] > 0.3:
                cause = "Entry speed too high"
            else:
                cause = "Steering input insufficient"
                
            explanations.append({
                "step": trace.step,
                "track_position": trace.track_position,
                "lateral_offset_m": trace.state.get("lateral_offset_m"),
                "cause": cause,
                "steering_input": float(trace.action_taken[0]) if trace.action_taken is not None else None,
            })
            
    return explanations
```

## What Reviewers Care About Seeing

Reviewers (academic, industry, safety) want:

1. **Reproducibility evidence**
   - Exact config used
   - Random seeds
   - Git commit hash
   - Hardware specification

2. **Learning curves**
   - Episode return over time
   - Value loss over time
   - Policy entropy over time
   - NOT just final performance

3. **Failure analysis**
   - When does the agent fail?
   - What states lead to failure?
   - Is failure predictable?

4. **Comparison to baselines**
   - Random policy performance
   - Simple heuristic performance
   - Human performance (if available)

5. **Ablation studies**
   - Which components matter?
   - What happens without world model?
   - What happens with different reward weights?

## Metrics Logger

```python
# src/analysis/metrics.py
import csv
from pathlib import Path
from typing import Dict, List
import json

class MetricsLogger:
    """Structured metrics logging for analysis."""
    
    def __init__(self, log_dir: Path):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        self.csv_path = self.log_dir / "metrics.csv"
        self.json_path = self.log_dir / "metrics.json"
        
        self._metrics_history: List[Dict] = []
        self._csv_initialized = False
        
    def log(self, step: int, metrics: Dict[str, float]):
        """Log metrics for a training step."""
        record = {"step": step, **metrics}
        self._metrics_history.append(record)
        
        # Append to CSV
        if not self._csv_initialized:
            with open(self.csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=record.keys())
                writer.writeheader()
            self._csv_initialized = True
            
        with open(self.csv_path, "a", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=record.keys())
            writer.writerow(record)
            
    def save_summary(self):
        """Save complete metrics history as JSON."""
        with open(self.json_path, "w") as f:
            json.dump(self._metrics_history, f, indent=2)
            
    def get_metric_series(self, metric_name: str) -> List[float]:
        """Get time series of a specific metric."""
        return [m.get(metric_name) for m in self._metrics_history if metric_name in m]
```

## No Dashboards. Only Signal.

Dashboards are:
- Time-consuming to build
- Distracting during development
- Often unused after initial novelty

Instead:
- CSV files that can be plotted with any tool
- JSON files for programmatic analysis
- Text logs for debugging
- Decision traces for explainability

When you need visualization:
```bash
# Quick plot with matplotlib (not in codebase)
python -c "
import pandas as pd
import matplotlib.pyplot as plt
df = pd.read_csv('experiments/latest/logs/metrics.csv')
df.plot(x='step', y=['episode_return', 'value_loss'])
plt.savefig('learning_curve.png')
"
```

Visualization is analysis, not infrastructure.
