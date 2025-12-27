# PHASE 1 — Repository & Architecture Blueprint

## Directory Tree (Exact)

```
f1_digital_twin/
│
├── configs/
│   ├── base.yaml                 # Default configuration
│   ├── experiments/              # Experiment-specific overrides
│   │   └── .gitkeep
│   └── schemas/
│       └── config_schema.json    # Validation schema
│
├── src/
│   ├── __init__.py
│   │
│   ├── core/                     # PURE — No side effects, no I/O
│   │   ├── __init__.py
│   │   ├── math_utils.py         # Coordinate transforms, interpolation
│   │   ├── physics.py            # Tire model equations, aero calculations
│   │   ├── racing_line.py        # Optimal line computation
│   │   └── types.py              # Dataclasses, type definitions
│   │
│   ├── env/                      # IMPURE — Simulator interface
│   │   ├── __init__.py
│   │   ├── interface.py          # Abstract base class
│   │   ├── stub_env.py           # CPU testing environment
│   │   ├── gym_wrapper.py        # Gymnasium compatibility
│   │   └── normalizers.py        # Observation/action normalization
│   │
│   ├── models/                   # PURE-ISH — Only torch, no I/O
│   │   ├── __init__.py
│   │   ├── world_model.py        # Dynamics prediction
│   │   ├── policy.py             # Actor network
│   │   ├── value.py              # Critic network
│   │   ├── encoder.py            # State encoding
│   │   └── blocks.py             # Reusable network components
│   │
│   ├── training/                 # IMPURE — Orchestration
│   │   ├── __init__.py
│   │   ├── trainer.py            # Main training loop
│   │   ├── rollout.py            # Experience collection
│   │   ├── buffer.py             # Replay/rollout buffer
│   │   ├── ppo.py                # PPO algorithm
│   │   ├── sac.py                # SAC algorithm (optional)
│   │   └── scheduler.py          # Learning rate, entropy scheduling
│   │
│   ├── telemetry/                # PURE — State representation
│   │   ├── __init__.py
│   │   ├── state.py              # State vector definition
│   │   ├── observation.py        # Observation construction
│   │   ├── normalization.py      # Normalization statistics
│   │   └── validation.py         # State bounds checking
│   │
│   └── analysis/                 # IMPURE — Logging, visualization
│       ├── __init__.py
│       ├── logger.py             # Structured logging
│       ├── metrics.py            # Performance metrics
│       ├── explainability.py     # Decision tracing
│       └── checkpointing.py      # Model save/load
│
├── tests/
│   ├── __init__.py
│   ├── conftest.py               # Pytest fixtures
│   ├── test_core/
│   │   ├── test_math_utils.py
│   │   ├── test_physics.py
│   │   └── test_types.py
│   ├── test_env/
│   │   ├── test_stub_env.py
│   │   └── test_normalizers.py
│   ├── test_models/
│   │   ├── test_world_model.py
│   │   ├── test_policy.py
│   │   └── test_shapes.py
│   ├── test_training/
│   │   ├── test_rollout.py
│   │   ├── test_buffer.py
│   │   └── test_ppo.py
│   └── test_integration/
│       └── test_full_loop.py
│
├── scripts/
│   ├── train.py                  # Training entry point
│   ├── evaluate.py               # Evaluation entry point
│   ├── analyze.py                # Post-hoc analysis
│   └── validate_config.py        # Config validation utility
│
├── experiments/                  # OUTPUT — Git-ignored except .gitkeep
│   └── .gitkeep
│
├── requirements_cpu.txt
├── requirements_gpu.txt
├── pyproject.toml
├── setup.py
├── .gitignore
└── README.md
```

## Module Purity Classification

### PURE Modules (No side effects, deterministic given inputs)

| Module | May Import | Must Never Import |
|--------|-----------|-------------------|
| `core.math_utils` | numpy, typing | torch, logging, pathlib |
| `core.physics` | numpy, core.types | torch, any I/O |
| `core.racing_line` | numpy, scipy.interpolate | torch, env.* |
| `core.types` | dataclasses, typing, numpy | anything else |
| `telemetry.state` | numpy, core.types | torch, logging |
| `telemetry.normalization` | numpy | torch |

### PURE-ISH Modules (Deterministic given seeds, no I/O)

| Module | May Import | Must Never Import |
|--------|-----------|-------------------|
| `models.world_model` | torch, core.*, models.blocks | env.*, logging, pathlib |
| `models.policy` | torch, models.blocks | env.*, training.* |
| `models.value` | torch, models.blocks | env.*, training.* |
| `models.encoder` | torch, models.blocks | env.* |
| `models.blocks` | torch | everything else |

### IMPURE Modules (Side effects allowed, controlled)

| Module | May Import | Must Never Import |
|--------|-----------|-------------------|
| `env.interface` | abc, numpy, core.types | models.*, training.* |
| `env.stub_env` | numpy, env.interface, core.* | models.*, torch |
| `env.gym_wrapper` | gymnasium, env.interface | models.* |
| `training.trainer` | everything in src/ | nothing external except torch, numpy |
| `training.rollout` | torch, env.*, models.* | analysis.* |
| `training.buffer` | torch, numpy | env.*, models.* |
| `analysis.logger` | logging, json, pathlib | models.*, training.* |
| `analysis.explainability` | numpy, core.* | training.* |

## Forbidden Import Rules

```python
# THESE IMPORTS ARE FORBIDDEN — CI will fail

# In src/core/*:
import torch          # FORBIDDEN — core is numpy-only
import logging        # FORBIDDEN — core has no side effects
from src.env import * # FORBIDDEN — core does not know about environments

# In src/models/*:
from src.env import *      # FORBIDDEN — models do not know about environments
from src.training import * # FORBIDDEN — models do not know about training
import os, pathlib         # FORBIDDEN — models do no I/O

# In src/env/*:
from src.models import *   # FORBIDDEN — env does not know about neural networks
from src.training import * # FORBIDDEN — env does not know about training

# In src/telemetry/*:
import torch               # FORBIDDEN — telemetry is numpy-only
from src.models import *   # FORBIDDEN — telemetry is pre-model

# EVERYWHERE:
from configs import *      # FORBIDDEN — config is passed, not imported
import cv2, PIL            # FORBIDDEN — no image processing in core system
```

## Config Ownership

| Config Section | Owner Module | May Read | May Write |
|---------------|--------------|----------|-----------|
| `env.*` | `training.trainer` | env.*, training.* | None |
| `model.*` | `training.trainer` | models.* | None |
| `training.*` | `training.trainer` | training.* | None |
| `logging.*` | `analysis.logger` | analysis.* | None |
| `experiment.*` | `scripts/train.py` | All | None |

Config is read-only after startup. No module modifies config at runtime.

## Experiment Isolation Rules

1. Each experiment creates: `experiments/{YYYYMMDD_HHMMSS}_{experiment_name}/`

2. Experiment directory contains:
   ```
   experiments/20250627_143022_baseline/
   ├── config.yaml           # Frozen copy of config used
   ├── git_info.txt          # Commit hash, branch, dirty status
   ├── checkpoints/
   │   ├── step_10000.pt
   │   ├── step_20000.pt
   │   └── best.pt
   ├── logs/
   │   ├── train.log         # Text log
   │   └── metrics.csv       # Structured metrics
   └── analysis/
       └── (generated post-hoc)
   ```

3. Nothing outside experiment directory is modified during training

4. Experiments are never overwritten — timestamp ensures uniqueness

5. Failed experiments are marked: `experiments/20250627_143022_baseline_FAILED/`
