# F1 Autonomous Racing Digital Twin

Production-grade AI system for autonomous F1 racing simulation and control.

## Architecture Overview

Closed decision loop: Telemetry → World Model → Decision → Vehicle Dynamics → Outcome → Learning

## Hardware Requirements

**Development (CPU-only):**
- Intel i5-13500H or equivalent
- 32GB RAM minimum
- No GPU required

**Training (GPU):**
- RTX A4000 (16GB VRAM)
- 128GB system RAM
- Arch Linux

## Quick Start

```bash
# CPU environment setup
cd f1_digital_twin
python -m venv .venv_cpu
source .venv_cpu/bin/activate
pip install -r requirements_cpu.txt

# Verify installation
python -m pytest tests/ -v

# Run CPU training (stub environment)
python scripts/train.py --config configs/base.yaml --device cpu
```

## Repository Structure

```
f1_digital_twin/
├── configs/                 # All configuration (YAML only)
├── src/
│   ├── core/               # Pure functions, no side effects
│   ├── env/                # Environment abstraction layer
│   ├── models/             # World model, policy networks
│   ├── training/           # Training loops, rollout management
│   ├── telemetry/          # State representation, normalization
│   └── analysis/           # Explainability, logging
├── tests/                  # Unit and integration tests
├── scripts/                # Entry points only
└── experiments/            # Isolated experiment outputs
```

## Design Principles

1. Configuration controls everything
2. No simulator code in ML code
3. CPU-first, GPU-ready
4. Reproducibility by default
5. Explainability over performance
