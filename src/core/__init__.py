# Core module - Pure functions, no side effects
# FORBIDDEN: torch, logging, pathlib, any I/O

from .types import StateDefinition, ActionBounds
from .math_utils import normalize_angle, interpolate_curvature
from .physics import tire_grip_coefficient, aero_downforce
