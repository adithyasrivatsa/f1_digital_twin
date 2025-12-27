# Physics calculations
# FORBIDDEN: torch, logging, any I/O
# These are simplified models for validation, not simulation

import numpy as np


def tire_grip_coefficient(
    temperature: float,
    wear: float,
    optimal_temp: float = 90.0,
    temp_sensitivity: float = 0.01,
) -> float:
    """Calculate tire grip coefficient.
    
    Simplified model: grip peaks at optimal temperature,
    decreases with wear and temperature deviation.
    
    Args:
        temperature: Tire temperature in °C
        wear: Tire wear [0, 1] where 1 is new
        optimal_temp: Optimal operating temperature
        temp_sensitivity: Sensitivity to temperature deviation
        
    Returns:
        Grip coefficient [0, 1]
    """
    # Temperature factor: Gaussian around optimal
    temp_factor = np.exp(-temp_sensitivity * (temperature - optimal_temp) ** 2)
    
    # Wear factor: linear degradation
    wear_factor = 0.7 + 0.3 * wear  # 70% grip at zero wear
    
    return float(np.clip(temp_factor * wear_factor, 0.0, 1.0))


def aero_downforce(
    velocity: float,
    downforce_coefficient: float = 3.5,
    frontal_area: float = 1.5,
    air_density: float = 1.225,
) -> float:
    """Calculate aerodynamic downforce.
    
    F = 0.5 * rho * v^2 * Cl * A
    
    Args:
        velocity: Vehicle velocity in m/s
        downforce_coefficient: Lift coefficient (negative for downforce)
        frontal_area: Frontal area in m²
        air_density: Air density in kg/m³
        
    Returns:
        Downforce in Newtons
    """
    return 0.5 * air_density * velocity ** 2 * downforce_coefficient * frontal_area


def aero_drag(
    velocity: float,
    drag_coefficient: float = 1.0,
    frontal_area: float = 1.5,
    air_density: float = 1.225,
) -> float:
    """Calculate aerodynamic drag.
    
    F = 0.5 * rho * v^2 * Cd * A
    
    Args:
        velocity: Vehicle velocity in m/s
        drag_coefficient: Drag coefficient
        frontal_area: Frontal area in m²
        air_density: Air density in kg/m³
        
    Returns:
        Drag force in Newtons
    """
    return 0.5 * air_density * velocity ** 2 * drag_coefficient * frontal_area


def tire_wear_rate(
    slip_ratio: float,
    slip_angle: float,
    load: float,
    base_rate: float = 0.0001,
) -> float:
    """Calculate tire wear rate per timestep.
    
    Wear increases with slip and load.
    
    Args:
        slip_ratio: Longitudinal slip ratio
        slip_angle: Lateral slip angle in radians
        load: Vertical load on tire in Newtons
        base_rate: Base wear rate per timestep
        
    Returns:
        Wear rate [0, 1] per timestep
    """
    slip_factor = 1.0 + abs(slip_ratio) * 10.0 + abs(slip_angle) * 5.0
    load_factor = load / 5000.0  # Normalize by typical F1 tire load
    
    return base_rate * slip_factor * load_factor


def fuel_consumption_rate(
    throttle: float,
    rpm: float,
    base_rate: float = 0.0001,
) -> float:
    """Calculate fuel consumption rate per timestep.
    
    Args:
        throttle: Throttle position [0, 1]
        rpm: Engine RPM
        base_rate: Base consumption rate
        
    Returns:
        Fuel consumed [0, 1] per timestep (fraction of tank)
    """
    throttle_factor = 0.3 + 0.7 * throttle  # Idle consumption + throttle
    rpm_factor = rpm / 15000.0  # Normalize by max RPM
    
    return base_rate * throttle_factor * rpm_factor


def lateral_acceleration_limit(
    velocity: float,
    grip: float,
    downforce: float,
    mass: float = 800.0,
    gravity: float = 9.81,
) -> float:
    """Calculate maximum lateral acceleration.
    
    Args:
        velocity: Vehicle velocity in m/s
        grip: Tire grip coefficient [0, 1]
        downforce: Aerodynamic downforce in Newtons
        mass: Vehicle mass in kg
        gravity: Gravitational acceleration
        
    Returns:
        Maximum lateral acceleration in m/s²
    """
    # Total vertical force = weight + downforce
    vertical_force = mass * gravity + downforce
    
    # Maximum lateral force = grip * vertical force
    max_lateral_force = grip * vertical_force
    
    # Maximum lateral acceleration
    return max_lateral_force / mass
