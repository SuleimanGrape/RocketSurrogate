"""RocketPy flight simulation with timeout protection.

Prevents hangs via wall-clock timeout (threading) and RocketPy's internal
max_time safety net. Only receives pre-validated parameters.
"""

import os
import sys
import warnings
import threading
import numpy as np
import rocketpy

import config as cfg


def compute_pressure(elevation_m: float, temperature_c: float) -> float:
    """Atmospheric pressure from elevation (barometric formula)."""
    p0, L, T0, g, M, R = 101325.0, 0.0065, 288.15, 9.80665, 0.0289644, 8.31447
    if elevation_m <= 0:
        return p0
    ratio = 1 - (L * elevation_m) / T0
    return p0 * ratio ** (g * M / (R * L)) if ratio > 0 else p0 * 0.5


class SimulationTimeout(Exception):
    pass


def _run_simulation_core(rocket: rocketpy.Rocket, params: dict) -> rocketpy.Flight:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        env = rocketpy.Environment(
            latitude=0, longitude=0, elevation=params["elevation_m"])
        env.set_atmospheric_model(
            type="custom_atmosphere",
            pressure=compute_pressure(params["elevation_m"], params["temperature_c"]),
            temperature=params["temperature_c"] + 273.15,
            wind_u=params["wind_speed_mps"] * np.cos(np.radians(params["wind_direction_deg"])),
            wind_v=params["wind_speed_mps"] * np.sin(np.radians(params["wind_direction_deg"])),
        )
        return rocketpy.Flight(
            rocket=rocket, environment=env,
            rail_length=params["rail_length_m"],
            inclination=params["launch_angle_deg"],
            heading=90, time_overshoot=True,
            terminate_on_apogee=True, max_time=600, verbose=False,
        )


def run_simulation(rocket: rocketpy.Rocket, params: dict) -> rocketpy.Flight | None:
    """Run simulation with wall-clock timeout. Returns Flight or None."""
    timeout = cfg.SIM_TIMEOUT_S
    result = {"flight": None, "error": None}

    def _target():
        try:
            result["flight"] = _run_simulation_core(rocket, params)
        except Exception as e:
            result["error"] = e

    thread = threading.Thread(target=_target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        return None
    if result["error"] is not None:
        return None
    return result["flight"]
