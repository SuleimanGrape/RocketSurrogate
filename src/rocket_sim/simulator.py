"""RocketPy flight simulation with threading-based timeout.

Uses threading.Thread for the wall-clock timeout. On timeout the daemon
thread continues running in the background (RocketPy can't be interrupted
from outside), but per-class timeouts (15-120s vs the old uniform 60s)
minimize zombie thread accumulation.

Designed to run inside multiprocessing.Pool workers — threading avoids the
"daemonic processes can't have children" restriction that mp.Process hits.
"""

import os
import sys
import threading
import warnings

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


def run_simulation(rocket: rocketpy.Rocket, params: dict) -> rocketpy.Flight | None:
    """Run simulation with per-class threading timeout.

    Returns Flight on success, None on timeout or error.
    On timeout the daemon thread continues running (no kill mechanism
    available via threads), but per-class timeouts are short enough that
    zombie impact is limited.
    """
    timeout = cfg.SIM_TIMEOUT_BY_CLASS.get(params["motor_class"], cfg.SIM_TIMEOUT_S)
    result = {"flight": None, "error": None}

    def _target():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            try:
                env = rocketpy.Environment(
                    latitude=0, longitude=0, elevation=params["elevation_m"])
                env.set_atmospheric_model(
                    type="custom_atmosphere",
                    pressure=compute_pressure(params["elevation_m"], params["temperature_c"]),
                    temperature=params["temperature_c"] + 273.15,
                    wind_u=params["wind_speed_mps"] * np.cos(np.radians(params["wind_direction_deg"])),
                    wind_v=params["wind_speed_mps"] * np.sin(np.radians(params["wind_direction_deg"])),
                )
                result["flight"] = rocketpy.Flight(
                    rocket=rocket, environment=env,
                    rail_length=params["rail_length_m"],
                    inclination=params["launch_angle_deg"],
                    heading=90, time_overshoot=True,
                    terminate_on_apogee=True, max_time=600, verbose=False,
                )
            except Exception as e:
                result["error"] = e

    thread = threading.Thread(target=_target)
    thread.daemon = True
    thread.start()
    thread.join(timeout=timeout)

    if thread.is_alive():
        # Timed out — thread keeps running as daemon, dies when process exits
        return None
    if result["error"] is not None:
        return None
    return result["flight"]