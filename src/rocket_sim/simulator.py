"""RocketPy flight simulation.

The wall-clock timeout is enforced by the PARENT process (see gen_worker.py):
each simulation runs synchronously in a disposable child process which the
parent kills on overrun, so the OS reclaims all solver state and arrays
regardless of what RocketPy/scipy leaked. ``simulate_flight`` is that
synchronous entry point.

``run_simulation`` is the legacy thread-based timeout. It is retained only for
reference/compatibility: a timed-out solve leaves an unkillable daemon thread
running the integrator in the background, which is the source of the overnight
memory leak. Do not use it in the generation path.
"""

import os
import sys
import threading
import warnings

import numpy as np
import rocketpy

import config as cfg


def _build_flight(rocket: rocketpy.Rocket, params: dict) -> rocketpy.Flight:
    """Construct and solve a Flight synchronously. Raises on failure."""
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


def simulate_flight(rocket: rocketpy.Rocket, params: dict) -> rocketpy.Flight | None:
    """Synchronous solve with NO internal timeout.

    Intended to run inside a disposable child process whose wall-clock timeout
    is enforced by the parent (kill on overrun). Returns Flight on success,
    None on error.
    """
    try:
        return _build_flight(rocket, params)
    except Exception:
        return None


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
        try:
            result["flight"] = _build_flight(rocket, params)
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