"""Extract and structure simulation outputs for JSONL storage."""

import math
import numpy as np
import rocketpy
from utils import compute_cp_barrowman, stability_margin_calibers


def extract_output(params: dict, flight: rocketpy.Flight) -> dict:
    """Extract output fields from a successful flight."""
    def _safe(getter, default=0.0):
        try:
            return float(getter())
        except Exception:
            return default

    apogee_m = _safe(lambda: flight.apogee)
    max_velocity = _safe(lambda: flight.max_speed)

    try:
        max_mach = float(flight.max_mach_number) if hasattr(flight, 'max_mach_number') and flight.max_mach_number is not None else max_velocity / 340.0
    except Exception:
        max_mach = 0.0

    cg, cp = compute_cp_barrowman(params)
    sm = stability_margin_calibers(cg, cp, params["diameter_mm"])

    # Extract flight trajectory from RocketPy Function objects
    # flight.z is a Function(time -> altitude); flight.vz is Function(time -> vertical_velocity)
    # RocketPy 1.12.1: .x_array = time values (raw numpy), .y_array = dependent values (raw numpy)
    # DO NOT use get_inputs()/get_outputs() — those return string column headers as first element.
    try:
        traj_times = flight.z.x_array
        traj_altitudes = flight.z.y_array
        traj_velocities = flight.vz.y_array
    except Exception:
        traj_times = np.array([])
        traj_altitudes = np.array([])
        traj_velocities = np.array([])

    # Burnout state: find index closest to burn_time_s in trajectory
    burnout_alt, burnout_vel = _find_burnout_state(flight, params, traj_times, traj_altitudes, traj_velocities)

    # Time to apogee: find max altitude index in trajectory
    time_to_apogee = 0.0
    if len(traj_times) > 0 and len(traj_altitudes) > 0:
        apex_idx = np.argmax(traj_altitudes)
        time_to_apogee = float(traj_times[apex_idx])

    # Flight time: last time in trajectory
    # Note: with terminate_on_apogee=True, this is time-to-apogee, not full flight
    t_flight = float(traj_times[-1]) if len(traj_times) > 0 else 0.0

    return {
        "apogee_m": round(apogee_m, 1),
        "max_velocity_mps": round(max_velocity, 1),
        "max_mach": round(max_mach, 3),
        "max_acceleration_mps2": _safe(lambda: flight.max_acceleration),
        "burnout_altitude_m": round(burnout_alt, 1),
        "burnout_velocity_mps": round(burnout_vel, 1),
        "flight_time_s": round(t_flight, 1),
        "time_to_apogee_s": round(time_to_apogee, 1),
        "stability_margin_calibers": round(sm, 2),
        "rail_exit_velocity_mps": _safe(lambda: flight.out_of_rail_velocity) if hasattr(flight, 'out_of_rail_velocity') else 0.0,
        "max_dynamic_pressure_pa": _safe(lambda: flight.max_dynamic_pressure) if hasattr(flight, 'max_dynamic_pressure') else 0.0,
        "cg_m": round(cg, 4),
        "cp_m": round(cp, 4),
        "motor_class": params["motor_class"],
        # Computability label: a successful extract is, by definition, within
        # bounds. Timed-out / not-computable designs are labelled False where
        # their records are emitted (negative-class generation).
        "within_bounds": True,
    }


def _find_burnout_state(
    flight: rocketpy.Flight,
    params: dict,
    traj_times: np.ndarray,
    traj_altitudes: np.ndarray,
    traj_velocities: np.ndarray,
) -> tuple:
    """Find (altitude, velocity) at motor burnout via flight trajectory functions.

    RocketPy 1.12.1 stores trajectory data in Function objects (flight.z, flight.vz).
    Uses the caller-supplied arrays to avoid redundant Function calls.
    """
    try:
        burn_time = params["burn_time_s"]
        if len(traj_times) == 0:
            return 0.0, 0.0
        idx = np.argmin(np.abs(traj_times - burn_time))
        alt = float(traj_altitudes[idx])
        vel = abs(float(traj_velocities[idx]))
        return alt, vel
    except Exception:
        return 0.0, 0.0


def extract_input(params: dict) -> dict:
    """Extract input fields for JSONL storage."""
    return {
        "diameter_mm": int(params["diameter_mm"]),
        "length_m": round(params["length_m"], 3),
        "nose_type": params["nose_type"],
        "nose_length_m": round(params["nose_length_m"], 3),
        "fin_count": int(params["fin_count"]),
        "fin_root_chord_m": round(params["fin_root_chord_m"], 4),
        "fin_tip_chord_m": round(params["fin_tip_chord_m"], 4),
        "fin_span_m": round(params["fin_span_m"], 4),
        "fin_sweep_m": round(params["fin_sweep_m"], 4),
        "fin_thickness_mm": round(params["fin_thickness_mm"], 1),
        "dry_mass_kg": round(params["dry_mass_kg"], 3),
        "motor_class": params["motor_class"],
        "propellant_mass_kg": round(params["propellant_mass_kg"], 4),
        "burn_time_s": round(params["burn_time_s"], 2),
        "avg_thrust_N": round(params["avg_thrust_N"], 1),
        "wind_speed_mps": round(params["wind_speed_mps"], 1),
        "wind_direction_deg": round(params["wind_direction_deg"], 1),
        "elevation_m": round(params["elevation_m"], 0),
        "temperature_c": round(params["temperature_c"], 1),
        "rail_length_m": round(params["rail_length_m"], 1),
        "launch_angle_deg": round(params["launch_angle_deg"], 1),
    }
