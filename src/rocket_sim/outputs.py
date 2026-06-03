"""Extract and structure simulation outputs for JSONL storage."""

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

    try:
        t_flight = float(flight.t_final) if hasattr(flight, 't_final') else float(flight.t_impact) if hasattr(flight, 't_impact') else 0.0
    except Exception:
        t_flight = 0.0

    try:
        landing_v = abs(float(flight.impact_velocity)) if hasattr(flight, 'impact_velocity') and flight.impact_velocity is not None else abs(float(flight.speed(flight.t_final))) if hasattr(flight, 't_final') else 7.0
    except Exception:
        landing_v = 7.0

    return {
        "apogee_m": round(apogee_m, 1),
        "max_velocity_mps": round(max_velocity, 1),
        "max_mach": round(max_mach, 3),
        "max_acceleration_mps2": _safe(lambda: flight.max_acceleration),
        "burnout_altitude_m": _safe(lambda: flight.burn_out_altitude) if hasattr(flight, 'burn_out_altitude') else _find_burnout_alt(flight, params),
        "burnout_velocity_mps": _safe(lambda: flight.burn_out_velocity) if hasattr(flight, 'burn_out_velocity') else 0.0,
        "flight_time_s": round(t_flight, 1),
        "landing_velocity_mps": round(landing_v, 2),
        "stability_margin_calibers": round(sm, 2),
        "rail_exit_velocity_mps": _safe(lambda: flight.out_of_rail_velocity) if hasattr(flight, 'out_of_rail_velocity') else 0.0,
        "max_dynamic_pressure_pa": _safe(lambda: flight.max_dynamic_pressure) if hasattr(flight, 'max_dynamic_pressure') else 0.0,
        "cg_m": round(cg, 4),
        "cp_m": round(cp, 4),
        "motor_class": params["motor_class"],
    }


def _find_burnout_alt(flight: rocketpy.Flight, params: dict) -> float:
    try:
        burn_time = params["burn_time_s"]
        times = np.array(flight.xArray) if hasattr(flight, 'xArray') else None
        if times is None and hasattr(flight, 'solution') and hasattr(flight.solution, 't'):
            times = np.array(flight.solution.t)
        if times is not None:
            alts = np.array(flight.yArray) if hasattr(flight, 'yArray') else None
            if alts is None and hasattr(flight, 'solution') and hasattr(flight.solution, 'y'):
                alts = np.array(flight.solution.y[2]) if len(flight.solution.y) > 2 else None
            if alts is not None and len(alts) > 0:
                idx = np.argmin(np.abs(np.array(times) - burn_time))
                return float(alts[idx])
    except Exception:
        pass
    return 0.0


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
