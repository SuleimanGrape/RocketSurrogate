"""Build a RocketPy Rocket from parameter dicts.

RocketPy uses tail-to-nose coordinates: position 0 = tail, increasing forward.
Total length = nose_length + body_length.
"""

import tempfile
import os
import warnings
import numpy as np
import rocketpy

import config as cfg


def _make_thrust_curve_file(motor_class: str, avg_thrust: float, burn_time: float,
                             prop_mass: float) -> str:
    """Create a temporary .eng thrust curve file."""
    peak_thrust = avg_thrust * 1.3
    ramp_time = burn_time * 0.05
    tail_time = burn_time * 0.10
    times = [0.0, ramp_time, ramp_time + 0.1, burn_time - tail_time, burn_time, burn_time + 0.01]
    thrusts = [0.0, peak_thrust, avg_thrust * 1.02, avg_thrust * 0.98, avg_thrust * 0.3, 0.0]
    lines = ["; Synthetic thrust curve\n"]
    for t, f in zip(times, thrusts):
        lines.append(f"{t:.4f}    {f:.4f}\n")
    tmp = tempfile.NamedTemporaryFile(mode="w", suffix=".eng", delete=False)
    tmp.write("".join(lines))
    tmp.close()
    return tmp.name


def build_rocket(params: dict) -> rocketpy.Rocket:
    d_mm = params["diameter_mm"]
    d_m = d_mm / 1000.0
    r_m = d_m / 2.0
    body_l = params["length_m"]
    nose_l = params["nose_length_m"]
    nose_kind = cfg.NOSE_TYPE_MAP.get(params["nose_type"], "tangent")
    dry_mass = params["dry_mass_kg"]
    prop_mass = params["propellant_mass_kg"]
    total_mass = dry_mass + prop_mass

    thrust_file = _make_thrust_curve_file(
        params["motor_class"], params["avg_thrust_N"],
        params["burn_time_s"], prop_mass)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        motor = rocketpy.SolidMotor(
            thrust_source=thrust_file, dry_mass=0.0, dry_inertia=(0, 0, 0),
            nozzle_radius=r_m * 0.4, grain_number=1, grain_density=1815.0,
            grain_outer_radius=r_m * 0.8, grain_initial_inner_radius=r_m * 0.5,
            grain_initial_height=body_l * 0.2, grain_separation=0,
            grains_center_of_mass_position=body_l * 0.1,
            center_of_dry_mass_position=body_l * 0.1,
            burn_time=params["burn_time_s"], nozzle_position=0.0,
            throat_radius=max(0.003, r_m * 0.15), reshape_thrust_curve=False,
            coordinate_system_orientation="nozzle_to_combustion_chamber",
            interpolation_method="linear",
        )
    try:
        os.remove(thrust_file)
    except OSError:
        pass

    rocket = rocketpy.Rocket(
        radius=r_m, mass=total_mass,
        inertia=(0.5 * total_mass * r_m ** 2, 0.5 * total_mass * r_m ** 2,
                 total_mass * body_l ** 2 / 12.0),
        power_off_drag=0.5, power_on_drag=0.4,
        center_of_mass_without_motor=body_l * 0.45 + nose_l * 0.3,
        coordinate_system_orientation="tail_to_nose",
    )
    rocket.add_motor(motor, position=0.0)

    rocket.add_surfaces(
        rocketpy.NoseCone(length=nose_l, kind=nose_kind, base_radius=r_m, name="Nose Cone"),
        positions=body_l)

    n_fins = params["fin_count"]
    root_chord = params["fin_root_chord_m"]
    tip_chord = params["fin_tip_chord_m"]
    span = params["fin_span_m"]
    sweep = params["fin_sweep_m"]
    fin_kwargs = dict(
        n=n_fins, root_chord=root_chord, tip_chord=tip_chord, span=span,
        rocket_radius=r_m, cant_angle=0, name="Fins")
    if sweep > 0:
        fin_kwargs["sweep_length"] = sweep
    rocket.add_surfaces(rocketpy.TrapezoidalFins(**fin_kwargs), positions=0.0)

    rocket.add_parachute(name="Main", cd_s=1.0, trigger="apogee")
    rocket.set_rail_buttons(
        upper_button_position=body_l * 0.3, lower_button_position=0.05, angular_position=45)

    return rocket
