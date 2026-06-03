"""Constants and configuration for the rocket data generator.

Motor specs are based on typical commercial APCP motors.
Parameter ranges match the ROCKET.md specification with tightened
constraints to prevent simulation hangs.
"""

# ── Discrete choices ──────────────────────────────────────────────────────────

BODY_DIAMETERS_MM = [24, 29, 38, 54, 75, 98]
NOSE_TYPES = ["conical", "ogive", "von_karman", "elliptical"]
FIN_COUNTS = [3, 4]
MOTOR_CLASSES = ["D", "E", "F", "G", "H", "I", "J", "K", "L", "M"]

# ── Motor class specifications ────────────────────────────────────────────────
# (prop_min_kg, prop_max_kg, burn_min_s, burn_max_s, thrust_min_N, thrust_max_N)

MOTOR_SPECS = {
    "D":  (0.012, 0.025, 1.2, 2.0,  18,   35),
    "E":  (0.025, 0.050, 1.3, 2.2,  35,   65),
    "F":  (0.050, 0.100, 1.5, 2.5,  65,   130),
    "G":  (0.100, 0.200, 1.8, 3.0,  130,  260),
    "H":  (0.200, 0.400, 2.0, 3.5,  260,  520),
    "I":  (0.400, 0.800, 2.2, 4.0,  520,  1100),
    "J":  (0.800, 1.600, 2.5, 4.5,  1100, 2200),
    "K":  (1.600, 3.200, 3.0, 5.0,  2200, 4400),
    "L":  (3.200, 6.400, 3.5, 6.0,  4400, 8800),
    "M":  (6.400, 12.80, 4.0, 7.0,  8800, 17600),
}

# ── Motor class allowed per body diameter ─────────────────────────────────────
# Prevents small motors in big rockets and overpowered motors in small rockets.

ALLOWED_MOTORS_BY_DIAMETER = {
    24: [0, 1, 2, 3],           # D, E, F, G
    29: [0, 1, 2, 3, 4],        # D, E, F, G, H
    38: [1, 2, 3, 4, 5],        # E, F, G, H, I
    54: [4, 5, 6, 7],           # H, I, J, K
    75: [5, 6, 7, 8],           # I, J, K, L
    98: [6, 7, 8, 9],           # J, K, L, M
}

# ── Continuous parameter ranges ──────────────────────────────────────────────

BODY_LENGTH_MIN_M = 0.5
BODY_LENGTH_MAX_M = 6.0

NOSE_LENGTH_MIN_DIAMETERS = 0.5
NOSE_LENGTH_MAX_DIAMETERS = 5.0

FIN_ROOT_CHORD_MIN_DIAMETERS = 1.0
FIN_ROOT_CHORD_MAX_DIAMETERS = 2.5

FIN_TIP_CHORD_MIN_FRAC = 0.2
FIN_SPAN_MIN_DIAMETERS = 0.5
FIN_SPAN_MAX_DIAMETERS = 2.0
FIN_SWEEP_MAX_FRAC = 1.0
FIN_THICKNESS_MIN_MM = 2.0
FIN_THICKNESS_MAX_MM = 12.0

DRY_MASS_MIN_KG = 0.3
DRY_MASS_MAX_KG = 120.0
DRY_MASS_K_MIN = 200.0
DRY_MASS_K_MAX = 600.0

MAX_FIN_TO_BODY_AREA_RATIO = 2.0

# ── Environment ranges ───────────────────────────────────────────────────────

ELEVATION_MIN_M = 0.0
ELEVATION_MAX_M = 3000.0
TEMPERATURE_MIN_C = -10.0
TEMPERATURE_MAX_C = 40.0
WIND_SPEED_MIN_MS = 0.0
WIND_SPEED_MAX_MS = 15.0
WIND_DIRECTION_MIN_DEG = 0.0
WIND_DIRECTION_MAX_DEG = 360.0
RAIL_LENGTH_MIN_M = 1.0
RAIL_LENGTH_MAX_M = 8.0
LAUNCH_ANGLE_MIN_DEG = 85.0
LAUNCH_ANGLE_MAX_DEG = 90.0

# ── Validity filter thresholds ───────────────────────────────────────────────

STABILITY_MARGIN_MIN_CAL = 0.5
STABILITY_MARGIN_MAX_CAL = 4.0
THRUST_TO_WEIGHT_MIN = 3.0
MAX_MACH = 5.0
MAX_APOGEE_KM = 100.0
SIM_TIMEOUT_S = 60

# ── Nose shape to RocketPy type mapping ──────────────────────────────────────

NOSE_TYPE_MAP = {
    "conical":    "conical",
    "ogive":      "tangent",
    "von_karman": "von karman",
    "elliptical": "elliptical",
}
