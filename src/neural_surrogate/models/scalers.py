"""Back-compat shim — the canonical scalers now live in common/scalers.py.

Existing imports (`from models.scalers import StandardScaler`) keep working.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "common"))
from scalers import StandardScaler, MinMaxScaler  # noqa: F401
