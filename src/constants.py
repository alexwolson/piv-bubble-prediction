"""
Shared constants for PIV bubble prediction.
"""

# SEN geometry mappings (from Steel-XGBoost)
SEN_GEOMETRY = {
    "10": {"Angle": -15, "Depth": 40},
    "09": {"Angle": 15, "Depth": 40},
    "08": {"Angle": 0, "Depth": 20},
    "07": {"Angle": -15, "Depth": 0},
    "06": {"Angle": 15, "Depth": 0},
}

# Quadrant definitions
# Quadrants are defined relative to the center of the measurement plane
# LL: Lower Left, LQ: Lower Quadrant (left), RQ: Right Quadrant, RR: Lower Right
QUADRANT_NAMES = ["LL", "LQ", "RQ", "RR"]

# Quadrant indices for array operations
QUADRANT_INDICES = {
    "LL": 0,  # Lower Left
    "LQ": 1,  # Lower Quadrant (left)
    "RQ": 2,  # Right Quadrant
    "RR": 3,  # Lower Right
}
