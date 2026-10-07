import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
DATA_FILE = DATA_DIR / "energy_data.csv"

# Thresholds for demand categories (percentiles)
DEMAND_PERCENTILES = {
    "LOW": 25,
    "NORMAL": 75,
    "HIGH": 90,
    "CRITICAL": 100
}

# TARIFF SIMULATION
TARIFFS = {
    "LOW": 5,     # ₹5/kWh
    "NORMAL": 7,  # ₹7/kWh
    "PEAK": 10    # ₹10/kWh
}

def get_demand_category(value: float, percentiles: dict) -> str:
    """Classify demand based on percentile values."""
    if value < percentiles["LOW"]:
        return "LOW"
    elif value < percentiles["NORMAL"]:
        return "NORMAL"
    elif value < percentiles["HIGH"]:
        return "HIGH"
    else:
        return "CRITICAL"
