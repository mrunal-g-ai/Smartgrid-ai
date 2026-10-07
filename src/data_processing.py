"""
SmartGrid AI -- Data Processing & Feature Engineering
=====================================================
Builds the feature-engineered DataFrame used by XGBoost and SARIMAX,
calibrated to the All-India national grid scale (MW).

Feature set:
  hour, day_of_week, month, is_weekend, is_holiday,
  temperature, humidity, wind_speed,
  load_lag_1h, load_lag_24h, rolling_mean_24h
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import numpy as np
import pandas as pd

_logger = logging.getLogger(__name__)

BASE_DIR  = Path(__file__).resolve().parent.parent
DATA_DIR  = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "energy_data.csv"
IST       = timezone(timedelta(hours=5, minutes=30))

INDIA_HOLIDAYS = {
    "01-01", "01-26", "03-25", "04-14", "04-18",
    "05-01", "08-15", "10-02", "11-01", "12-25",
}

TARGET   = "demand_mw"
FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "temperature", "humidity", "wind_speed",
    "load_lag_1h", "load_lag_24h", "rolling_mean_24h",
]


def _is_holiday(ts: pd.Timestamp) -> int:
    return int(ts.strftime("%m-%d") in INDIA_HOLIDAYS)


def load_and_preprocess_data(file_path: Optional[Path] = None) -> pd.DataFrame:
    """
    Load historical energy dataset and construct feature matrix.
    Calibrates demand to the All-India National Grid MW scale (150,000 - 350,000 MW).
    """
    target_path = file_path or DATA_FILE
    if not target_path.exists():
        raise FileNotFoundError(f"Training dataset not found at {target_path}")

    df = pd.read_csv(target_path)
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)

    # Scale to All-India MW if needed
    col = "demand_mw" if "demand_mw" in df.columns else "energy_consumption"
    if df[col].max() < 2000.0:
        # Scale 150-400 kW/substation scale to National Grid MW (~150k - 400k MW)
        df["demand_mw"] = df[col] * 1000.0
    else:
        df["demand_mw"] = df[col].astype(float)

    # Ensure backwards compatibility with energy_consumption name
    df["energy_consumption"] = df["demand_mw"]

    # Fill weather missing values if needed
    for wcol, def_val in [("temperature", 30.0), ("humidity", 55.0), ("wind_speed", 3.0)]:
        if wcol not in df.columns:
            df[wcol] = def_val
        else:
            df[wcol] = df[wcol].interpolate(method="linear").bfill().ffill()

    # Time features
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"]  = df["timestamp"].apply(_is_holiday)

    # Lags & Rolling Mean
    df = df.set_index("timestamp")
    df["load_lag_1h"]      = df["demand_mw"].shift(1)
    df["load_lag_24h"]     = df["demand_mw"].shift(24)
    df["rolling_mean_24h"] = df["demand_mw"].shift(1).rolling(window=24, min_periods=6).mean()
    df = df.dropna(subset=["load_lag_1h", "load_lag_24h", "rolling_mean_24h"]).reset_index()

    return df


def get_percentile_thresholds(df: pd.DataFrame) -> Dict[str, float]:
    """
    Compute demand-level percentile thresholds from historical demand.
    """
    col = TARGET if TARGET in df.columns else (
        "demand_mw" if "demand_mw" in df.columns else "energy_consumption"
    )
    return {
        "LOW":    float(df[col].quantile(0.25)),
        "NORMAL": float(df[col].quantile(0.75)),
        "HIGH":   float(df[col].quantile(0.90)),
    }


def build_features_from_history(demand_df: pd.DataFrame,
                                 weather_df: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    df = demand_df[["timestamp", "demand_mw"]].copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    if df["timestamp"].dt.tz is None:
        df["timestamp"] = df["timestamp"].dt.tz_localize(IST)

    df = df.sort_values("timestamp").drop_duplicates("timestamp").reset_index(drop=True)
    df["hour"]        = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["month"]       = df["timestamp"].dt.month
    df["is_weekend"]  = df["day_of_week"].isin([5, 6]).astype(int)
    df["is_holiday"]  = df["timestamp"].apply(_is_holiday)

    if weather_df is not None:
        w = weather_df[["timestamp", "temperature", "humidity", "wind_speed"]].copy()
        w["timestamp"] = pd.to_datetime(w["timestamp"])
        if w["timestamp"].dt.tz is None:
            w["timestamp"] = w["timestamp"].dt.tz_localize(IST)
        w = w.sort_values("timestamp").drop_duplicates("timestamp")
        df = pd.merge_asof(
            df.sort_values("timestamp"),
            w.sort_values("timestamp"),
            on="timestamp",
            direction="nearest",
            tolerance=pd.Timedelta("2h"),
        )
    else:
        df["temperature"] = 30.0
        df["humidity"]    = 55.0
        df["wind_speed"]  = 3.0

    df = df.set_index("timestamp")
    df["load_lag_1h"]      = df["demand_mw"].shift(1)
    df["load_lag_24h"]     = df["demand_mw"].shift(24)
    df["rolling_mean_24h"] = df["demand_mw"].shift(1).rolling(window=24, min_periods=6).mean()
    df = df.dropna(subset=["load_lag_1h", "load_lag_24h", "rolling_mean_24h"]).reset_index()

    return df
