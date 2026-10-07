"""
SmartGrid AI -- Forecasting Engine (XGBoost & SARIMAX Baseline)
===============================================================
24-Hour Electricity Demand Forecasting.

Key principles:
  - Real-time forecast starts 1 hour after the latest observed demand.
  - Future weather (real Open-Meteo forecast) enters every prediction step.
  - No data leakage: recursive lag computation for 24 hours.
  - Robust to cold-start when cache has fewer than 24 historical points.
"""

from __future__ import annotations
import logging
import warnings
from typing import Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX

warnings.filterwarnings("ignore")
_logger = logging.getLogger(__name__)

FEATURES = [
    "hour", "day_of_week", "month", "is_weekend", "is_holiday",
    "temperature", "humidity", "wind_speed",
    "load_lag_1h", "load_lag_24h", "rolling_mean_24h",
]
TARGET = "demand_mw"


def split_data(df: pd.DataFrame, test_size: float = 0.2) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Strict chronological train/test split -- no shuffling."""
    n = len(df)
    split = int(n * (1 - test_size))
    return df.iloc[:split].copy(), df.iloc[split:].copy()


def train_xgboost(train: pd.DataFrame) -> xgb.XGBRegressor:
    """Train XGBoost Regressor on feature-engineered training data."""
    X = train[FEATURES]
    y = train[TARGET] if TARGET in train.columns else train["energy_consumption"]
    model = xgb.XGBRegressor(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    return model


def train_sarimax(train: pd.DataFrame) -> Tuple[Any, pd.DataFrame]:
    """Train a lightweight SARIMAX baseline model."""
    y = train[TARGET] if TARGET in train.columns else train["energy_consumption"]
    # Downsample or take last 720 hours (30 days) to prevent training timeout
    sub_y = y.iloc[-720:] if len(y) > 720 else y
    exog_cols = ["temperature", "humidity"]
    exog = train[exog_cols].iloc[-len(sub_y):]

    model = SARIMAX(
        sub_y,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 0, 24),
        exog=exog,
        enforce_stationarity=False,
        enforce_invertibility=False,
    )
    fitted = model.fit(disp=False, maxiter=50)
    return fitted, exog


def evaluate_model(y_true, y_pred) -> Tuple[float, float, float]:
    mae  = float(mean_absolute_error(y_true, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    r2   = float(r2_score(y_true, y_pred))
    return mae, rmse, r2


def get_xgboost_metrics(model: xgb.XGBRegressor, test: pd.DataFrame):
    X = test[FEATURES]
    y = test[TARGET] if TARGET in test.columns else test["energy_consumption"]
    pred = model.predict(X)
    return evaluate_model(y, pred), pred


def get_sarimax_metrics(model_tuple_or_fitted, exog, test: pd.DataFrame):
    try:
        fitted = model_tuple_or_fitted[0] if isinstance(model_tuple_or_fitted, tuple) else model_tuple_or_fitted
        test_y = test[TARGET] if TARGET in test.columns else test["energy_consumption"]
        sample_len = min(120, len(test))
        test_exog = test[["temperature", "humidity"]].iloc[:sample_len]
        pred = fitted.forecast(steps=sample_len, exog=test_exog)
        mae, rmse, r2 = evaluate_model(test_y.iloc[:sample_len], pred)
        return (mae, rmse, r2), pred, test_y.iloc[:sample_len].values
    except Exception as exc:
        _logger.warning("SARIMAX eval error: %s", exc)
        return (float("nan"), float("nan"), float("nan")), np.array([]), np.array([])


def compare_models(metrics_dict: Dict[str, Any]) -> pd.DataFrame:
    xgb_m = metrics_dict.get("xgb", (0, 0, 0))
    sari_m = metrics_dict.get("sarimax", (0, 0, 0))
    return pd.DataFrame({
        "Model": ["XGBoost Regressor", "SARIMAX (Baseline)"],
        "MAE (MW)": [xgb_m[0], sari_m[0]],
        "RMSE (MW)": [xgb_m[1], sari_m[1]],
        "R² Score": [xgb_m[2], sari_m[2]],
    })


def forecast_next_24h_xgb(model: xgb.XGBRegressor,
                           history_df: pd.DataFrame,
                           weather_forecast_df: pd.DataFrame,
                           latest_obs_demand: Optional[float] = None,
                           latest_obs_timestamp: Optional[Any] = None) -> pd.DataFrame:
    """
    Generate 24 recursive hourly predictions starting after the latest observation.
    Uses real future weather from Open-Meteo for each hour.
    """
    hist = history_df.copy()
    if "energy_consumption" in hist.columns and TARGET not in hist.columns:
        hist[TARGET] = hist["energy_consumption"]

    if TARGET not in hist.columns and "demand_mw" in hist.columns:
        hist[TARGET] = hist["demand_mw"]

    # Extract latest demand
    if latest_obs_demand is not None and latest_obs_demand > 0:
        base_demand = float(latest_obs_demand)
    elif not hist.empty and TARGET in hist.columns:
        base_demand = float(hist[TARGET].iloc[-1])
    else:
        base_demand = 245000.0

    # Extract latest timestamp
    if latest_obs_timestamp is not None:
        last_ts = pd.to_datetime(latest_obs_timestamp)
    elif not hist.empty and "timestamp" in hist.columns:
        last_ts = pd.to_datetime(hist["timestamp"].iloc[-1])
    else:
        last_ts = pd.Timestamp.now(tz="Asia/Kolkata")

    # Round last_ts to the top of the hour for clean forecast grid
    last_hourly_ts = last_ts.replace(minute=0, second=0, microsecond=0)

    # Initialize running history for recursive lags
    if len(hist) >= 24 and TARGET in hist.columns:
        demand_series = list(hist[TARGET].tail(48).values)
    else:
        # Typical diurnal load curve multipliers for India grid relative to current hour
        current_hour = last_hourly_ts.hour
        # Diurnal pattern (relative ratio to daily mean)
        diurnal_ratios = [
            0.82, 0.79, 0.77, 0.76, 0.78, 0.83, 0.90, 0.96, 1.01, 1.03,
            1.02, 1.01, 0.99, 0.98, 1.00, 1.02, 1.04, 1.06, 1.08, 1.07,
            1.04, 0.99, 0.92, 0.86
        ]
        mean_proxy = base_demand / diurnal_ratios[current_hour % 24]
        demand_series = []
        for back_h in range(24, 0, -1):
            h = (current_hour - back_h) % 24
            demand_series.append(mean_proxy * diurnal_ratios[h])
        demand_series[-1] = base_demand

    wf = weather_forecast_df.reset_index(drop=True)
    if len(wf) < 24:
        raise ValueError("weather_forecast_df must contain at least 24 hours.")

    predictions = []

    for i in range(24):
        pred_time = last_hourly_ts + pd.Timedelta(hours=i + 1)
        w_row = wf.iloc[i]

        lag_1h  = demand_series[-1]
        lag_24h = demand_series[-24] if len(demand_series) >= 24 else demand_series[0]
        roll    = float(np.mean(demand_series[-24:]))

        features = {
            "hour":             pred_time.hour,
            "day_of_week":      pred_time.dayofweek,
            "month":            pred_time.month,
            "is_weekend":       int(pred_time.dayofweek in [5, 6]),
            "is_holiday":       0,
            "temperature":      float(w_row["temperature"]),
            "humidity":         float(w_row["humidity"]),
            "wind_speed":       float(w_row["wind_speed"]),
            "load_lag_1h":      float(lag_1h),
            "load_lag_24h":     float(lag_24h),
            "rolling_mean_24h": float(roll),
        }

        X_pred = pd.DataFrame([features])[FEATURES]
        pred_val = float(model.predict(X_pred)[0])

        predictions.append({
            "timestamp":        pred_time,
            "predicted_demand": max(50000.0, pred_val),
            **features,
        })

        demand_series.append(pred_val)

    return pd.DataFrame(predictions)
