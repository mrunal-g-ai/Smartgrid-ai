"""
SmartGrid AI -- Weather Module (Open-Meteo, no API key required)
================================================================
Source:       Open-Meteo (https://open-meteo.com/)
              Free, no authentication required.
              License: CC BY 4.0 (attribution required).

Location:     New Delhi, India
              Latitude:  28.6139
              Longitude: 77.2090
              Timezone:  Asia/Kolkata
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any

import pandas as pd
import requests

_logger = logging.getLogger(__name__)

# Location
WEATHER_LAT     = 28.6139    # New Delhi
WEATHER_LON     = 77.2090
WEATHER_TZ      = "Asia/Kolkata"
SOURCE_LABEL    = "Open-Meteo (open-meteo.com) -- New Delhi, India"
IST             = timezone(timedelta(hours=5, minutes=30))
REQUEST_TIMEOUT = 10
OPENMETEO_URL   = "https://api.open-meteo.com/v1/forecast"


def _build_openmeteo_params():
    return {
        "latitude":         WEATHER_LAT,
        "longitude":        WEATHER_LON,
        "current":          "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
        "hourly":           "temperature_2m,relative_humidity_2m,wind_speed_10m",
        "past_days":        2,
        "forecast_days":    2,
        "timezone":         WEATHER_TZ,
        "wind_speed_unit":  "ms",
    }


def get_weather_openmeteo(start_after: Optional[datetime] = None) -> Dict[str, Any]:
    """
    Fetch current conditions and 24-hour hourly forecast from Open-Meteo.
    """
    if start_after is None:
        start_after = datetime.now(IST)

    try:
        resp = requests.get(OPENMETEO_URL, params=_build_openmeteo_params(), timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        _logger.warning("Open-Meteo fetch failed: %s", exc)
        return _fallback_weather(start_after, str(exc))

    try:
        cur = data["current"]
        current = {
            "temperature":   float(cur["temperature_2m"]),
            "humidity":      float(cur["relative_humidity_2m"]),
            "wind_speed":    float(cur["wind_speed_10m"]),
            "weather_code":  int(cur.get("weather_code", 0)),
            "source":        SOURCE_LABEL,
            "fetched_at":    datetime.now(IST),
            "status":        "LIVE",
        }
    except (KeyError, TypeError) as exc:
        _logger.warning("Open-Meteo current parse error: %s", exc)
        return _fallback_weather(start_after, str(exc))

    try:
        hourly = data["hourly"]
        fc_df = pd.DataFrame({
            "timestamp":   pd.to_datetime(hourly["time"]).tz_localize("Asia/Kolkata"),
            "temperature": hourly["temperature_2m"],
            "humidity":    hourly["relative_humidity_2m"],
            "wind_speed":  hourly["wind_speed_10m"],
        })

        # Select hours strictly AFTER start_after, take next 24
        future_df = fc_df[fc_df["timestamp"] > start_after].head(24).reset_index(drop=True)

        if len(future_df) < 24:
            while len(future_df) < 24:
                last_row = future_df.iloc[-1].copy()
                last_row["timestamp"] = last_row["timestamp"] + pd.Timedelta(hours=1)
                future_df = pd.concat([future_df, pd.DataFrame([last_row])], ignore_index=True)

    except (KeyError, TypeError, IndexError) as exc:
        _logger.warning("Open-Meteo hourly parse error: %s", exc)
        return _fallback_weather(start_after, str(exc))

    return {
        "status":    "LIVE",
        "current":   current,
        "forecast":  future_df,
        "all_hourly": fc_df,
        "error_msg": None,
    }


def _fallback_weather(start_after: datetime, error_msg: str = "") -> Dict[str, Any]:
    now_ist = datetime.now(IST)
    timestamps = [start_after + timedelta(hours=i) for i in range(1, 25)]
    import math
    fc_rows = []
    for ts in timestamps:
        hour = ts.hour
        temp = 28.0 + 5.0 * math.sin((hour - 5) * 3.14159 / 12)
        fc_rows.append({
            "timestamp":   ts,
            "temperature": round(float(temp), 1),
            "humidity":    55.0,
            "wind_speed":  3.0,
        })

    fc_df = pd.DataFrame(fc_rows)

    return {
        "status": "FALLBACK",
        "current": {
            "temperature":  30.0,
            "humidity":     55.0,
            "wind_speed":   3.0,
            "weather_code": 0,
            "source":       "Fallback (Open-Meteo unreachable) -- Seasonal baseline",
            "fetched_at":   now_ist,
            "status":       "FALLBACK",
        },
        "forecast":   fc_df,
        "all_hourly": fc_df,
        "error_msg":  f"Open-Meteo unavailable: {error_msg}",
    }


# Backwards compatibility helper functions
def get_current_weather() -> Dict[str, Any]:
    res = get_weather_openmeteo()
    cur = res["current"]
    cur["status"] = res["status"]
    return cur


def get_weather_forecast(start_after: Optional[datetime] = None) -> Dict[str, Any]:
    res = get_weather_openmeteo(start_after=start_after)
    return {
        "status": res["status"],
        "forecast": res["forecast"],
    }


def get_fallback_weather(df: Optional[pd.DataFrame] = None) -> Dict[str, Any]:
    fb = _fallback_weather(datetime.now(IST))
    return fb["current"]


def get_fallback_weather_forecast(df: Optional[pd.DataFrame] = None, start_ts: Optional[datetime] = None) -> Dict[str, Any]:
    if start_ts is None:
        start_ts = datetime.now(IST)
    fb = _fallback_weather(start_ts)
    return {
        "status": "FALLBACK",
        "forecast": fb["forecast"],
    }
