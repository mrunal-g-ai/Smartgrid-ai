"""
SmartGrid AI -- Real-Time India Electricity Demand & Generation Ingestion
========================================================================
Source:       MERIT India (meritindia.in)
              Operated by POSOCO / National Load Despatch Centre (NLDC),
              Ministry of Power, Government of India.
Endpoint:     https://meritindia.in/Dashboard/BindAllIndiaMap
Data:         Real-time All India Demand Met (MW) + Generation Mix Breakdown
Update cycle: Sub-hourly telemetry (floored to 15-minute IST timestamp).
"""

from __future__ import annotations
import logging
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import requests
from bs4 import BeautifulSoup

MERIT_URL     = "https://meritindia.in/Dashboard/BindAllIndiaMap"
REGION        = "ALL INDIA"
SOURCE_LABEL  = "MERIT India (meritindia.in) / POSOCO / NLDC, Ministry of Power"
CACHE_PATH    = Path(__file__).resolve().parent.parent / "data" / "demand_history.csv"
REQUEST_TIMEOUT = 10
IST           = timezone(timedelta(hours=5, minutes=30))

_logger = logging.getLogger(__name__)

CACHE_COLUMNS = [
    "timestamp", "demand_mw", "region", "source", "source_url", "is_live",
    "thermal_mw", "renewable_mw", "hydro_mw", "nuclear_mw", "gas_mw",
    "storage_mw", "other_mw", "trans_national_mw"
]


def _floor_to_15min(dt: datetime) -> datetime:
    minute = (dt.minute // 15) * 15
    return dt.replace(minute=minute, second=0, microsecond=0)


def _parse_merit_page(soup: BeautifulSoup) -> Dict[str, Optional[float]]:
    metrics = {
        "demand_mw": None,
        "thermal_mw": None,
        "renewable_mw": None,
        "hydro_mw": None,
        "nuclear_mw": None,
        "gas_mw": None,
        "storage_mw": None,
        "other_mw": None,
        "trans_national_mw": None,
    }
    
    key_map = {
        "DEMAND MET": "demand_mw",
        "THERMAL GENERATION": "thermal_mw",
        "RENEWABLE GENERATION": "renewable_mw",
        "HYDRO GENERATION": "hydro_mw",
        "NUCLEAR GENERATION": "nuclear_mw",
        "GAS GENERATION": "gas_mw",
        "STORAGE GENERATION": "storage_mw",
        "OTHER GENERATION": "other_mw",
        "TRANS NATIONAL EXCHANGE": "trans_national_mw",
    }
    
    for card in soup.find_all("div", class_="stat-card"):
        title_div = card.find("div", class_="gen_title_sec")
        counter = card.find("span", class_="counter")
        if title_div and counter:
            raw_title = title_div.get_text(" ", strip=True).upper()
            raw_val = counter.get_text(strip=True).replace(",", "").strip()
            for k_prefix, k_target in key_map.items():
                if k_prefix in raw_title:
                    try:
                        metrics[k_target] = float(raw_val)
                    except ValueError:
                        pass
                    break

    return metrics


def _fetch_merit_india() -> Optional[Dict[str, Any]]:
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Accept": "text/html,application/xhtml+xml",
        "Referer": "https://meritindia.in/",
    }
    try:
        resp = requests.get(MERIT_URL, headers=headers, timeout=REQUEST_TIMEOUT)
        resp.raise_for_status()
    except Exception as exc:
        _logger.warning("MERIT India fetch failed: %s", exc)
        return None

    soup = BeautifulSoup(resp.text, "html.parser")
    metrics = _parse_merit_page(soup)
    demand_mw = metrics.get("demand_mw")
    
    if demand_mw is None or demand_mw <= 0:
        _logger.warning("MERIT India: could not parse DEMAND MET value.")
        return None

    fetched_at = _floor_to_15min(datetime.now(IST))
    return {
        "demand_mw":         demand_mw,
        "fetched_at":        fetched_at,
        "source":            SOURCE_LABEL,
        "source_url":        MERIT_URL,
        "is_live":           True,
        "thermal_mw":        metrics.get("thermal_mw"),
        "renewable_mw":      metrics.get("renewable_mw"),
        "hydro_mw":          metrics.get("hydro_mw"),
        "nuclear_mw":        metrics.get("nuclear_mw"),
        "gas_mw":            metrics.get("gas_mw"),
        "storage_mw":        metrics.get("storage_mw"),
        "other_mw":          metrics.get("other_mw"),
        "trans_national_mw": metrics.get("trans_national_mw"),
    }


def _load_cache() -> pd.DataFrame:
    if CACHE_PATH.exists():
        try:
            df = pd.read_csv(CACHE_PATH, parse_dates=["timestamp"])
            for col in CACHE_COLUMNS:
                if col not in df.columns:
                    df[col] = None
            return df
        except Exception as exc:
            _logger.warning("Cache load error: %s", exc)
    return pd.DataFrame(columns=CACHE_COLUMNS)


def _save_cache(df: pd.DataFrame) -> None:
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CACHE_PATH, index=False)


def _append_observation(df: pd.DataFrame, obs: Dict[str, Any]) -> pd.DataFrame:
    row_data = {
        "timestamp":         obs["fetched_at"].isoformat(),
        "demand_mw":         obs["demand_mw"],
        "region":            REGION,
        "source":            obs["source"],
        "source_url":        obs["source_url"],
        "is_live":           obs["is_live"],
        "thermal_mw":        obs.get("thermal_mw"),
        "renewable_mw":      obs.get("renewable_mw"),
        "hydro_mw":          obs.get("hydro_mw"),
        "nuclear_mw":        obs.get("nuclear_mw"),
        "gas_mw":            obs.get("gas_mw"),
        "storage_mw":        obs.get("storage_mw"),
        "other_mw":          obs.get("other_mw"),
        "trans_national_mw": obs.get("trans_national_mw"),
    }
    new_row = pd.DataFrame([row_data])
    combined = pd.concat([df, new_row], ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp"])
    combined = combined.drop_duplicates(subset=["timestamp"]).sort_values("timestamp")
    return combined.reset_index(drop=True)


def fetch_and_update_demand() -> Dict[str, Any]:
    cache_df = _load_cache()
    obs = _fetch_merit_india()

    if obs is not None:
        updated_df = _append_observation(cache_df, obs)
        _save_cache(updated_df)
        return {
            "status":            "LIVE",
            "demand_mw":         obs["demand_mw"],
            "timestamp":         obs["fetched_at"],
            "region":            REGION,
            "source":            obs["source"],
            "source_url":        obs["source_url"],
            "cache_rows":        len(updated_df),
            "generation_mix": {
                "thermal_mw":        obs.get("thermal_mw"),
                "renewable_mw":      obs.get("renewable_mw"),
                "hydro_mw":          obs.get("hydro_mw"),
                "nuclear_mw":        obs.get("nuclear_mw"),
                "gas_mw":            obs.get("gas_mw"),
                "storage_mw":        obs.get("storage_mw"),
                "other_mw":          obs.get("other_mw"),
                "trans_national_mw": obs.get("trans_national_mw"),
            },
            "error_msg":         None,
        }

    if not cache_df.empty:
        last = cache_df.iloc[-1]
        ts = pd.to_datetime(last["timestamp"])
        if ts.tzinfo is None:
            ts = ts.tz_localize(IST)
        return {
            "status":            "CACHED",
            "demand_mw":         float(last["demand_mw"]),
            "timestamp":         ts,
            "region":            REGION,
            "source":            str(last.get("source", SOURCE_LABEL)),
            "source_url":        str(last.get("source_url", MERIT_URL)),
            "cache_rows":        len(cache_df),
            "generation_mix": {
                "thermal_mw":        last.get("thermal_mw"),
                "renewable_mw":      last.get("renewable_mw"),
                "hydro_mw":          last.get("hydro_mw"),
                "nuclear_mw":        last.get("nuclear_mw"),
                "gas_mw":            last.get("gas_mw"),
                "storage_mw":        last.get("storage_mw"),
                "other_mw":          last.get("other_mw"),
                "trans_national_mw": last.get("trans_national_mw"),
            },
            "error_msg":         "Live fetch failed -- showing last cached observation.",
        }

    return {
        "status":            "UNAVAILABLE",
        "demand_mw":         None,
        "timestamp":         None,
        "region":            REGION,
        "source":            SOURCE_LABEL,
        "source_url":        MERIT_URL,
        "cache_rows":        0,
        "generation_mix":    {},
        "error_msg":         "MERIT India fetch failed and no local cache found.",
    }


def load_demand_history() -> pd.DataFrame:
    return _load_cache()


def get_latest_cached_observation() -> Optional[Dict[str, Any]]:
    df = _load_cache()
    if df.empty:
        return None
    last = df.iloc[-1].to_dict()
    last["timestamp"] = pd.to_datetime(last["timestamp"])
    if last["timestamp"].tzinfo is None:
        last["timestamp"] = last["timestamp"].tz_localize(IST)
    return last
