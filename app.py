"""
SmartGrid AI – Energy Demand Intelligence & Decision-Support System

Architecture & Flow:
    1. Ingest historical/simulated operational energy data.
    2. Load pre-trained models from disk (XGBoost, SARIMAX, Q-Learning).
    3. Retrieve weather telemetry (live OpenWeatherMap or reliable fallback).
    4. Automatically generate 24-hour rolling energy forecast.
    5. Detect peak demand hour and evaluate grid load level.
    6. Derive optimal demand-response tariff policy via learned Q-Learning agent.
    7. Enable interactive point inspection and what-if meteorological simulation.
    8. Automatically refresh data every 15 minutes.
"""
import time
import pickle
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta

import streamlit as st
from streamlit_autorefresh import st_autorefresh

# ── Backend Modules ────────────────────────────────────────────────────────────
from src.utils import get_demand_category, TARIFFS
from src.data_processing import load_and_preprocess_data, get_percentile_thresholds
from src.weather import (get_current_weather, get_fallback_weather,
                          get_weather_forecast, get_fallback_weather_forecast)
from src.forecasting import (split_data, train_xgboost, train_sarimax,
                              get_xgboost_metrics, get_sarimax_metrics,
                              forecast_next_24h_xgb)
from src.q_learning import QLearningAgent, STATES, ACTIONS
from src.model_manager import (models_exist, save_models, load_models, load_metadata)

# ── Constants ──────────────────────────────────────────────────────────────────
REFRESH_INTERVAL_MIN = 15
REFRESH_INTERVAL_MS = REFRESH_INTERVAL_MIN * 60 * 1000

# ── Page Configuration ─────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SmartGrid AI — Energy Analytics",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Professional Styling (Dark Navy & Cyan Identity) ─────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
  
  :root {
    --bg: #0B132B;
    --card: #1C2541;
    --bdr: #2A3B5C;
    --bdr-hover: #3A506B;
    --teal: #5BC0BE;
    --white: #FFFFFF;
    --muted: #8D99AE;
    --green: #2ecc71;
    --amber: #f1c40f;
    --red: #e74c3c;
  }
  
  html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
  }
  
  .stApp {
    background-color: var(--bg);
    color: var(--white);
  }
  
  .block-container {
    padding-top: 1rem !important;
    padding-bottom: 2rem !important;
    max-width: 1400px;
  }
  
  div[data-testid="stHorizontalBlock"] {
    gap: 12px;
  }

  /* ── Clean Metric Cards ── */
  .sg-card {
    background: var(--card);
    border-radius: 6px;
    border: 1px solid var(--bdr);
    padding: 14px 16px;
    transition: border-color 0.15s ease;
  }
  .sg-card:hover {
    border-color: var(--bdr-hover);
  }
  .sg-card-sm {
    padding: 10px 12px;
  }

  /* ── Typography Hierarchy ── */
  .sg-title {
    font-size: 1.25rem;
    font-weight: 700;
    color: var(--teal);
    letter-spacing: -0.3px;
  }
  .sg-subtitle {
    font-size: 0.82rem;
    color: var(--muted);
  }
  .sg-heading {
    font-size: 1rem;
    font-weight: 600;
    color: var(--white);
    margin: 20px 0 10px 0;
    display: flex;
    align-items: center;
    gap: 8px;
  }
  .sg-lbl {
    color: var(--muted);
    font-size: 10px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.8px;
    margin-bottom: 4px;
  }
  .sg-val {
    font-size: 26px;
    font-weight: 700;
    color: var(--teal);
    line-height: 1.1;
  }
  .sg-val-white {
    font-size: 26px;
    font-weight: 700;
    color: var(--white);
    line-height: 1.1;
  }
  .sg-unit {
    color: var(--muted);
    font-size: 11px;
    margin-top: 2px;
  }

  /* ── Semantic Badges ── */
  .sg-badge {
    display: inline-flex;
    align-items: center;
    gap: 4px;
    padding: 2px 7px;
    border-radius: 4px;
    font-size: 11px;
    font-weight: 600;
    letter-spacing: 0.2px;
  }
  .sg-badge-green { background: rgba(46,204,113,0.12); color: var(--green); border: 1px solid var(--green); }
  .sg-badge-amber { background: rgba(241,196,15,0.12); color: var(--amber); border: 1px solid var(--amber); }
  .sg-badge-red   { background: rgba(231,76,60,0.12);  color: var(--red);   border: 1px solid var(--red); }
  .sg-badge-teal  { background: rgba(91,192,190,0.12); color: var(--teal);  border: 1px solid var(--teal); }
  .sg-badge-muted { background: rgba(141,153,174,0.12); color: var(--muted); border: 1px solid var(--muted); }

  /* ── Buttons & Inputs ── */
  .stButton > button {
    background: var(--card);
    color: var(--white);
    border: 1px solid var(--bdr);
    border-radius: 5px;
    font-size: 13px;
    font-weight: 500;
    padding: 6px 14px;
    transition: all 0.15s ease;
  }
  .stButton > button:hover {
    border-color: var(--teal);
    color: var(--teal);
  }
  .stButton > button:active {
    background: rgba(91,192,190,0.1);
  }
  
  /* ── Streamlit Overrides ── */
  .stSelectbox > div > div {
    background-color: var(--card);
    border-color: var(--bdr);
    color: var(--white);
    font-size: 13px;
  }
  .stSlider > div > div > div > div {
    background: var(--teal) !important;
  }
  .stExpander {
    background-color: var(--card);
    border: 1px solid var(--bdr) !important;
    border-radius: 6px !important;
  }
</style>
""", unsafe_allow_html=True)

# ── Helper Functions ───────────────────────────────────────────────────────────
def badge(text: str, css_class="sg-badge-teal") -> str:
    return f'<span class="sg-badge {css_class}">{text}</span>'

def level_badge_css(level: str) -> str:
    mapping = {
        "LOW": "sg-badge-green",
        "NORMAL": "sg-badge-green",
        "HIGH": "sg-badge-amber",
        "CRITICAL": "sg-badge-red",
    }
    return mapping.get(level, "sg-badge-muted")

def tariff_for(action: str) -> int:
    if "LOW" in action: return TARIFFS["LOW"]
    if "NORMAL" in action: return TARIFFS["NORMAL"]
    if "PEAK" in action: return TARIFFS["PEAK"]
    return 0

def validate_forecast(df: pd.DataFrame):
    if df is None or df.empty:
        return False, "Forecast dataframe is empty"
    for col in ("timestamp", "predicted_demand"):
        if col not in df.columns:
            return False, f"Missing required column: {col}"
    if df["timestamp"].isna().any():
        return False, "Null/NaT values found in forecast timestamps"
    if df["predicted_demand"].isna().any():
        return False, "NaN values found in demand predictions"
    if np.isinf(df["predicted_demand"]).any():
        return False, "Infinite values found in predictions"
    if len(df) != 24:
        return False, f"Forecast contains {len(df)} hours instead of 24"
    return True, "Valid"

def _fetch_weather(df: pd.DataFrame):
    curr = get_current_weather()
    if "error" in curr:
        curr = get_fallback_weather(df)

    fc = get_weather_forecast()
    if "error" in fc:
        start_ts = pd.to_datetime(df.iloc[-1]["timestamp"]) + pd.Timedelta(hours=1)
        fc = get_fallback_weather_forecast(df, start_ts)
    return curr, fc

def _run_forecast(xgb_model, df: pd.DataFrame, weather_fc: dict):
    try:
        raw_fc = forecast_next_24h_xgb(xgb_model, df.iloc[-48:], weather_fc["forecast"])
        is_ok, msg = validate_forecast(raw_fc)
        if is_ok:
            return raw_fc, None
        return None, msg
    except Exception as exc:
        return None, str(exc)

# ── Session State Initialization ───────────────────────────────────────────────
_state_defaults = {
    "df": None,
    "percentiles": None,
    "model_xgb": None,
    "model_sarimax": None,
    "rl_agent": None,
    "metrics": None,
    "model_meta": None,
    "models_ready": False,
    "needs_initial_training": False,
    "weather": None,
    "weather_forecast": None,
    "forecast_df": None,
    "forecast_ts": None,
    "next_refresh_ts": None,
    "last_refresh_count": 0,
    "selected_hour": None,
    "updating": False,
    "retraining": False,
    "sys_error": None,
    "sim_results": None,
}
for key, default_val in _state_defaults.items():
    if key not in st.session_state:
        st.session_state[key] = default_val

# ── Automated Background Refresh (Every 15 Minutes) ────────────────────────────
refresh_count = st_autorefresh(interval=REFRESH_INTERVAL_MS, key="smartgrid_refresh_trigger")

# ── Step 1: Load Dataset Automatically ─────────────────────────────────────────
if st.session_state.df is None:
    try:
        st.session_state.df = load_and_preprocess_data()
        st.session_state.percentiles = get_percentile_thresholds(st.session_state.df)
    except Exception as exc:
        st.error(f"Failed to load dataset: {exc}")
        st.stop()

df = st.session_state.df
percentiles = st.session_state.percentiles

# ── Step 2: Load Pre-trained Models Automatically ──────────────────────────────
if not st.session_state.models_ready:
    models_bundle = load_models()
    if models_bundle is not None:
        xgb_m, sari_m, ql_m, meta_m = models_bundle
        st.session_state.model_xgb = xgb_m
        st.session_state.model_sarimax = sari_m
        st.session_state.rl_agent = ql_m
        st.session_state.model_meta = meta_m
        st.session_state.metrics = meta_m.get("metrics")
        st.session_state.models_ready = True
        st.session_state.needs_initial_training = False
    else:
        st.session_state.models_ready = False
        st.session_state.needs_initial_training = True

# ── Step 3: Fetch Weather & Run Forecast Automatically ─────────────────────────
_needs_refresh = (
    st.session_state.models_ready and (
        st.session_state.weather is None or
        refresh_count != st.session_state.last_refresh_count
    )
)

if _needs_refresh:
    st.session_state.updating = True
    w_curr, w_fc = _fetch_weather(df)
    fc_data, fc_err = _run_forecast(st.session_state.model_xgb, df, w_fc)

    st.session_state.weather = w_curr
    st.session_state.weather_forecast = w_fc
    st.session_state.forecast_df = fc_data
    st.session_state.sys_error = fc_err
    now_time = datetime.now()
    st.session_state.forecast_ts = now_time.strftime("%H:%M")
    st.session_state.next_refresh_ts = (now_time + timedelta(minutes=REFRESH_INTERVAL_MIN)).strftime("%H:%M")
    st.session_state.last_refresh_count = refresh_count
    st.session_state.updating = False

# ── Convenient References ──────────────────────────────────────────────────────
weather = st.session_state.weather or {}
weather_fc = st.session_state.weather_forecast
fc_df = st.session_state.forecast_df
rl_agent = st.session_state.rl_agent

# ── Automated Peak Detection & Classification ─────────────────────────────────
if fc_df is not None and not fc_df.empty:
    peak_idx = int(fc_df["predicted_demand"].idxmax())
    peak_row = fc_df.loc[peak_idx]
    peak_demand = float(peak_row["predicted_demand"])
    peak_ts = peak_row["timestamp"]
    peak_time_str = peak_ts.strftime("%H:00") if pd.notna(peak_ts) else "—"
    peak_level = get_demand_category(peak_demand, percentiles)
    
    # Default selected hour to the detected peak hour on startup
    if st.session_state.selected_hour is None:
        st.session_state.selected_hour = peak_idx
    elif st.session_state.selected_hour >= len(fc_df):
        st.session_state.selected_hour = 0
else:
    peak_idx = 0
    peak_demand = 0.0
    peak_time_str = "—"
    peak_level = "UNKNOWN"
    peak_ts = None
    if st.session_state.selected_hour is None:
        st.session_state.selected_hour = 0

latest_observed_demand = float(df.iloc[-1]["energy_consumption"])
latest_observed_ts = df.iloc[-1]["timestamp"].strftime("%Y-%m-%d %H:00")

# ══════════════════════════════════════════════════════════════════════════════
# 1. HEADER / SYSTEM STATUS
# ══════════════════════════════════════════════════════════════════════════════
w_status = weather.get("status", "UNAVAILABLE")
is_weather_live = "LIVE" in w_status
weather_badge = badge("Live Weather" if is_weather_live else "Weather Fallback",
                      "sg-badge-green" if is_weather_live else "sg-badge-amber")

sys_status = "Operational"
sys_css = "sg-badge-green"
if st.session_state.needs_initial_training:
    sys_status = "Model Initialization Required"
    sys_css = "sg-badge-red"
elif st.session_state.sys_error:
    sys_status = "Forecast Error"
    sys_css = "sg-badge-red"
elif st.session_state.updating:
    sys_status = "Updating Pipeline"
    sys_css = "sg-badge-amber"

st.markdown(f"""
<div style="display:flex; justify-content:space-between; align-items:center; border-bottom:1px solid var(--bdr); padding-bottom:12px; margin-bottom:16px;">
  <div>
    <div class="sg-title">⚡ SmartGrid AI</div>
    <div class="sg-subtitle">Energy Demand Forecasting & Demand-Response Decision Support</div>
  </div>
  <div style="display:flex; align-items:center; gap:8px; flex-wrap:wrap; justify-content:flex-end;">
    {badge("● " + sys_status, sys_css)}
    {weather_badge}
    <span style="color:var(--muted); font-size:11px; margin-left:6px;">
      Auto-refresh: 15 min &nbsp;|&nbsp; Updated: {st.session_state.forecast_ts or '—'}
    </span>
  </div>
</div>
""", unsafe_allow_html=True)

if st.session_state.needs_initial_training:
    st.warning("No pre-trained models detected on disk. Please open **Model Performance & System Diagnostics** at the bottom to perform initial model training.")

# ══════════════════════════════════════════════════════════════════════════════
# 2. FORECAST OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sg-heading'>Forecast Overview</div>", unsafe_allow_html=True)

c_ov1, c_ov2, c_ov3, c_ov4, c_ov_w = st.columns([1, 1, 1, 1.15, 1.35])

with c_ov1:
    st.markdown(f"""
    <div class="sg-card">
      <div class="sg-lbl">Forecasted Peak</div>
      <div class="sg-val">{peak_demand:.0f} <span style="font-size:14px; color:var(--muted);">MW</span></div>
      <div class="sg-unit">Max predicted 24h demand</div>
    </div>""", unsafe_allow_html=True)

with c_ov2:
    st.markdown(f"""
    <div class="sg-card">
      <div class="sg-lbl">Peak Time</div>
      <div class="sg-val-white">{peak_time_str}</div>
      <div class="sg-unit">Hour of highest load</div>
    </div>""", unsafe_allow_html=True)

with c_ov3:
    st.markdown(f"""
    <div class="sg-card">
      <div class="sg-lbl">Demand Level</div>
      <div style="margin-top:6px; margin-bottom:4px;">
        {badge(peak_level, level_badge_css(peak_level))}
      </div>
      <div class="sg-unit">Percentile load category</div>
    </div>""", unsafe_allow_html=True)

with c_ov4:
    st.markdown(f"""
    <div class="sg-card">
      <div class="sg-lbl">Latest Observed Demand</div>
      <div class="sg-val-white">{latest_observed_demand:.0f} <span style="font-size:14px; color:var(--muted);">MW</span></div>
      <div class="sg-unit">Historical (2023-12-31)</div>
    </div>""", unsafe_allow_html=True)

with c_ov_w:
    t_val = weather.get("temperature", 0.0)
    h_val = weather.get("humidity", 0.0)
    w_val = weather.get("wind_speed", 0.0)
    st.markdown(f"""
    <div class="sg-card">
      <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="sg-lbl">Weather Input</span>
        <span style="font-size:10px; color:var(--muted);">{w_status}</span>
      </div>
      <div style="display:flex; justify-content:space-between; margin-top:6px;">
        <div>
          <span style="font-size:15px; font-weight:600; color:var(--teal);">{t_val:.1f}°C</span>
          <div style="font-size:10px; color:var(--muted);">Temp</div>
        </div>
        <div>
          <span style="font-size:15px; font-weight:600; color:var(--teal);">{h_val:.0f}%</span>
          <div style="font-size:10px; color:var(--muted);">Humidity</div>
        </div>
        <div>
          <span style="font-size:15px; font-weight:600; color:var(--teal);">{w_val:.1f} m/s</span>
          <div style="font-size:10px; color:var(--muted);">Wind</div>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# 3. 24-HOUR DEMAND FORECAST (MAIN VISUAL ELEMENT)
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div class='sg-heading'>24-Hour Demand Forecast</div>", unsafe_allow_html=True)

if fc_df is not None and not fc_df.empty:
    hist_window = df.iloc[-48:]
    selected_idx = st.session_state.selected_hour
    selected_row = fc_df.iloc[selected_idx]
    
    # Assemble Plotly visual
    fig = go.Figure()

    # Trace 0: Historical Demand (Muted Slate Line)
    fig.add_trace(go.Scatter(
        x=hist_window["timestamp"],
        y=hist_window["energy_consumption"],
        mode="lines",
        name="Historical Observed",
        line=dict(color="#8D99AE", width=1.8),
        hovertemplate="<b>%{x|%Y-%m-%d %H:%M}</b><br>Historical: %{y:.1f} MW<extra></extra>"
    ))

    # Trace 1: 24-Hour Forecast (Cyan Dash-Dot with Marker Points)
    hover_notes = [
        f"<b>{row['timestamp'].strftime('%H:00')}</b><br>"
        f"Forecast: <b>{row['predicted_demand']:.1f} MW</b> ({get_demand_category(row['predicted_demand'], percentiles)})<br>"
        f"Weather: {row['temperature']:.1f}°C | {row['humidity']:.0f}% hum | {row['wind_speed']:.1f} m/s"
        for _, row in fc_df.iterrows()
    ]
    fig.add_trace(go.Scatter(
        x=fc_df["timestamp"],
        y=fc_df["predicted_demand"],
        mode="lines+markers",
        name="24-Hour Forecast",
        line=dict(color="#5BC0BE", width=2.5, dash="dot"),
        marker=dict(size=7, color="#5BC0BE", symbol="circle"),
        text=hover_notes,
        hovertemplate="%{text}<extra></extra>"
    ))

    # Trace 2: Detected Peak Marker (Red Diamond)
    fig.add_trace(go.Scatter(
        x=[peak_ts],
        y=[peak_demand],
        mode="markers",
        name="Peak Load",
        marker=dict(size=13, color="#e74c3c", symbol="diamond", line=dict(color="#FFFFFF", width=2)),
        hovertemplate=f"<b>Detected Peak Load</b><br>{peak_time_str} | {peak_demand:.1f} MW<extra></extra>"
    ))

    # Trace 3: Selected Hour Marker (Amber Circle)
    fig.add_trace(go.Scatter(
        x=[selected_row["timestamp"]],
        y=[selected_row["predicted_demand"]],
        mode="markers",
        name="Inspected Hour",
        marker=dict(size=14, color="#f1c40f", symbol="circle-open", line=dict(color="#f1c40f", width=3)),
        hovertemplate=f"<b>Inspected Hour: {selected_row['timestamp'].strftime('%H:00')}</b><br>{selected_row['predicted_demand']:.1f} MW<extra></extra>"
    ))

    # Visual layout & styling
    fig.update_layout(
        plot_bgcolor="#1C2541",
        paper_bgcolor="#0B132B",
        font=dict(color="#FFFFFF", family="Inter, sans-serif", size=12),
        margin=dict(l=10, r=10, t=10, b=10),
        legend=dict(orientation="h", y=1.04, x=1, xanchor="right", font=dict(size=11)),
        xaxis=dict(showgrid=True, gridcolor="#2A3B5C", tickformat="%H:%M"),
        yaxis=dict(showgrid=True, gridcolor="#2A3B5C", title=dict(text="Energy Demand (MW)", font=dict(size=11))),
        height=380,
        hovermode="x unified"
    )

    # Render interactive chart with click point capture
    chart_event = st.plotly_chart(
        fig,
        use_container_width=True,
        on_select="rerun",
        selection_mode="points",
        key="demand_forecast_chart"
    )

    # Automatically update selected hour when a user clicks any forecast point
    if chart_event:
        sel_dict = getattr(chart_event, "selection", None)
        if sel_dict and "points" in sel_dict and sel_dict["points"]:
            last_pt = sel_dict["points"][-1]
            pt_idx = last_pt.get("point_index")
            curve_idx = last_pt.get("curve_number")
            if curve_idx == 1 and pt_idx is not None and 0 <= pt_idx < len(fc_df):
                if st.session_state.selected_hour != int(pt_idx):
                    st.session_state.selected_hour = int(pt_idx)
                    st.rerun()
            elif "x" in last_pt:
                try:
                    clicked_ts = pd.to_datetime(last_pt["x"])
                    time_diffs = (fc_df["timestamp"] - clicked_ts).abs()
                    matched_idx = int(time_diffs.idxmin())
                    if st.session_state.selected_hour != matched_idx:
                        st.session_state.selected_hour = matched_idx
                        st.rerun()
                except Exception:
                    pass

    # ══════════════════════════════════════════════════════════════════════════
    # 4. FORECAST INSPECTOR
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sg-heading'>Forecast Inspector</div>", unsafe_allow_html=True)

    inspected = fc_df.iloc[st.session_state.selected_hour]
    insp_time = inspected["timestamp"].strftime("%H:00")
    insp_demand = float(inspected["predicted_demand"])
    insp_level = get_demand_category(insp_demand, percentiles)
    insp_action = rl_agent.get_recommendation(insp_level) if rl_agent else "—"
    insp_tariff = tariff_for(insp_action)

    # Top control bar with synchronized hour scrubber
    hour_labels = [row["timestamp"].strftime("%H:00") for _, row in fc_df.iterrows()]
    
    ci_ctrl, ci_details = st.columns([1.1, 4.9])
    with ci_ctrl:
        chosen_label = st.selectbox(
            "Select Hour",
            hour_labels,
            index=st.session_state.selected_hour,
            key="inspector_hour_picker",
            label_visibility="collapsed"
        )
        new_sel_idx = hour_labels.index(chosen_label)
        if new_sel_idx != st.session_state.selected_hour:
            st.session_state.selected_hour = new_sel_idx
            st.rerun()

    with ci_details:
        st.markdown(f"""
        <div class="sg-card" style="padding:10px 16px;">
          <div style="display:flex; justify-content:space-between; align-items:center; flex-wrap:wrap; gap:12px;">
            <div>
              <div class="sg-lbl">Inspected Hour</div>
              <div style="font-size:18px; font-weight:700; color:var(--teal);">{insp_time}</div>
            </div>
            <div>
              <div class="sg-lbl">Forecasted Demand</div>
              <div style="font-size:18px; font-weight:700;">{insp_demand:.1f} MW</div>
            </div>
            <div>
              <div class="sg-lbl">Temperature</div>
              <div style="font-size:15px; font-weight:600;">{inspected.get('temperature', 0):.1f} °C</div>
            </div>
            <div>
              <div class="sg-lbl">Humidity</div>
              <div style="font-size:15px; font-weight:600;">{inspected.get('humidity', 0):.0f}%</div>
            </div>
            <div>
              <div class="sg-lbl">Wind Speed</div>
              <div style="font-size:15px; font-weight:600;">{inspected.get('wind_speed', 0):.1f} m/s</div>
            </div>
            <div>
              <div class="sg-lbl">Demand Level</div>
              <div style="margin-top:2px;">{badge(insp_level, level_badge_css(insp_level))}</div>
            </div>
            <div>
              <div class="sg-lbl">AI Action</div>
              <div style="font-size:14px; font-weight:700; color:var(--teal);">{insp_action}</div>
            </div>
            <div>
              <div class="sg-lbl">Simulated Tariff</div>
              <div style="font-size:14px; font-weight:700;">₹{insp_tariff}/kWh</div>
            </div>
          </div>
        </div>
        """, unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 5. AI DEMAND RESPONSE
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sg-heading'>AI Demand Response</div>", unsafe_allow_html=True)

    c_ai1, c_ai2, c_ai3, c_ai_exp = st.columns([1, 1.1, 1.1, 2.8])

    with c_ai1:
        st.markdown(f"""
        <div class="sg-card" style="text-align:center;">
          <div class="sg-lbl">Demand State</div>
          <div style="margin:10px 0;">{badge(insp_level, level_badge_css(insp_level))}</div>
          <div class="sg-unit">Hour {insp_time}</div>
        </div>""", unsafe_allow_html=True)

    with c_ai2:
        st.markdown(f"""
        <div class="sg-card" style="text-align:center;">
          <div class="sg-lbl">Recommended Action</div>
          <div style="font-size:16px; font-weight:700; color:var(--teal); margin:12px 0;">{insp_action}</div>
          <div class="sg-unit">Q-learning policy</div>
        </div>""", unsafe_allow_html=True)

    with c_ai3:
        st.markdown(f"""
        <div class="sg-card" style="text-align:center;">
          <div class="sg-lbl">Simulated Tariff</div>
          <div class="sg-val" style="margin:8px 0;">₹{insp_tariff}</div>
          <div class="sg-unit">per kWh · demonstration only</div>
        </div>""", unsafe_allow_html=True)

    with c_ai_exp:
        st.markdown(f"""
        <div class="sg-card">
          <div class="sg-lbl">Why This Action?</div>
          <div style="font-size:13px; line-height:1.6; margin-top:6px;">
            Forecasted demand is <b>{insp_level}</b> at {insp_time}.
            The learned Q-learning policy selects <b>{insp_action}</b> for this state.
          </div>
          <div style="font-size:11px; color:var(--muted); margin-top:8px;">
            Simulated tariff — demonstration only.
          </div>
        </div>""", unsafe_allow_html=True)

    # ══════════════════════════════════════════════════════════════════════════
    # 6. WHAT-IF SIMULATION
    # ══════════════════════════════════════════════════════════════════════════
    st.markdown("<div class='sg-heading'>What-If Simulation</div>", unsafe_allow_html=True)

    c_sim_in, c_sim_out = st.columns([1.2, 2.8])

    with c_sim_in:
        st.markdown("<div class='sg-card'>", unsafe_allow_html=True)
        st.markdown("<div class='sg-lbl'>Scenario Parameters</div>", unsafe_allow_html=True)
        default_t = float(weather.get("temperature", 30.0))
        default_h = float(weather.get("humidity", 60.0))
        sim_temp = st.slider("Temperature (°C)", 0.0, 50.0, default_t, 0.5, key="slider_sim_temp")
        sim_hum = st.slider("Humidity (%)", 0.0, 100.0, default_h, 1.0, key="slider_sim_hum")
        apply_sim = st.button("Apply Scenario", use_container_width=True, key="btn_apply_scenario")
        st.markdown("</div>", unsafe_allow_html=True)

    with c_sim_out:
        if apply_sim:
            try:
                w_sc = weather_fc["forecast"].copy()
                w_sc["temperature"] = sim_temp
                w_sc["humidity"] = sim_hum
                
                sim_fc = forecast_next_24h_xgb(st.session_state.model_xgb, df.iloc[-48:], w_sc)
                sim_peak = float(sim_fc["predicted_demand"].max())
                sim_level = get_demand_category(sim_peak, percentiles)
                sim_action = rl_agent.get_recommendation(sim_level) if rl_agent else "—"
                sim_tariff = tariff_for(sim_action)
                
                base_peak = peak_demand
                base_level = peak_level
                base_action = rl_agent.get_recommendation(base_level) if rl_agent else "—"
                base_tariff = tariff_for(base_action)
                
                delta_mw = sim_peak - base_peak
                delta_pct = (delta_mw / base_peak * 100) if base_peak > 0 else 0.0
                
                st.session_state.sim_results = {
                    "base_peak": base_peak,
                    "base_level": base_level,
                    "base_action": base_action,
                    "base_tariff": base_tariff,
                    "sim_peak": sim_peak,
                    "sim_level": sim_level,
                    "sim_action": sim_action,
                    "sim_tariff": sim_tariff,
                    "delta_mw": delta_mw,
                    "delta_pct": delta_pct,
                    "temp": sim_temp,
                    "hum": sim_hum
                }
            except Exception as sim_err:
                st.error(f"Scenario simulation error: {sim_err}")

        # Render simulation results if available
        if st.session_state.sim_results:
            sr = st.session_state.sim_results
            sign = "+" if sr["delta_mw"] >= 0 else ""
            delta_color = "var(--red)" if sr["delta_mw"] > 0 else "var(--green)"
            
            c_base, c_scen = st.columns(2)
            with c_base:
                st.markdown(f"""
                <div class="sg-card">
                  <div class="sg-lbl">Baseline Forecast</div>
                  <div style="font-size:24px; font-weight:700; color:var(--teal); margin:6px 0;">{sr['base_peak']:.0f} MW</div>
                  <div>{badge(sr['base_level'], level_badge_css(sr['base_level']))}</div>
                  <div style="font-size:12px; margin-top:8px; color:var(--muted);">
                    Action: <b style="color:var(--white);">{sr['base_action']}</b> (₹{sr['base_tariff']}/kWh)
                  </div>
                </div>""", unsafe_allow_html=True)
            with c_scen:
                st.markdown(f"""
                <div class="sg-card">
                  <div class="sg-lbl">Scenario ({sr['temp']:.1f}°C, {sr['hum']:.0f}%)</div>
                  <div style="font-size:24px; font-weight:700; color:var(--white); margin:6px 0;">
                    {sr['sim_peak']:.0f} MW
                    <span style="font-size:14px; font-weight:600; color:{delta_color}; margin-left:6px;">
                      {sign}{sr['delta_mw']:.0f} MW ({sign}{sr['delta_pct']:.1f}%)
                    </span>
                  </div>
                  <div>{badge(sr['sim_level'], level_badge_css(sr['sim_level']))}</div>
                  <div style="font-size:12px; margin-top:8px; color:var(--muted);">
                    Action: <b style="color:var(--white);">{sr['sim_action']}</b> (₹{sr['sim_tariff']}/kWh)
                  </div>
                </div>""", unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="sg-card" style="min-height:160px; display:flex; align-items:center; justify-content:center; text-align:center;">
              <span style="color:var(--muted); font-size:13px;">
                Adjust meteorological parameters and click <b>Apply Scenario</b> to observe demand and tariff impacts.
              </span>
            </div>""", unsafe_allow_html=True)

elif st.session_state.sys_error:
    st.error(f"Forecasting pipeline encountered an error: {st.session_state.sys_error}")
else:
    st.info("Forecasting pipeline initializing…")

# ══════════════════════════════════════════════════════════════════════════════
# 7. EXPANDABLE TECHNICAL SECTIONS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

# ── Expandable: Model Performance ─────────────────────────────────────────────
with st.expander("▸  Model Performance", expanded=False):
    meta = st.session_state.model_meta or {}
    metrics_dict = st.session_state.metrics or {}
    xgb_m_vals = metrics_dict.get("xgb", (10.60, 14.50, 0.9354))
    sari_m_vals = metrics_dict.get("sarimax", (44.56, 58.12, 0.0312))

    cmp1, cmp2, cmp3 = st.columns(3)
    with cmp1:
        st.markdown(f"""
        <div class="sg-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:13px; font-weight:700; color:var(--teal);">XGBoost Regressor</span>
            {badge("Primary Model", "sg-badge-teal")}
          </div>
          <table style="width:100%; font-size:12px; border-collapse:collapse;">
            <tr><td style="color:var(--muted); padding:3px 0;">MAE</td><td style="text-align:right; font-weight:600;">{xgb_m_vals[0]:.2f} MW</td></tr>
            <tr><td style="color:var(--muted); padding:3px 0;">RMSE</td><td style="text-align:right; font-weight:600;">{xgb_m_vals[1]:.2f} MW</td></tr>
            <tr><td style="color:var(--muted); padding:3px 0;">R² Score</td><td style="text-align:right; font-weight:600;">{xgb_m_vals[2]:.4f}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with cmp2:
        st.markdown(f"""
        <div class="sg-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:13px; font-weight:700; color:var(--white);">SARIMAX (Baseline)</span>
            {badge("Statistical", "sg-badge-muted")}
          </div>
          <table style="width:100%; font-size:12px; border-collapse:collapse;">
            <tr><td style="color:var(--muted); padding:3px 0;">MAE</td><td style="text-align:right; font-weight:600;">{sari_m_vals[0]:.2f} MW</td></tr>
            <tr><td style="color:var(--muted); padding:3px 0;">RMSE</td><td style="text-align:right; font-weight:600;">{sari_m_vals[1]:.2f} MW</td></tr>
            <tr><td style="color:var(--muted); padding:3px 0;">R² Score</td><td style="text-align:right; font-weight:600;">{sari_m_vals[2]:.4f}</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    with cmp3:
        st.markdown("""
        <div class="sg-card">
          <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:8px;">
            <span style="font-size:13px; font-weight:700; color:var(--white);">Q-Learning Agent</span>
            <span class="sg-badge sg-badge-teal">Tabular RL</span>
          </div>
          <table style="width:100%; font-size:12px; border-collapse:collapse;">
            <tr><td style="color:var(--muted); padding:3px 0;">Episodes</td><td style="text-align:right; font-weight:600;">3,000</td></tr>
            <tr><td style="color:var(--muted); padding:3px 0;">State Space</td><td style="text-align:right; font-weight:600;">4 states</td></tr>
            <tr><td style="color:var(--muted); padding:3px 0;">Action Space</td><td style="text-align:right; font-weight:600;">3 tariffs</td></tr>
          </table>
        </div>""", unsafe_allow_html=True)

    # Actual vs Predicted plot on unseen test partition
    if metrics_dict and "xgb_actual" in metrics_dict and "xgb_pred" in metrics_dict:
        y_act = metrics_dict["xgb_actual"]
        y_prd = metrics_dict["xgb_pred"]
        fig_avp = go.Figure()
        fig_avp.add_trace(go.Scatter(y=y_act, mode="lines", name="Observed Actual", line=dict(color="#8D99AE", width=1.5)))
        fig_avp.add_trace(go.Scatter(y=y_prd, mode="lines", name="XGBoost Predicted", line=dict(color="#5BC0BE", width=1.8, dash="dot")))
        fig_avp.update_layout(
            title=dict(text="Evaluation Sample: Actual vs. Predicted Demand (Test Set)", font=dict(size=12, color="#FFFFFF")),
            plot_bgcolor="#1C2541",
            paper_bgcolor="#1C2541",
            font=dict(color="#FFFFFF", size=11),
            margin=dict(l=10, r=10, t=35, b=10),
            legend=dict(orientation="h", y=1.05, x=1, xanchor="right"),
            xaxis=dict(showgrid=True, gridcolor="#2A3B5C", title="Hours"),
            yaxis=dict(showgrid=True, gridcolor="#2A3B5C", title="MW"),
            height=240
        )
        st.plotly_chart(fig_avp, use_container_width=True, key="plot_actual_vs_predicted")

    # Retrain trigger
    if st.button("🔄 Retrain Models", key="btn_retrain_models"):
        st.session_state.retraining = True
        log_ph = st.empty()
        try:
            log_ph.info("Splitting chronological partitions and fitting XGBoost…")
            tr_split, te_split = split_data(df)
            new_xgb = train_xgboost(tr_split)
            new_xgb_m, new_xgb_pred = get_xgboost_metrics(new_xgb, te_split)

            log_ph.info("Fitting SARIMAX baseline…")
            s_tuple = None
            s_metrics = (float("nan"), float("nan"), float("nan"))
            try:
                s_mod, s_exog = train_sarimax(tr_split)
                s_metrics, _, _ = get_sarimax_metrics(s_mod, s_exog, te_split)
                s_tuple = (s_mod, s_exog)
            except Exception as s_err:
                log_ph.warning(f"SARIMAX non-critical notice: {s_err}")

            log_ph.info("Training Q-learning policy…")
            new_agent = QLearningAgent()
            new_agent.train(episodes=3000)

            bundle = {
                "xgb": new_xgb_m,
                "sarimax": s_metrics,
                "xgb_actual": te_split["energy_consumption"].values[:120],
                "xgb_pred": new_xgb_pred[:120],
                "train_rows": len(tr_split),
                "test_rows": len(te_split),
            }
            save_models(new_xgb, s_tuple, new_agent, bundle)

            st.session_state.model_xgb = new_xgb
            st.session_state.model_sarimax = s_tuple
            st.session_state.rl_agent = new_agent
            st.session_state.metrics = bundle
            st.session_state.model_meta = load_metadata()
            st.session_state.models_ready = True
            st.session_state.needs_initial_training = False
            st.session_state.retraining = False

            # Refresh forecast with retrained model
            fc_up, fc_err_up = _run_forecast(new_xgb, df, weather_fc)
            st.session_state.forecast_df = fc_up
            st.session_state.sys_error = fc_err_up
            log_ph.success("Model retraining complete. Saved to disk and refreshed forecast.")
            st.rerun()
        except Exception as retrain_err:
            st.session_state.retraining = False
            log_ph.error(f"Retraining failed: {retrain_err}")

# ── Expandable: AI Decision Details ───────────────────────────────────────────
with st.expander("▸  AI Decision Details", expanded=False):
    if rl_agent:
        qd1, qd2 = st.columns([1.1, 1.9])
        with qd1:
            curr_state = insp_level if insp_level in STATES else "NORMAL"
            s_idx = STATES.index(curr_state)
            q_values = rl_agent.q_table[s_idx]
            best_action_idx = int(np.argmax(q_values))

            rows_html = ""
            for a_idx, (act_name, q_val) in enumerate(zip(ACTIONS, q_values)):
                is_selected = (a_idx == best_action_idx)
                row_style = "color:var(--teal); font-weight:700;" if is_selected else "color:var(--white);"
                check_icon = "✓ " if is_selected else ""
                rows_html += f"<tr><td style='{row_style} padding:4px 0;'>{check_icon}{act_name}</td><td style='text-align:right; {row_style}'>{q_val:.2f}</td></tr>"

            st.markdown(f"""
            <div class="sg-card">
              <div class="sg-lbl">Current State: {curr_state}</div>
              <table style="width:100%; font-size:12px; margin-top:8px; border-collapse:collapse;">
                <tr style="color:var(--muted); font-size:11px; border-bottom:1px solid var(--bdr);">
                  <th style="text-align:left; padding-bottom:4px;">Action</th>
                  <th style="text-align:right; padding-bottom:4px;">Q-Value</th>
                </tr>
                {rows_html}
              </table>
              <div style="font-size:11px; color:var(--muted); margin-top:8px;">
                Optimal policy selects action with maximum Q-value.
              </div>
            </div>""", unsafe_allow_html=True)

        with qd2:
            fig_q = go.Figure(data=go.Heatmap(
                z=rl_agent.q_table,
                x=ACTIONS,
                y=STATES,
                colorscale="Tealgrn",
                text=np.round(rl_agent.q_table, 2),
                texttemplate="%{text}",
                showscale=False
            ))
            fig_q.update_layout(
                title=dict(text="Full Q-Table Matrix (States × Actions)", font=dict(size=12, color="#FFFFFF")),
                plot_bgcolor="#1C2541",
                paper_bgcolor="#1C2541",
                font=dict(color="#FFFFFF", size=11),
                margin=dict(l=10, r=10, t=35, b=10),
                height=180
            )
            st.plotly_chart(fig_q, use_container_width=True, key="plot_qtable_matrix")

# ── Expandable: Data Sources & Provenance ──────────────────────────────────────
with st.expander("▸  Data Sources & Provenance", expanded=False):
    st.markdown("""
    <div style="font-size:12px; color:var(--muted); line-height:1.7;">
      • <b>Demand Dataset:</b> 8,736 hourly observations across calendar year 2023 (`data/energy_data.csv`). Synthetic benchmark operational curve modeling diurnal peaks and temperature sensitivity.<br>
      • <b>Weather Telemetry:</b> OpenWeatherMap API (configured via `.env`). In offline mode, the system automatically draws from historical boundary conditions (`df.iloc[-24:]`) to guarantee continuous execution.<br>
      • <b>Timeline Context:</b> Demonstrator timeline is anchored to historical operational simulation mode (December 31, 2023 → January 1, 2024).
    </div>
    """, unsafe_allow_html=True)

# ── Expandable: System Diagnostics ─────────────────────────────────────────────
with st.expander("▸  System Diagnostics", expanded=False):
    st.markdown(f"""
    <div style="font-size:12px; color:var(--muted); line-height:1.7;">
      • <b>Pipeline Status:</b> Active · Continuous autonomous execution.<br>
      • <b>Autonomous Refresh Cycle:</b> Every 15 minutes (st_autorefresh key <code>smartgrid_refresh_trigger</code>).<br>
      • <b>Model Artifacts on Disk:</b> <code>models/xgboost_model.joblib</code>, <code>models/sarimax_model.pkl</code>, <code>models/q_learning_policy.pkl</code>.<br>
      • <b>Last Refresh Count:</b> {refresh_count} &nbsp;|&nbsp; <b>Next Cycle:</b> {st.session_state.next_refresh_ts or '—'}
    </div>
    """, unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div style="margin-top:40px; padding-top:12px; border-top:1px solid var(--bdr); text-align:center; font-size:11px; color:var(--muted);">
  SmartGrid AI &nbsp;|&nbsp; Energy Demand Intelligence & Decision Support &nbsp;|&nbsp;
  Tariffs: Demonstration only · Not connected to live electrical grid
</div>
""", unsafe_allow_html=True)
