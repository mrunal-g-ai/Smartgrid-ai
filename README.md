# SmartGrid AI ⚡

## Energy Consumption Forecasting & AI-Based Demand Response

### Project Overview
SmartGrid AI is a comprehensive web application designed to forecast electricity demand and recommend dynamic pricing tariffs. It integrates historical energy consumption data, live weather metrics, machine learning (XGBoost & SARIMAX), and a reinforcement learning (Q-Learning) agent into a single intuitive dashboard.

### Problem Statement
Electricity grids face extreme stress during peak demand hours, risking blackouts and infrastructural strain. Traditional flat-rate pricing does not incentivize consumers to shift their usage to off-peak hours. There is a need for an intelligent system capable of forecasting these demand peaks and recommending demand-response actions (such as simulated dynamic tariffs) to balance the grid load.

### Features
- **Hourly Energy Demand Forecasting:** XGBoost-based predictive model using temporal and weather features.
- **SARIMAX Comparison:** A traditional statistical time-series model for baseline comparison.
- **24-Hour Forecast Generation:** Interactive visualizations predicting tomorrow's energy landscape.
- **Peak Demand Detection:** Statistical categorization (LOW, NORMAL, HIGH, CRITICAL) based on historical percentiles.
- **AI Demand Response (Q-Learning):** A reinforcement learning agent that recommends simulated tariffs (e.g., ₹5/kWh, ₹7/kWh, ₹10/kWh) to encourage demand shifting.
- **Live Weather Integration:** Real-time data from OpenWeatherMap (with local offline fallback).
- **Interactive Dashboard:** Built with Streamlit and Plotly for a responsive, dark-themed control center experience.
- **What-If Simulation:** Test how changing temperatures or humidity levels affect the demand forecast and AI recommendation.

### Technology Stack
- **Python 3.9+**
- **Data & Modeling:** Pandas, NumPy, Scikit-learn, XGBoost, Statsmodels
- **Reinforcement Learning:** Custom tabular Q-Learning via NumPy
- **Frontend & Visualization:** Streamlit, Plotly
- **APIs:** Requests, python-dotenv (OpenWeatherMap)

### Architecture
1. **Data Ingestion & Processing (`src/data_processing.py`)**: Loads, cleans, interpolates missing values, and engineers features (lags, rolling means, temporal markers) from `data/energy_data.csv`.
2. **Weather Service (`src/weather.py`)**: Fetches live metrics via OpenWeatherMap or gracefully degrades to fallback data.
3. **Forecasting Engine (`src/forecasting.py`)**: Trains XGBoost and SARIMAX models on historical data and iteratively forecasts the next 24 hours.
4. **Demand Response Agent (`src/q_learning.py`)**: A Q-Learning agent that learns an optimal policy mapping demand states to tariff actions based on a defined reward system.
5. **Dashboard Controller (`app.py`)**: The central Streamlit application tying the logic together.

### Installation

1. Clone the repository and navigate to the project root:
   ```bash
   cd smartgrid-ai
   ```

2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Set up your environment variables:
   - Copy `.env.example` to `.env`.
   - Add your OpenWeatherMap API key (optional; system will use demo data if missing).

### How to Run
Start the Streamlit application by running:
```bash
streamlit run app.py
```
The application will be accessible at `http://localhost:8501`.

### ML Approach
The primary forecasting model uses **XGBoost Regression**, which captures non-linear relationships between temporal variables (hour, day, month, weekend/holiday status), weather variables (temperature, humidity, wind), and autoregressive features (lagged demand, rolling means). A chronological train/test split prevents data leakage. A secondary **SARIMAX** model is implemented for statistical comparison.

### Q-Learning Approach
The demand response recommendation uses a tabular **Q-Learning** algorithm:
- **States:** LOW, NORMAL, HIGH, CRITICAL (derived from percentile thresholds).
- **Actions:** LOW TARIFF, NORMAL TARIFF, PEAK TARIFF.
- **Rewards:** The environment heavily penalizes grid overloads (low tariffs during critical demand) and exploitative pricing (peak tariffs during low demand), while rewarding load-shifting behavior (peak tariffs during high demand to shave peaks, low tariffs during low demand to fill valleys).

### Limitations
- The simulated tariffs (₹5, ₹7, ₹10) are purely for demonstration and do not reflect real-world electricity costs or regulatory constraints.
- The SARIMAX model is trained on a reduced subset of data to ensure the dashboard trains quickly during demonstrations.
- Long-term multi-step forecasting inherently accumulates error over time without updated lag features.

### Future Scope
- Integration with live smart-meter data streams (e.g., via Kafka or MQTT).
- Upgrading the RL agent to Deep Q-Networks (DQN) for continuous state spaces.
- User-specific dashboards for individual household consumption vs. grid-level view.
