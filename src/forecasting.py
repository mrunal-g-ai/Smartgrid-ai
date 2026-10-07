import pandas as pd
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
from statsmodels.tsa.statespace.sarimax import SARIMAX
import warnings
warnings.filterwarnings("ignore")

FEATURES = [
    'temperature', 'humidity', 'wind_speed', 'hour', 'day_of_week', 
    'month', 'is_weekend', 'is_holiday', 'load_lag_1h', 'load_lag_24h', 
    'rolling_mean_24h'
]
TARGET = 'energy_consumption'

def split_data(df, test_size=0.2):
    """Chronological train/test split."""
    split_index = int(len(df) * (1 - test_size))
    train = df.iloc[:split_index].copy()
    test = df.iloc[split_index:].copy()
    return train, test

def train_xgboost(train):
    """Train XGBoost model."""
    X_train = train[FEATURES]
    y_train = train[TARGET]
    
    model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
    model.fit(X_train, y_train)
    return model

def train_sarimax(train):
    """
    Train a simplified SARIMAX model.
    Since training SARIMAX on a full year of hourly data is very slow,
    we'll train it on a smaller subset (e.g. last 2 weeks) for demonstration.
    """
    # Use last 14 days of hourly data for SARIMAX to make it fast enough for an app
    train_subset = train.iloc[-24*14:]
    
    exog_features = ['temperature', 'humidity', 'wind_speed', 'is_weekend']
    endog = train_subset[TARGET]
    exog = train_subset[exog_features]
    
    # Order (p, d, q) and seasonal_order (P, D, Q, s) - simplified for speed
    model = SARIMAX(endog, exog=exog, order=(1, 1, 1), seasonal_order=(1, 0, 0, 24))
    fitted_model = model.fit(disp=False)
    return fitted_model, exog_features

def evaluate_model(y_true, y_pred):
    """Calculate MAE, RMSE, R²."""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return mae, rmse, r2

def get_xgboost_metrics(model, test):
    """Evaluate XGBoost on the test set."""
    X_test = test[FEATURES]
    y_test = test[TARGET]
    y_pred = model.predict(X_test)
    return evaluate_model(y_test, y_pred), y_pred

def get_sarimax_metrics(model, exog_features, test):
    """Evaluate SARIMAX on a small portion of the test set for speed."""
    test_subset = test.iloc[:24*7] # 1 week test for SARIMAX
    exog = test_subset[exog_features]
    y_test = test_subset[TARGET]
    
    y_pred = model.forecast(steps=len(y_test), exog=exog)
    return evaluate_model(y_test, y_pred), y_pred, y_test

def forecast_next_24h_xgb(model, last_data, weather_forecast_df):
    """
    Generate a 24-hour forecast using the trained XGBoost model.
    last_data: The last 24 hours of data from the dataset.
    weather_forecast_df: A DataFrame containing 24 hours of weather forecast.
    """
    predictions = []
    current_state = last_data.copy()
    
    # Start time for forecast
    start_time = pd.to_datetime(current_state.iloc[-1]['timestamp']) + pd.Timedelta(hours=1)
    
    for i in range(24):
        pred_time = start_time + pd.Timedelta(hours=i)
        
        # Get weather for this specific hour
        w_row = weather_forecast_df.iloc[i]
        
        # Prepare features for the next hour
        features = {}
        features['temperature'] = w_row['temperature']
        features['humidity'] = w_row['humidity']
        features['wind_speed'] = w_row['wind_speed']
        
        features['hour'] = pred_time.hour
        features['day_of_week'] = pred_time.dayofweek
        features['month'] = pred_time.month
        features['is_weekend'] = int(pred_time.dayofweek in [5, 6])
        
        # Simplistic approximation
        features['is_holiday'] = 0 
        
        features['load_lag_1h'] = current_state.iloc[-1]['energy_consumption']
        features['load_lag_24h'] = current_state.iloc[-24]['energy_consumption']
        
        # Calculate rolling mean using the last 24 values
        features['rolling_mean_24h'] = current_state.iloc[-24:]['energy_consumption'].mean()
        
        # Predict
        X_pred = pd.DataFrame([features])[FEATURES]
        pred_val = model.predict(X_pred)[0]
        
        predictions.append({
            'timestamp': pred_time,
            'predicted_demand': pred_val,
            **features
        })
        
        # Append to current_state to use for next iteration
        current_state = pd.concat([current_state, pd.DataFrame([{
            'timestamp': pred_time,
            'energy_consumption': pred_val,
            'temperature': features['temperature'],
            'humidity': features['humidity'],
            'wind_speed': features['wind_speed']
        }])], ignore_index=True)
        
    return pd.DataFrame(predictions)
