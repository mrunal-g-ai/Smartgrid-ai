import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from src.utils import DATA_FILE

def generate_synthetic_data(file_path=DATA_FILE, start_date="2023-01-01", days=365):
    """Generate realistic synthetic hourly energy data for 1 year if file doesn't exist."""
    if os.path.exists(file_path):
        return
    
    print("Generating synthetic data...")
    date_rng = pd.date_range(start=start_date, periods=days*24, freq='h')
    df = pd.DataFrame(date_rng, columns=['timestamp'])
    
    # Feature engineering for generation
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Simple holiday approximation (random few days)
    np.random.seed(42)
    holidays = np.random.choice(date_rng.normalize().unique(), size=15, replace=False)
    df['is_holiday'] = df['timestamp'].dt.normalize().isin(holidays).astype(int)
    
    # Temperature: seasonal sine wave + diurnal sine wave + noise
    annual_temp = 25 + 10 * np.sin(2 * np.pi * (df['timestamp'].dt.dayofyear - 100) / 365.25)
    daily_temp = 5 * np.sin(2 * np.pi * (df['hour'] - 6) / 24)
    df['temperature'] = annual_temp + daily_temp + np.random.normal(0, 2, len(df))
    
    # Humidity and Wind speed
    df['humidity'] = 60 + 20 * np.sin(2 * np.pi * df['timestamp'].dt.dayofyear / 365.25) + np.random.normal(0, 5, len(df))
    df['humidity'] = df['humidity'].clip(10, 100)
    df['wind_speed'] = np.random.lognormal(mean=1.5, sigma=0.5, size=len(df)).clip(0, 30)
    
    # Energy consumption
    # Base load
    base_load = 300
    # Daily pattern: peaks at 10:00 and 19:00
    hour_profile = np.array([0.5, 0.45, 0.4, 0.4, 0.45, 0.55, 0.7, 0.85, 0.95, 1.0, 
                             0.95, 0.9, 0.85, 0.85, 0.9, 0.95, 0.98, 1.0, 1.05, 1.0, 
                             0.95, 0.85, 0.75, 0.6])
    daily_pattern = base_load * hour_profile[df['hour']]
    
    # Weekend/Holiday effect (lower demand)
    weekend_effect = np.where(df['is_weekend'] | df['is_holiday'], 0.8, 1.0)
    
    # Temperature effect (cooling degree days)
    temp_effect = np.where(df['temperature'] > 25, (df['temperature'] - 25) * 5, 0)
    
    # Add noise
    noise = np.random.normal(0, 15, len(df))
    
    # Combine
    df['energy_consumption'] = daily_pattern * weekend_effect + temp_effect + noise
    df['energy_consumption'] = df['energy_consumption'].clip(150, 800) # clip to realistic bounds
    
    # Keep required columns only
    cols = ['timestamp', 'energy_consumption', 'temperature', 'humidity', 'wind_speed', 'is_weekend', 'is_holiday']
    df = df[cols]
    
    # Introduce some missing values and duplicates to test robust processing
    # but not too many
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")

def load_and_preprocess_data(file_path=DATA_FILE):
    """Load and preprocess the dataset according to requirements."""
    # Ensure data exists
    generate_synthetic_data(file_path)
    
    # 1. Load dataset
    df = pd.read_csv(file_path)
    
    # 2. Convert timestamp
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # 3. Sort chronologically
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # 4. Remove duplicates
    df = df.drop_duplicates(subset=['timestamp']).reset_index(drop=True)
    
    # 5. Handle missing values
    # First set timestamp as index for interpolation
    df = df.set_index('timestamp')
    df = df.interpolate(method='time')
    df = df.bfill().ffill()
    df = df.reset_index()
    
    # 6. Handle obvious outliers (e.g. negative energy or absurd temperature)
    df['energy_consumption'] = np.where(df['energy_consumption'] < 0, df['energy_consumption'].median(), df['energy_consumption'])
    
    # Feature Engineering
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek
    df['month'] = df['timestamp'].dt.month
    
    # Re-calculate is_weekend/is_holiday just in case
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    
    # Lags and Rolling Means
    df = df.set_index('timestamp')
    df['load_lag_1h'] = df['energy_consumption'].shift(1)
    df['load_lag_24h'] = df['energy_consumption'].shift(24)
    df['rolling_mean_24h'] = df['energy_consumption'].shift(1).rolling(window=24).mean()
    
    df = df.dropna().reset_index()
    
    return df

def get_percentile_thresholds(df):
    """Calculate percentile thresholds for peak demand detection."""
    p25 = df['energy_consumption'].quantile(0.25)
    p75 = df['energy_consumption'].quantile(0.75)
    p90 = df['energy_consumption'].quantile(0.90)
    
    return {
        "LOW": p25,
        "NORMAL": p75,
        "HIGH": p90
    }
