import os
import requests
import pandas as pd
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY")

def get_current_weather(city="Mumbai"):
    """
    Fetch current weather from OpenWeatherMap.
    """
    if not OPENWEATHER_API_KEY:
        return {"status": "FALLBACK WEATHER", "error": "API Key Missing"}

    url = f"http://api.openweathermap.org/data/2.5/weather?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        return {
            "temperature": data["main"]["temp"],
            "humidity": data["main"]["humidity"],
            "wind_speed": data["wind"]["speed"],
            "status": "LIVE WEATHER"
        }
    except Exception as e:
        return {"status": "FALLBACK WEATHER", "error": str(e)}

def get_fallback_weather(df):
    """
    Get the most recent weather from the dataset as a fallback.
    """
    recent = df.iloc[-1]
    return {
        "temperature": recent["temperature"],
        "humidity": recent["humidity"],
        "wind_speed": recent["wind_speed"],
        "status": "FALLBACK WEATHER"
    }

def get_weather_forecast(city="Mumbai"):
    """
    Fetch 24-hour weather forecast from OpenWeatherMap (interpolated to hourly).
    """
    if not OPENWEATHER_API_KEY:
        return {"status": "FALLBACK WEATHER", "error": "API Key Missing"}
        
    url = f"http://api.openweathermap.org/data/2.5/forecast?q={city}&appid={OPENWEATHER_API_KEY}&units=metric"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        
        forecast = []
        for item in data['list']:
            forecast.append({
                'timestamp': pd.to_datetime(item['dt_txt']),
                'temperature': item['main']['temp'],
                'humidity': item['main']['humidity'],
                'wind_speed': item['wind']['speed']
            })
            
        df = pd.DataFrame(forecast).set_index('timestamp')
        df = df.resample('1h').interpolate(method='linear')
        
        # We just need 24 rows, we'll align them to the energy forecast in the energy model
        df = df.reset_index().head(24)
        
        if len(df) < 24:
            raise ValueError("Insufficient forecast data from API")
            
        return {"status": "LIVE WEATHER", "forecast": df}
    except Exception as e:
        return {"status": "FALLBACK WEATHER", "error": str(e)}

def get_fallback_weather_forecast(df, start_time):
    """
    Get fallback 24-hour weather forecast using the last 24 hours of data.
    """
    last_24 = df.iloc[-24:].copy()
    
    timestamps = [start_time + pd.Timedelta(hours=i) for i in range(24)]
    
    forecast_df = pd.DataFrame({
        'timestamp': timestamps,
        'temperature': last_24['temperature'].values,
        'humidity': last_24['humidity'].values,
        'wind_speed': last_24['wind_speed'].values
    })
    return {"status": "FALLBACK WEATHER", "forecast": forecast_df}
