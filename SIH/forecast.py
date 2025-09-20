import openmeteo_requests
import pandas as pd
import requests_cache
from retry_requests import retry

# Setup the Open-Meteo API client with cache and retry on error
cache_session = requests_cache.CachedSession('.cache', expire_after=3600)
retry_session = retry(cache_session, retries=5, backoff_factor=0.2)
openmeteo = openmeteo_requests.Client(session=retry_session)

# Open-Meteo API parameters
url = "https://api.open-meteo.com/v1/forecast"
params = {
    "latitude": 20.54,
    "longitude": 84.14,
    "hourly": ["temperature_2m", "rain"],
    "timezone": "Asia/Singapore",
    "forecast_days": 16,
}

# Make the API request
responses = openmeteo.weather_api(url, params=params)

# Process first location
response = responses[0]

hourly = response.Hourly()
hourly_temperature_2m = hourly.Variables(0).ValuesAsNumpy()
hourly_rain = hourly.Variables(1).ValuesAsNumpy()

hourly_data = {
    "date": pd.date_range(
        start=pd.to_datetime(hourly.Time(), unit="s", utc=True),
        end=pd.to_datetime(hourly.TimeEnd(), unit="s", utc=True),
        freq=pd.Timedelta(seconds=hourly.Interval()),
        inclusive="left",
    )
}

hourly_data["temperature_2m"] = hourly_temperature_2m
hourly_data["rain"] = hourly_rain

hourly_dataframe = pd.DataFrame(data=hourly_data)

# --- FILTER FOR LAST FORECAST DAY ---
last_day = hourly_dataframe["date"].dt.date.max()
last_day_data = hourly_dataframe[hourly_dataframe["date"].dt.date == last_day]

# --- CALCULATE DAILY AVERAGES ---
avg_temp = last_day_data["temperature_2m"].mean()
total_rain = last_day_data["rain"].sum()

print(f"Weather summary for the LAST day ({last_day}):")
print(f"Average Temperature: {avg_temp:.2f} °C")
print(f"Total Rainfall: {total_rain:.2f} mm")
