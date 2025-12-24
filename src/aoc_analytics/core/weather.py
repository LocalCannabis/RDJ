"""
Open-Meteo Weather Client

Fetches historical weather data and forecasts from the Open-Meteo API.
Free, no API key required, and provides high-quality weather data.

Usage:
    from aoc_analytics.core.weather import WeatherClient
    
    client = WeatherClient(latitude=49.1913, longitude=-123.9583)  # Parksville
    
    # Get historical data
    historical = client.get_historical("2024-01-01", "2024-12-31")
    
    # Get forecast
    forecast = client.get_forecast(days=7)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

import httpx

logger = logging.getLogger(__name__)

# Open-Meteo API endpoints
HISTORICAL_API = "https://archive-api.open-meteo.com/v1/archive"
FORECAST_API = "https://api.open-meteo.com/v1/forecast"

# Store locations (latitude, longitude) - geocoded from actual addresses
STORE_LOCATIONS = {
    "Parksville": (49.3208840, -124.3116989),    # 491B Island Hwy E, Parksville BC V9P 1V9
    "Kingsway": (49.2561347, -123.0893652),      # 726 Kingsway, Vancouver BC V5V 3C1
    "Victoria Drive": (49.2759582, -123.0656508), # 6945 Victoria Dr, Vancouver BC V5P 3Y7
}

# Weather variables to fetch
HOURLY_VARIABLES = [
    "temperature_2m",
    "apparent_temperature",
    "precipitation",
    "rain",
    "snowfall",
    "cloud_cover",
    "relative_humidity_2m",
    "wind_speed_10m",
    "weather_code",
]

DAILY_VARIABLES = [
    "temperature_2m_max",
    "temperature_2m_min",
    "apparent_temperature_max",
    "apparent_temperature_min",
    "precipitation_sum",
    "rain_sum",
    "snowfall_sum",
    "precipitation_hours",
    "wind_speed_10m_max",
    "weather_code",
]

# WMO Weather Codes mapping
WMO_CODES = {
    0: ("Clear sky", "clear"),
    1: ("Mainly clear", "clear"),
    2: ("Partly cloudy", "cloudy"),
    3: ("Overcast", "cloudy"),
    45: ("Fog", "fog"),
    48: ("Depositing rime fog", "fog"),
    51: ("Light drizzle", "drizzle"),
    53: ("Moderate drizzle", "drizzle"),
    55: ("Dense drizzle", "drizzle"),
    56: ("Light freezing drizzle", "freezing_rain"),
    57: ("Dense freezing drizzle", "freezing_rain"),
    61: ("Slight rain", "rain"),
    63: ("Moderate rain", "rain"),
    65: ("Heavy rain", "rain"),
    66: ("Light freezing rain", "freezing_rain"),
    67: ("Heavy freezing rain", "freezing_rain"),
    71: ("Slight snowfall", "snow"),
    73: ("Moderate snowfall", "snow"),
    75: ("Heavy snowfall", "snow"),
    77: ("Snow grains", "snow"),
    80: ("Slight rain showers", "rain"),
    81: ("Moderate rain showers", "rain"),
    82: ("Violent rain showers", "rain"),
    85: ("Slight snow showers", "snow"),
    86: ("Heavy snow showers", "snow"),
    95: ("Thunderstorm", "thunderstorm"),
    96: ("Thunderstorm with slight hail", "thunderstorm"),
    99: ("Thunderstorm with heavy hail", "thunderstorm"),
}


@dataclass
class HourlyWeather:
    """Hourly weather observation."""
    datetime: datetime
    temp_c: float
    feels_like_c: float
    precip_mm: float
    rain_mm: float
    snow_mm: float
    cloud_cover_pct: float
    humidity_pct: float
    wind_kph: float
    weather_code: int
    condition: str
    precip_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "datetime": self.datetime.isoformat(),
            "date": self.datetime.strftime("%Y-%m-%d"),
            "hour": self.datetime.hour,
            "temp_c": self.temp_c,
            "feels_like_c": self.feels_like_c,
            "precip_mm": self.precip_mm,
            "rain_mm": self.rain_mm,
            "snow_mm": self.snow_mm,
            "cloud_cover_pct": self.cloud_cover_pct,
            "humidity_pct": self.humidity_pct,
            "wind_kph": self.wind_kph,
            "weather_code": self.weather_code,
            "condition": self.condition,
            "precip_type": self.precip_type,
        }


@dataclass
class DailyWeather:
    """Daily weather summary."""
    date: date
    temp_max_c: float
    temp_min_c: float
    feels_like_max_c: float
    feels_like_min_c: float
    precip_mm: float
    rain_mm: float
    snow_mm: float
    precip_hours: float
    wind_max_kph: float
    weather_code: int
    condition: str
    precip_type: str
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "date": self.date.isoformat(),
            "temp_max_c": self.temp_max_c,
            "temp_min_c": self.temp_min_c,
            "feels_like_max_c": self.feels_like_max_c,
            "feels_like_min_c": self.feels_like_min_c,
            "precip_mm": self.precip_mm,
            "rain_mm": self.rain_mm,
            "snow_mm": self.snow_mm,
            "precip_hours": self.precip_hours,
            "wind_max_kph": self.wind_max_kph,
            "weather_code": self.weather_code,
            "condition": self.condition,
            "precip_type": self.precip_type,
        }


def _get_precip_type(rain_mm: float, snow_mm: float, weather_code: int) -> str:
    """Determine precipitation type from values and weather code."""
    if snow_mm > 0:
        return "snow"
    if rain_mm > 0:
        return "rain"
    if weather_code in (51, 53, 55):
        return "drizzle"
    if weather_code in (56, 57, 66, 67):
        return "freezing_rain"
    return "none"


def _get_condition(weather_code: int) -> str:
    """Get human-readable condition from WMO code."""
    if weather_code in WMO_CODES:
        return WMO_CODES[weather_code][0]
    return "Unknown"


class WeatherClient:
    """Client for fetching weather data from Open-Meteo."""
    
    def __init__(
        self,
        latitude: Optional[float] = None,
        longitude: Optional[float] = None,
        location: Optional[str] = None,
        timezone: str = "America/Vancouver",
    ):
        """
        Initialize weather client.
        
        Args:
            latitude: Latitude coordinate (use with longitude)
            longitude: Longitude coordinate (use with latitude)
            location: Named location from STORE_LOCATIONS
            timezone: Timezone for data (default: America/Vancouver)
        """
        if location and location in STORE_LOCATIONS:
            self.latitude, self.longitude = STORE_LOCATIONS[location]
            self.location_name = location
        elif latitude is not None and longitude is not None:
            self.latitude = latitude
            self.longitude = longitude
            self.location_name = f"({latitude}, {longitude})"
        else:
            # Default to Parksville
            self.latitude, self.longitude = STORE_LOCATIONS["Parksville"]
            self.location_name = "Parksville"
        
        self.timezone = timezone
        self._client = httpx.Client(timeout=30.0)
    
    def __del__(self):
        if hasattr(self, '_client'):
            self._client.close()
    
    def get_historical_hourly(
        self,
        start_date: str,
        end_date: str,
    ) -> List[HourlyWeather]:
        """
        Fetch historical hourly weather data.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of HourlyWeather observations
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
        }
        
        logger.info(f"Fetching historical hourly weather for {self.location_name} from {start_date} to {end_date}")
        
        response = self._client.get(HISTORICAL_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        results = []
        for i, time_str in enumerate(times):
            dt = datetime.fromisoformat(time_str)
            
            rain_mm = hourly.get("rain", [0] * len(times))[i] or 0
            snow_mm = hourly.get("snowfall", [0] * len(times))[i] or 0
            weather_code = hourly.get("weather_code", [0] * len(times))[i] or 0
            
            results.append(HourlyWeather(
                datetime=dt,
                temp_c=hourly.get("temperature_2m", [0] * len(times))[i] or 0,
                feels_like_c=hourly.get("apparent_temperature", [0] * len(times))[i] or 0,
                precip_mm=hourly.get("precipitation", [0] * len(times))[i] or 0,
                rain_mm=rain_mm,
                snow_mm=snow_mm,
                cloud_cover_pct=hourly.get("cloud_cover", [0] * len(times))[i] or 0,
                humidity_pct=hourly.get("relative_humidity_2m", [0] * len(times))[i] or 0,
                wind_kph=hourly.get("wind_speed_10m", [0] * len(times))[i] or 0,
                weather_code=weather_code,
                condition=_get_condition(weather_code),
                precip_type=_get_precip_type(rain_mm, snow_mm, weather_code),
            ))
        
        logger.info(f"Fetched {len(results)} hourly observations")
        return results
    
    def get_historical_daily(
        self,
        start_date: str,
        end_date: str,
    ) -> List[DailyWeather]:
        """
        Fetch historical daily weather summaries.
        
        Args:
            start_date: Start date (YYYY-MM-DD)
            end_date: End date (YYYY-MM-DD)
            
        Returns:
            List of DailyWeather summaries
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "start_date": start_date,
            "end_date": end_date,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": self.timezone,
        }
        
        logger.info(f"Fetching historical daily weather for {self.location_name} from {start_date} to {end_date}")
        
        response = self._client.get(HISTORICAL_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        
        results = []
        for i, date_str in enumerate(dates):
            d = date.fromisoformat(date_str)
            
            rain_mm = daily.get("rain_sum", [0] * len(dates))[i] or 0
            snow_mm = daily.get("snowfall_sum", [0] * len(dates))[i] or 0
            weather_code = daily.get("weather_code", [0] * len(dates))[i] or 0
            
            results.append(DailyWeather(
                date=d,
                temp_max_c=daily.get("temperature_2m_max", [0] * len(dates))[i] or 0,
                temp_min_c=daily.get("temperature_2m_min", [0] * len(dates))[i] or 0,
                feels_like_max_c=daily.get("apparent_temperature_max", [0] * len(dates))[i] or 0,
                feels_like_min_c=daily.get("apparent_temperature_min", [0] * len(dates))[i] or 0,
                precip_mm=daily.get("precipitation_sum", [0] * len(dates))[i] or 0,
                rain_mm=rain_mm,
                snow_mm=snow_mm,
                precip_hours=daily.get("precipitation_hours", [0] * len(dates))[i] or 0,
                wind_max_kph=daily.get("wind_speed_10m_max", [0] * len(dates))[i] or 0,
                weather_code=weather_code,
                condition=_get_condition(weather_code),
                precip_type=_get_precip_type(rain_mm, snow_mm, weather_code),
            ))
        
        logger.info(f"Fetched {len(results)} daily summaries")
        return results
    
    def get_forecast_hourly(self, days: int = 7) -> List[HourlyWeather]:
        """
        Fetch hourly weather forecast.
        
        Args:
            days: Number of days to forecast (max 16)
            
        Returns:
            List of HourlyWeather forecasts
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "hourly": ",".join(HOURLY_VARIABLES),
            "timezone": self.timezone,
            "forecast_days": min(days, 16),
        }
        
        logger.info(f"Fetching {days}-day hourly forecast for {self.location_name}")
        
        response = self._client.get(FORECAST_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        hourly = data.get("hourly", {})
        times = hourly.get("time", [])
        
        results = []
        for i, time_str in enumerate(times):
            dt = datetime.fromisoformat(time_str)
            
            rain_mm = hourly.get("rain", [0] * len(times))[i] or 0
            snow_mm = hourly.get("snowfall", [0] * len(times))[i] or 0
            weather_code = hourly.get("weather_code", [0] * len(times))[i] or 0
            
            results.append(HourlyWeather(
                datetime=dt,
                temp_c=hourly.get("temperature_2m", [0] * len(times))[i] or 0,
                feels_like_c=hourly.get("apparent_temperature", [0] * len(times))[i] or 0,
                precip_mm=hourly.get("precipitation", [0] * len(times))[i] or 0,
                rain_mm=rain_mm,
                snow_mm=snow_mm,
                cloud_cover_pct=hourly.get("cloud_cover", [0] * len(times))[i] or 0,
                humidity_pct=hourly.get("relative_humidity_2m", [0] * len(times))[i] or 0,
                wind_kph=hourly.get("wind_speed_10m", [0] * len(times))[i] or 0,
                weather_code=weather_code,
                condition=_get_condition(weather_code),
                precip_type=_get_precip_type(rain_mm, snow_mm, weather_code),
            ))
        
        logger.info(f"Fetched {len(results)} hourly forecasts")
        return results
    
    def get_forecast_daily(self, days: int = 7) -> List[DailyWeather]:
        """
        Fetch daily weather forecast.
        
        Args:
            days: Number of days to forecast (max 16)
            
        Returns:
            List of DailyWeather forecasts
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "daily": ",".join(DAILY_VARIABLES),
            "timezone": self.timezone,
            "forecast_days": min(days, 16),
        }
        
        logger.info(f"Fetching {days}-day daily forecast for {self.location_name}")
        
        response = self._client.get(FORECAST_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        daily = data.get("daily", {})
        dates = daily.get("time", [])
        
        results = []
        for i, date_str in enumerate(dates):
            d = date.fromisoformat(date_str)
            
            rain_mm = daily.get("rain_sum", [0] * len(dates))[i] or 0
            snow_mm = daily.get("snowfall_sum", [0] * len(dates))[i] or 0
            weather_code = daily.get("weather_code", [0] * len(dates))[i] or 0
            
            results.append(DailyWeather(
                date=d,
                temp_max_c=daily.get("temperature_2m_max", [0] * len(dates))[i] or 0,
                temp_min_c=daily.get("temperature_2m_min", [0] * len(dates))[i] or 0,
                feels_like_max_c=daily.get("apparent_temperature_max", [0] * len(dates))[i] or 0,
                feels_like_min_c=daily.get("apparent_temperature_min", [0] * len(dates))[i] or 0,
                precip_mm=daily.get("precipitation_sum", [0] * len(dates))[i] or 0,
                rain_mm=rain_mm,
                snow_mm=snow_mm,
                precip_hours=daily.get("precipitation_hours", [0] * len(dates))[i] or 0,
                wind_max_kph=daily.get("wind_speed_10m_max", [0] * len(dates))[i] or 0,
                weather_code=weather_code,
                condition=_get_condition(weather_code),
                precip_type=_get_precip_type(rain_mm, snow_mm, weather_code),
            ))
        
        logger.info(f"Fetched {len(results)} daily forecasts")
        return results
    
    def get_current(self) -> HourlyWeather:
        """
        Get current weather conditions.
        
        Returns:
            Current HourlyWeather observation
        """
        params = {
            "latitude": self.latitude,
            "longitude": self.longitude,
            "current": "temperature_2m,apparent_temperature,precipitation,rain,snowfall,cloud_cover,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": self.timezone,
        }
        
        response = self._client.get(FORECAST_API, params=params)
        response.raise_for_status()
        data = response.json()
        
        current = data.get("current", {})
        
        rain_mm = current.get("rain", 0) or 0
        snow_mm = current.get("snowfall", 0) or 0
        weather_code = current.get("weather_code", 0) or 0
        
        return HourlyWeather(
            datetime=datetime.fromisoformat(current.get("time", datetime.now().isoformat())),
            temp_c=current.get("temperature_2m", 0) or 0,
            feels_like_c=current.get("apparent_temperature", 0) or 0,
            precip_mm=current.get("precipitation", 0) or 0,
            rain_mm=rain_mm,
            snow_mm=snow_mm,
            cloud_cover_pct=current.get("cloud_cover", 0) or 0,
            humidity_pct=current.get("relative_humidity_2m", 0) or 0,
            wind_kph=current.get("wind_speed_10m", 0) or 0,
            weather_code=weather_code,
            condition=_get_condition(weather_code),
            precip_type=_get_precip_type(rain_mm, snow_mm, weather_code),
        )


def get_weather_for_location(location: str) -> WeatherClient:
    """Get a weather client for a named store location."""
    return WeatherClient(location=location)


def get_all_store_weather() -> Dict[str, WeatherClient]:
    """Get weather clients for all store locations."""
    return {name: WeatherClient(location=name) for name in STORE_LOCATIONS}
