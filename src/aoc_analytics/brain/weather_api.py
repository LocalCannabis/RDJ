"""
Real Weather API Integration

Fetches actual weather forecasts from Environment Canada (free, no API key)
to power the product-weather recommendations with real data.

Uses the open-meteo.com API as a reliable free alternative:
- No API key required
- 7-day forecast
- Historical data available
- Canadian coverage
"""

import json
import urllib.request
import urllib.error
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class DayWeather:
    """Weather for a single day."""
    date: str
    temp_max_c: float
    temp_min_c: float
    precipitation_mm: float
    precipitation_probability: int
    weather_code: int
    weather_description: str
    
    # Derived conditions for product-weather impacts
    is_rainy: bool = False
    is_cold: bool = False
    is_warm: bool = False
    is_mild: bool = False
    
    def __post_init__(self):
        self.is_rainy = self.precipitation_mm >= 5 or self.precipitation_probability >= 70
        self.is_cold = self.temp_max_c < 5
        self.is_warm = self.temp_min_c >= 18
        self.is_mild = 10 <= self.temp_max_c <= 20 and not self.is_rainy
    
    def get_conditions(self) -> List[str]:
        """Return list of active weather conditions."""
        conditions = []
        if self.is_rainy:
            conditions.append("rainy")
        if self.is_cold:
            conditions.append("cold")
        if self.is_warm:
            conditions.append("warm")
        if self.is_mild:
            conditions.append("mild")
        if not self.is_rainy and self.precipitation_mm < 1:
            conditions.append("dry")
        return conditions
    
    def __str__(self) -> str:
        conditions = self.get_conditions()
        cond_str = ", ".join(conditions) if conditions else "normal"
        return (
            f"{self.date}: {self.temp_min_c:.0f}°-{self.temp_max_c:.0f}°C, "
            f"{self.precipitation_mm:.1f}mm ({cond_str})"
        )


# Weather code descriptions (WMO standard)
WEATHER_CODES = {
    0: "Clear sky",
    1: "Mainly clear",
    2: "Partly cloudy",
    3: "Overcast",
    45: "Fog",
    48: "Depositing rime fog",
    51: "Light drizzle",
    53: "Moderate drizzle",
    55: "Dense drizzle",
    61: "Slight rain",
    63: "Moderate rain",
    65: "Heavy rain",
    66: "Light freezing rain",
    67: "Heavy freezing rain",
    71: "Slight snow",
    73: "Moderate snow",
    75: "Heavy snow",
    77: "Snow grains",
    80: "Slight rain showers",
    81: "Moderate rain showers",
    82: "Violent rain showers",
    85: "Slight snow showers",
    86: "Heavy snow showers",
    95: "Thunderstorm",
    96: "Thunderstorm with slight hail",
    99: "Thunderstorm with heavy hail",
}


class WeatherAPI:
    """
    Fetches weather data from Open-Meteo API.
    
    No API key required! Just lat/lon coordinates.
    """
    
    # Vancouver coordinates
    VANCOUVER_LAT = 49.2827
    VANCOUVER_LON = -123.1207
    
    # Parksville coordinates (for second store)
    PARKSVILLE_LAT = 49.3183
    PARKSVILLE_LON = -124.3123
    
    LOCATIONS = {
        "Vancouver": (VANCOUVER_LAT, VANCOUVER_LON),
        "Kingsway": (VANCOUVER_LAT, VANCOUVER_LON),
        "Victoria Drive": (VANCOUVER_LAT, VANCOUVER_LON),
        "Parksville": (PARKSVILLE_LAT, PARKSVILLE_LON),
    }
    
    BASE_URL = "https://api.open-meteo.com/v1/forecast"
    
    def __init__(self, location: str = "Vancouver"):
        if location in self.LOCATIONS:
            self.lat, self.lon = self.LOCATIONS[location]
        else:
            self.lat, self.lon = self.VANCOUVER_LAT, self.VANCOUVER_LON
        
        self.location = location
        self.cache_dir = Path(__file__).parent / "data" / "weather_cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def _get_cache_path(self) -> Path:
        """Get cache file path for today."""
        today = date.today().strftime("%Y-%m-%d")
        return self.cache_dir / f"forecast_{self.location}_{today}.json"
    
    def _load_from_cache(self) -> Optional[Dict]:
        """Load forecast from cache if fresh."""
        cache_path = self._get_cache_path()
        if cache_path.exists():
            with open(cache_path) as f:
                return json.load(f)
        return None
    
    def _save_to_cache(self, data: Dict):
        """Save forecast to cache."""
        cache_path = self._get_cache_path()
        with open(cache_path, "w") as f:
            json.dump(data, f, indent=2)
    
    def fetch_forecast(self, days: int = 7) -> List[DayWeather]:
        """
        Fetch weather forecast for upcoming days.
        
        Uses cache to avoid hitting API repeatedly.
        """
        # Try cache first
        cached = self._load_from_cache()
        if cached:
            return self._parse_forecast(cached)
        
        # Build API URL
        params = {
            "latitude": self.lat,
            "longitude": self.lon,
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,precipitation_probability_max,weather_code",
            "timezone": "America/Vancouver",
            "forecast_days": days,
        }
        
        query = "&".join(f"{k}={v}" for k, v in params.items())
        url = f"{self.BASE_URL}?{query}"
        
        try:
            with urllib.request.urlopen(url, timeout=10) as response:
                data = json.loads(response.read().decode())
                self._save_to_cache(data)
                return self._parse_forecast(data)
        except urllib.error.URLError as e:
            print(f"Weather API error: {e}")
            return self._get_fallback_forecast(days)
        except Exception as e:
            print(f"Weather fetch error: {e}")
            return self._get_fallback_forecast(days)
    
    def _parse_forecast(self, data: Dict) -> List[DayWeather]:
        """Parse API response into DayWeather objects."""
        daily = data.get("daily", {})
        
        dates = daily.get("time", [])
        temp_max = daily.get("temperature_2m_max", [])
        temp_min = daily.get("temperature_2m_min", [])
        precip = daily.get("precipitation_sum", [])
        precip_prob = daily.get("precipitation_probability_max", [])
        weather_codes = daily.get("weather_code", [])
        
        forecasts = []
        for i in range(len(dates)):
            code = weather_codes[i] if i < len(weather_codes) else 0
            forecasts.append(DayWeather(
                date=dates[i],
                temp_max_c=temp_max[i] if i < len(temp_max) else 15.0,
                temp_min_c=temp_min[i] if i < len(temp_min) else 10.0,
                precipitation_mm=precip[i] if i < len(precip) else 0.0,
                precipitation_probability=precip_prob[i] if i < len(precip_prob) else 0,
                weather_code=code,
                weather_description=WEATHER_CODES.get(code, "Unknown"),
            ))
        
        return forecasts
    
    def _get_fallback_forecast(self, days: int) -> List[DayWeather]:
        """Return fallback forecast if API fails."""
        forecasts = []
        today = date.today()
        
        # Vancouver seasonal averages
        month = today.month
        if month in (12, 1, 2):  # Winter
            base_max, base_min, precip = 7, 2, 5.0
        elif month in (3, 4, 5):  # Spring
            base_max, base_min, precip = 14, 7, 3.0
        elif month in (6, 7, 8):  # Summer
            base_max, base_min, precip = 22, 14, 1.0
        else:  # Fall
            base_max, base_min, precip = 12, 6, 4.0
        
        for i in range(days):
            forecast_date = today + timedelta(days=i)
            forecasts.append(DayWeather(
                date=forecast_date.strftime("%Y-%m-%d"),
                temp_max_c=base_max + (i % 3) - 1,
                temp_min_c=base_min + (i % 3) - 1,
                precipitation_mm=precip * (0.5 if i % 2 == 0 else 1.5),
                precipitation_probability=40 + (i * 5) % 30,
                weather_code=3 if i % 3 == 0 else 61,
                weather_description="Fallback forecast",
            ))
        
        return forecasts
    
    def get_conditions_for_date(self, target_date: date) -> List[str]:
        """Get weather conditions for a specific date."""
        forecast = self.fetch_forecast(days=14)
        
        date_str = target_date.strftime("%Y-%m-%d")
        for day in forecast:
            if day.date == date_str:
                return day.get_conditions()
        
        return []
    
    def get_forecast_dict(self) -> Dict[str, List[str]]:
        """
        Get forecast as dict mapping date strings to conditions.
        Ready for use with inventory_recommendations.
        """
        forecast = self.fetch_forecast(days=7)
        
        return {
            day.date: day.get_conditions()
            for day in forecast
        }


def demo():
    """Demonstrate weather API integration."""
    
    print("=" * 70)
    print("🌦️  REAL WEATHER FORECAST")
    print("   Powered by Open-Meteo API (free, no key required)")
    print("=" * 70)
    print()
    
    api = WeatherAPI("Vancouver")
    
    print(f"Location: {api.location} ({api.lat}, {api.lon})")
    print()
    
    # Fetch forecast
    print("Fetching 7-day forecast...")
    forecast = api.fetch_forecast(days=7)
    
    print()
    print("─" * 50)
    print("7-DAY FORECAST")
    print("─" * 50)
    
    for day in forecast:
        conditions = day.get_conditions()
        emoji = "🌧️" if day.is_rainy else "❄️" if day.is_cold else "☀️" if day.is_warm else "🌤️"
        print(f"  {emoji} {day}")
    
    # Show product impacts
    print()
    print("─" * 50)
    print("PRODUCT-WEATHER IMPLICATIONS")
    print("─" * 50)
    
    # Load category weather impacts
    brain_dir = Path(__file__).parent / "data"
    weather_file = brain_dir / "category_weather_impacts.json"
    
    if weather_file.exists():
        with open(weather_file) as f:
            impacts = json.load(f)
        
        actionable = impacts.get("actionable_insights", [])
        
        for day in forecast[:3]:  # First 3 days
            conditions = day.get_conditions()
            if not conditions:
                continue
            
            print(f"\n{day.date} ({', '.join(conditions)}):")
            
            for insight in actionable:
                if insight.get("weather_condition") in conditions:
                    lift = insight.get("avg_lift_pct", 0)
                    cat = insight.get("category", "?")
                    emoji = "📈" if lift > 0 else "📉"
                    print(f"   {emoji} {cat}: {lift:+.1f}%")
    
    # Get dict format for inventory recommendations
    print()
    print("─" * 50)
    print("INTEGRATION FORMAT")
    print("─" * 50)
    
    forecast_dict = api.get_forecast_dict()
    print("weather_forecast = {")
    for date_str, conditions in forecast_dict.items():
        print(f'    "{date_str}": {conditions},')
    print("}")
    
    print()
    print("💾 Weather cached for today")


if __name__ == "__main__":
    demo()
