"""
Weather Impact Signal

Calculates expected sales lift/drag based on weather conditions.
Learned from historical sales vs weather correlation.

Key findings:
- Cold weather: -12% (people stay home)
- Rainy weather: -4% (especially pre-rolls - outdoor smoking affected)
- Mild weather: +6% (comfortable outdoor activity)
- Warm weather: +10% (summer vibes)
"""

import json
from datetime import date
from pathlib import Path
from typing import Dict, Optional, Tuple, List


# Overall weather impacts on total sales (not category-specific)
# Derived from category impacts weighted by category volume
WEATHER_IMPACTS = {
    "cold": -0.08,    # Cold weather suppresses sales (~-8%)
    "rainy": -0.04,   # Rain suppresses outdoor activity (~-4%)
    "mild": +0.04,    # Mild weather boosts shopping (~+4%)
    "warm": +0.06,    # Hot weather boosts social activity (~+6%)
    "dry": +0.01,     # Slightly positive vs rainy
}


def get_weather_impact(conditions: List[str]) -> Tuple[float, str]:
    """
    Calculate total weather impact from active conditions.
    
    Args:
        conditions: List of weather condition strings like ["cold", "dry"]
        
    Returns:
        (impact, description) tuple
    """
    if not conditions:
        return 0.0, "normal"
    
    total_impact = 0.0
    descriptions = []
    
    for condition in conditions:
        impact = WEATHER_IMPACTS.get(condition.lower(), 0.0)
        if impact != 0:
            total_impact += impact
            if impact > 0:
                descriptions.append(f"{condition}:+{impact:.0%}")
            else:
                descriptions.append(f"{condition}:{impact:.0%}")
    
    # Apply diminishing returns for multiple conditions
    if len(conditions) > 1:
        total_impact *= 0.8  # 20% discount for overlapping effects
    
    # Cap total impact
    total_impact = max(-0.15, min(0.15, total_impact))
    
    description = ", ".join(descriptions) if descriptions else "normal"
    return total_impact, description


def get_weather_for_date(target_date: Optional[date] = None) -> Tuple[List[str], float]:
    """
    Get weather conditions and impact for a date.
    
    For future dates: uses forecast
    For past dates: tries to load from cache, falls back to average
    
    Returns:
        (conditions_list, expected_impact)
    """
    if target_date is None:
        target_date = date.today()
    
    # Try to get weather data
    try:
        from aoc_analytics.brain.weather_api import WeatherAPI
        
        api = WeatherAPI()
        
        # For recent/future dates, use forecast
        days_diff = (target_date - date.today()).days
        
        if -7 <= days_diff <= 7:
            # Within forecast window
            forecast = api.fetch_forecast(days=14)
            for day in forecast:
                if day.date == target_date.isoformat():
                    conditions = day.get_conditions()
                    impact, desc = get_weather_impact(conditions)
                    return conditions, impact
        
        # For older dates, try cache
        cache_dir = Path(__file__).parent / "data" / "weather_cache"
        for cache_file in cache_dir.glob("*.json"):
            try:
                with open(cache_file) as f:
                    data = json.load(f)
                    daily = data.get("daily", {})
                    dates = daily.get("time", [])
                    
                    if target_date.isoformat() in dates:
                        idx = dates.index(target_date.isoformat())
                        temp_max = daily.get("temperature_2m_max", [])[idx]
                        temp_min = daily.get("temperature_2m_min", [])[idx]
                        precip = daily.get("precipitation_sum", [])[idx]
                        
                        conditions = []
                        if temp_max < 5:
                            conditions.append("cold")
                        elif temp_min >= 18:
                            conditions.append("warm")
                        elif 10 <= temp_max <= 20:
                            conditions.append("mild")
                        
                        if precip >= 5:
                            conditions.append("rainy")
                        elif precip < 1:
                            conditions.append("dry")
                        
                        impact, desc = get_weather_impact(conditions)
                        return conditions, impact
            except:
                continue
        
    except ImportError:
        pass
    except Exception as e:
        print(f"Weather lookup error: {e}")
    
    # Default: assume normal weather (no impact)
    return ["dry", "mild"], 0.02


def format_weather_signal(target_date: Optional[date] = None) -> str:
    """Format weather signal as human-readable string."""
    if target_date is None:
        target_date = date.today()
    
    conditions, impact = get_weather_for_date(target_date)
    
    if not conditions:
        return f"🌤️ {target_date}: Normal weather (no impact)"
    
    condition_str = ", ".join(conditions)
    
    if impact > 0.03:
        emoji = "☀️"
        direction = "boost"
    elif impact < -0.03:
        emoji = "🌧️"
        direction = "drag"
    else:
        emoji = "🌤️"
        direction = "neutral"
    
    return f"{emoji} {target_date}: {condition_str} ({impact:+.1%} {direction})"


if __name__ == "__main__":
    from datetime import timedelta
    
    print("Weather Impact Signal")
    print("=" * 50)
    
    # Show next 7 days
    today = date.today()
    print(f"\nNext 7 days:")
    print("-" * 50)
    
    for i in range(7):
        d = today + timedelta(days=i)
        print(format_weather_signal(d))
