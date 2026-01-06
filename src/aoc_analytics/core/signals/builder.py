"""
Behavioral Signals Builder - THE KEYSTONE

This module transforms weather conditions into behavioral propensities.
This is the A PRIORI normalization layer - weather effects are accounted
for BEFORE any sales correlation occurs.

Philosophy:
> "Weather is not an insight — it's background radiation."

All downstream analytics operate on residual demand after this
normalization has been applied.

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
All Rights Reserved. Proprietary and Confidential.
"""

from __future__ import annotations

import json
import math
import sqlite3
from collections import defaultdict, deque
from datetime import date as _date, timedelta
from typing import Any, Union

from ..db_adapter import DBAdapter, wrap_connection
from .payday_index import build_payday_index


def rebuild_behavioral_signals(
    conn: Union[sqlite3.Connection, Any],
    *,
    location: str,
    start_date: str,
    end_date: str,
) -> int:
    """
    Rebuild the behavioral_signals_fact table for a date range.
    
    This is the master function that transforms raw weather + events + mood
    data into normalized behavioral propensities.
    
    Args:
        conn: Database connection (SQLite or PostgreSQL via psycopg2)
        location: Store location identifier
        start_date: Start of date range (YYYY-MM-DD)
        end_date: End of date range (YYYY-MM-DD)
    
    Returns:
        Number of records inserted
    """
    # Wrap connection in adapter for cross-database compatibility
    db = wrap_connection(conn)
    
    db.execute(
        "DELETE FROM behavioral_signals_fact WHERE date BETWEEN ? AND ?",
        (start_date, end_date),
    )

    date_range = list(_iter_dates(start_date, end_date))
    if not date_range:
        return 0

    store_locations = _fetch_sales_locations(db)
    payday_rows = list(
        build_payday_index(db, location=location, start_date=start_date, end_date=end_date)
    )
    payday_map = {row.date: row.payday for row in payday_rows}
    payday_meta = {row.date: row.metadata or {} for row in payday_rows}

    weather_stats = _collect_weather_stats(db, start_date, end_date)
    sales_stats = _collect_sales_stats(db, location, start_date, end_date)
    event_scores = _collect_event_scores(db, location, start_date, end_date)
    mood_components = _load_mood_components(db, date_range)

    payload = []
    for current in date_range:
        date_key = current.isoformat()
        weather = weather_stats.get(date_key)
        sales = sales_stats.get(date_key, {})
        events = event_scores.get(date_key, {})
        mood = mood_components.get(date_key, _empty_mood_entry())

        # Apply keystone transformations
        payday = min(payday_map.get(date_key, 0.0) or 0.0, 1.0)
        at_home = _score_at_home(weather, mood)
        out_about = _score_out_and_about(weather, sales, mood)
        holiday = _score_holiday(current, events)
        cultural = _score_cultural(current, events)
        sports = _score_sports(current, sales, events)
        concert = _score_concert(current, sales, events)
        local_vibe = _score_local_vibe(sales, events, mood)

        metadata = {
            "payday": payday_meta.get(date_key, {}),
            "weather": weather or {},
            "sales": sales,
            "events": events,
            "mood": mood,
        }

        for store_loc in store_locations:
            for hour in range(24):
                payload.append(
                    (
                        date_key,
                        hour,
                        store_loc,
                        at_home,
                        out_about,
                        holiday,
                        cultural,
                        sports,
                        concert,
                        payday,
                        local_vibe,
                        mood.get("music_component"),
                        mood.get("anxiety_component"),
                        mood.get("party_component"),
                        mood.get("coziness_component"),
                        mood.get("emotional_tone"),
                        mood.get("activation_component"),
                        mood.get("at_home_component"),
                        json.dumps(metadata),
                    )
                )

    db.executemany(
        """
        INSERT INTO behavioral_signals_fact (
            date, hour, location,
            at_home,
            out_and_about,
            holiday,
            cultural,
            sports,
            concert,
            payday,
            local_vibe,
            local_vibe_music_component,
            local_vibe_anxiety_component,
            local_vibe_party_energy,
            local_vibe_coziness,
            local_vibe_emotional_tone,
            local_vibe_activation,
            local_vibe_at_home_component,
            metadata
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        payload,
    )
    return len(payload)


# =============================================================================
# KEYSTONE FUNCTIONS - Weather to Behavioral Propensity
# =============================================================================

def _score_at_home(
    weather: dict[str, float] | None,
    mood: dict[str, float] | None = None
) -> float:
    """
    KEYSTONE FUNCTION - DO NOT MODIFY WITHOUT REVIEW
    
    Converts weather conditions into behavioral propensity to stay home.
    This is applied BEFORE any sales correlation - treating weather as
    a priori context, not a variable to correlate.
    
    Components:
    - rain_factor: 0-1 based on precipitation (6mm = max effect)
    - cold_factor: 0-1 based on feels_like temp (below 12°C increases)
    - wind_factor: 0-1 based on wind speed (40kph = max effect)
    - snow_factor: 0-1 based on snow probability (boosted 1.25x)
    
    Formula: base = 0.1 + 0.45*rain + 0.3*cold + 0.1*wind + 0.15*snow
    Then blended with mood signal at 35% weight.
    
    Args:
        weather: Weather statistics dict with avg_precip, feels_like, etc.
        mood: Optional mood components dict with at_home_component
    
    Returns:
        Float 0-1 representing propensity to stay home
    """
    if not weather:
        base = 0.35
    else:
        precip = weather.get("avg_precip", 0.0) or 0.0
        rain_factor = min(precip / 6.0, 1.0)
        cold_factor = max(0.0, (12.0 - (weather.get("feels_like") or weather.get("avg_temp") or 12.0)) / 15.0)
        wind_factor = min((weather.get("avg_wind", 0.0) or 0.0) / 40.0, 1.0)
        snow = min(weather.get("snow_share", 0.0) * 1.25, 1.0)
        base = _clamp(0.1 + 0.45 * rain_factor + 0.3 * cold_factor + 0.1 * wind_factor + 0.15 * snow)
    return _blend_scores(base, (mood or {}).get("at_home_component"), weight=0.35)


def _score_out_and_about(
    weather: dict[str, float] | None,
    sales: dict[str, float],
    mood: dict[str, float] | None = None,
) -> float:
    """
    KEYSTONE FUNCTION - Inverse of at_home
    
    Converts weather conditions into propensity for outdoor activity.
    Warm, dry weather increases this score.
    
    Args:
        weather: Weather statistics dict
        sales: Sales statistics dict with revenue_norm, evening_share
        mood: Optional mood components dict
    
    Returns:
        Float 0-1 representing propensity for outdoor activity
    """
    base = 0.25
    if weather:
        warmth = max(0.0, (weather.get("avg_temp", 12.0) - 5.0) / 18.0)
        dryness = 1.0 - min((weather.get("avg_precip", 0.0) or 0.0) / 5.0, 1.0)
        base += 0.5 * warmth + 0.2 * dryness
    base += 0.1 * sales.get("revenue_norm", 0.3)
    return _blend_scores(base, (mood or {}).get("activation_component"), weight=0.3)


def _score_local_vibe(
    sales: dict[str, float],
    events: dict[str, float],
    mood: dict[str, float] | None = None,
) -> float:
    """
    Composite local mood indicator.
    
    Combines revenue performance, evening activity, and event density
    into an overall "vibe" score for the location.
    """
    revenue_norm = sales.get("revenue_norm", 0.3)
    evening = sales.get("evening_share", 0.25)
    event_density = min(events.get("total", 0.0) / 2.0, 1.0)
    base = _clamp(0.2 + 0.45 * revenue_norm + 0.2 * evening + 0.25 * event_density)
    return _blend_scores(base, (mood or {}).get("emotional_tone"), weight=0.4)


# =============================================================================
# Event Scoring Functions
# =============================================================================

def _score_holiday(current: _date, events: dict[str, float]) -> float:
    """Score holiday influence."""
    event_val = _event_value(events, "holiday")
    seasonal = _holiday_seasonal_hint(current)
    base = 0.08 + 0.35 * seasonal
    if event_val > 0:
        base += 0.15
    score = base + 0.65 * event_val + 0.35 * seasonal
    return _clamp(score)


def _score_cultural(current: _date, events: dict[str, float]) -> float:
    """Score cultural event influence."""
    event_val = _event_value(events, "cultural")
    seasonal = 0.18
    if current.month in {1, 2}:
        seasonal += 0.25  # Lunar New Year
    if current.month == 4:
        seasonal += 0.2  # Vaisakhi
    if current.month in {9, 10, 11}:
        seasonal += 0.22  # Diwali/holiday markets
    if current.weekday() in {4, 5}:
        seasonal += 0.05
    return _clamp(seasonal + event_val)


def _score_sports(current: _date, sales: dict[str, float], events: dict[str, float]) -> float:
    """Score sports event influence."""
    event_val = _event_value(events, "sports")
    weekend = 0.5 if current.weekday() in {4, 5, 6} else 0.25
    evening = sales.get("evening_share", 0.3)
    return _clamp(0.15 + 0.4 * weekend + 0.25 * evening + 0.3 * event_val)


def _score_concert(current: _date, sales: dict[str, float], events: dict[str, float]) -> float:
    """Score concert/music event influence."""
    event_val = _event_value(events, "concert")
    weekend = 0.45 if current.weekday() in {3, 4, 5, 6} else 0.2
    evening = sales.get("evening_share", 0.3)
    return _clamp(0.12 + 0.35 * weekend + 0.3 * evening + 0.3 * event_val)


# =============================================================================
# Data Collection Functions
# =============================================================================

def _iter_dates(start: str, end: str) -> list[_date]:
    """Generate list of dates in range."""
    try:
        current = _date.fromisoformat(start)
        stop = _date.fromisoformat(end)
    except ValueError:
        return []
    dates: list[_date] = []
    delta = timedelta(days=1)
    while current <= stop:
        dates.append(current)
        current += delta
    return dates


def _fetch_sales_locations(db: DBAdapter) -> list[str]:
    """Get distinct store locations from sales data."""
    rows = db.execute("SELECT DISTINCT COALESCE(location, 'default') FROM sales").fetchall()
    locations = {row[0] or "default" for row in rows}
    if not locations:
        return ["default"]
    return sorted(locations)


def _collect_weather_stats(
    db: DBAdapter,
    start: str,
    end: str
) -> dict[str, dict[str, float]]:
    """Collect daily weather statistics."""
    sql = """
        SELECT
            date,
            AVG(COALESCE(temp_c, 12.0)) AS avg_temp,
            AVG(COALESCE(feels_like_c, temp_c, 12.0)) AS feels_like,
            AVG(COALESCE(precip_mm, 0.0)) AS avg_precip,
            AVG(COALESCE(is_rain, 0)) AS rain_share,
            AVG(COALESCE(is_snow, 0)) AS snow_share,
            AVG(COALESCE(wind_kph, 0.0)) AS avg_wind
        FROM weather_hourly
        WHERE date BETWEEN ? AND ?
        GROUP BY date
    """
    stats: dict[str, dict[str, float]] = {}
    for row in db.execute(sql, (start, end)).fetchall():
        stats[row[0]] = {
            "avg_temp": row[1],
            "feels_like": row[2],
            "avg_precip": row[3],
            "rain_share": row[4] or 0.0,
            "snow_share": row[5] or 0.0,
            "avg_wind": row[6] or 0.0,
        }
    return stats


def _collect_sales_stats(
    db: DBAdapter,
    location: str,
    start: str,
    end: str,
) -> dict[str, dict[str, float]]:
    """Collect daily sales statistics."""
    sql = """
        SELECT
            date,
            SUM(COALESCE(subtotal, 0)) AS revenue,
            SUM(COALESCE(gross_profit, 0)) AS gross_profit,
            AVG(COALESCE(is_evening, 0)) AS evening_share
        FROM sales_weather_fact
        WHERE date BETWEEN ? AND ?
          AND (location = ? OR location IS NULL)
        GROUP BY date
    """
    rows = db.execute(sql, (start, end, location)).fetchall()
    if not rows:
        return {}
    revenues = [row[1] or 0.0 for row in rows]
    rev_min = min(revenues)
    rev_max = max(revenues)
    stats: dict[str, dict[str, float]] = {}
    for row in rows:
        revenue = row[1] or 0.0
        gross_profit = row[2] or 0.0
        evening_share = row[3] or 0.0
        stats[row[0]] = {
            "revenue": revenue,
            "gross_profit": gross_profit,
            "revenue_norm": _normalize(revenue, rev_min, rev_max),
            "evening_share": evening_share,
        }
    return stats


def _collect_event_scores(
    db: DBAdapter,
    location: str,
    start: str,
    end: str,
) -> dict[str, dict[str, float]]:
    """Collect event scores by date."""
    sql = """
        SELECT date(start_ts) AS event_date, event_type, SUM(importance) AS weight
        FROM calendar_events
        WHERE date(start_ts) BETWEEN ? AND ?
          AND (location = ? OR location = 'default')
        GROUP BY event_date, event_type
    """
    events: dict[str, dict[str, float]] = defaultdict(lambda: {"total": 0.0})
    for row in db.execute(sql, (start, end, location)).fetchall():
        day = row[0]
        event_type = row[1] or ""
        weight = float(row[2] or 0.0)
        bucket = events[day]
        bucket[event_type] = bucket.get(event_type, 0.0) + weight
        bucket["total"] = bucket.get("total", 0.0) + weight
    return events


# =============================================================================
# Helper Functions
# =============================================================================

def _event_value(events: dict[str, float], key: str) -> float:
    """Extract normalized event value."""
    raw = events.get(key, 0.0)
    if raw is None:
        return 0.0
    return min(raw / 1.5, 1.0)


def _holiday_seasonal_hint(current: _date) -> float:
    """Get seasonal holiday likelihood."""
    score = 0.12
    if current.month in {10, 11, 12}:
        score += 0.35
    elif current.month in {5, 7}:
        score += 0.25
    elif current.month in {2, 3, 4}:
        score += 0.2
    if current.day in {1, 11, 24, 25, 31}:
        score += 0.15
    return min(score, 1.0)


def _normalize(value: float, minimum: float, maximum: float) -> float:
    """Normalize value to 0-1 range."""
    if maximum - minimum <= 1e-6:
        return 0.5
    return (value - minimum) / (maximum - minimum)


def _clamp(value: float, *, lower: float = 0.0, upper: float = 1.0) -> float:
    """Clamp value to range."""
    if math.isnan(value):
        return lower
    return max(lower, min(upper, value))


def _blend_scores(base: float, mood_component: float | None, *, weight: float) -> float:
    """Blend base score with mood component."""
    if mood_component is None:
        return _clamp(base)
    mood_index = _mood_to_index(mood_component)
    mixed = (1.0 - weight) * base + weight * mood_index
    return _clamp(mixed)


def _mood_to_index(value: float | None) -> float:
    """Convert mood z-score to 0-1 index."""
    if value is None:
        return 0.5
    return max(0.0, min(1.0, (value + 1.0) / 2.0))


# =============================================================================
# Mood Component Loading
# =============================================================================

def _load_mood_components(
    db: DBAdapter,
    dates: list[_date],
    window: int = 30
) -> dict[str, dict[str, float]]:
    """Load mood components from music and search data."""
    if not dates:
        return {}
    start = min(dates) - timedelta(days=window)
    end = max(dates)
    date_keys = [current.isoformat() for current in dates]

    music_rows = db.execute(
        """
        SELECT date, mean_valence, mean_energy
        FROM music_mood_daily
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    valence_series = [(row[0], row[1]) for row in music_rows]
    energy_series = [(row[0], row[2]) for row in music_rows]
    valence_z = _rolling_zscores(valence_series, window)
    energy_z = _rolling_zscores(energy_series, window)

    search_rows = db.execute(
        """
        SELECT date, stress_score, chill_score, party_score, money_pressure_score, cannabis_interest_score
        FROM search_mood_daily
        WHERE date BETWEEN ? AND ?
        ORDER BY date
        """,
        (start.isoformat(), end.isoformat()),
    ).fetchall()
    stress_series = [(row[0], row[1]) for row in search_rows]
    chill_series = [(row[0], row[2]) for row in search_rows]
    party_series = [(row[0], row[3]) for row in search_rows]
    money_series = [(row[0], row[4]) for row in search_rows]
    cannabis_series = [(row[0], row[5]) for row in search_rows]
    stress_z = _rolling_zscores(stress_series, window)
    chill_z = _rolling_zscores(chill_series, window)
    party_z = _rolling_zscores(party_series, window)
    money_z = _rolling_zscores(money_series, window)
    cannabis_z = _rolling_zscores(cannabis_series, window)

    mood_map: dict[str, dict[str, float]] = {}
    for key in date_keys:
        entry = _build_mood_entry(
            valence_z.get(key),
            energy_z.get(key),
            stress_z.get(key),
            money_z.get(key),
            chill_z.get(key),
            party_z.get(key),
            cannabis_z.get(key),
        )
        mood_map[key] = entry
    return mood_map


def _rolling_zscores(
    series: list[tuple[str, float | None]],
    window: int
) -> dict[str, float]:
    """Calculate rolling z-scores for a time series."""
    history: deque[float] = deque()
    result: dict[str, float] = {}
    for date_str, value in sorted(series, key=lambda item: item[0]):
        if value is None:
            continue
        if history:
            mean_value = sum(history) / len(history)
            variance = sum((val - mean_value) ** 2 for val in history) / len(history)
            std_dev = math.sqrt(variance)
            z_score = 0.0 if std_dev < 1e-6 else (float(value) - mean_value) / std_dev
        else:
            z_score = 0.0
        result[date_str] = z_score
        history.append(float(value))
        if len(history) > window:
            history.popleft()
    return result


def _build_mood_entry(
    valence_z: float | None,
    energy_z: float | None,
    stress_z: float | None,
    money_z: float | None,
    chill_z: float | None,
    party_z: float | None,
    cannabis_z: float | None,
) -> dict[str, Any]:
    """Build mood entry from component z-scores."""
    entry = _empty_mood_entry()
    if not any(
        value is not None
        for value in (valence_z, energy_z, stress_z, money_z, chill_z, party_z, cannabis_z)
    ):
        return entry

    entry["music_component"] = _scaled(0.6 * _value_or_zero(valence_z) + 0.4 * _value_or_zero(energy_z))
    entry["anxiety_component"] = _scaled(_value_or_zero(stress_z) + _value_or_zero(money_z))
    entry["party_component"] = _scaled(_value_or_zero(party_z) + 0.5 * _value_or_zero(cannabis_z))
    entry["coziness_component"] = _scaled(_value_or_zero(chill_z) - 0.5 * _value_or_zero(party_z))
    entry["emotional_tone"] = _scaled(
        0.5 * entry["music_component"] - 0.3 * entry["anxiety_component"] + 0.2 * entry["coziness_component"]
    )
    entry["activation_component"] = _scaled(
        0.4 * entry["party_component"] + 0.3 * entry["music_component"] - 0.3 * entry["anxiety_component"]
    )
    entry["at_home_component"] = _scaled(
        0.5 * entry["coziness_component"] - 0.3 * entry["party_component"] + 0.2 * entry["anxiety_component"]
    )
    entry["available"] = True
    return entry


def _scaled(value: float | None, cap: float = 2.0) -> float:
    """Scale value to -1 to 1 range."""
    if value is None:
        return 0.0
    return max(-cap, min(cap, value)) / cap


def _value_or_zero(value: float | None) -> float:
    """Return value or zero if None."""
    return float(value) if value is not None else 0.0


def _empty_mood_entry() -> dict[str, Any]:
    """Create empty mood entry."""
    return {
        "music_component": 0.0,
        "anxiety_component": 0.0,
        "party_component": 0.0,
        "coziness_component": 0.0,
        "emotional_tone": 0.0,
        "activation_component": 0.0,
        "at_home_component": 0.0,
        "available": False,
    }
