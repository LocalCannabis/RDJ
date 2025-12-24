"""
Conditions loading utilities.

Loads the unified conditions + outcomes tables from SQLite,
combining sales, weather, calendar signals, and behavioral mood components.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from typing import Optional

import pandas as pd

from .anomaly_registry import add_anomaly_flags_to_df


@dataclass
class ConditionsConfig:
    grain: str = "hour"  # currently only 'hour' is supported; 'day' can be added later


def load_conditions_df(
    connection: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    config: Optional[ConditionsConfig] = None,
) -> pd.DataFrame:
    """Load the unified conditions + outcomes table from SQLite.

    This is a thin convenience wrapper around the existing fact tables. It is
    intentionally conservative to start with: we read from `sales_weather_fact`
    (for sales + weather + calendar) and then left-join behavioral mood
    components from `behavioral_signals_fact` on date.

    Parameters
    ----------
    connection:
        DB connection to the analytics database.
    start_date, end_date:
        Optional ISO date strings (YYYY-MM-DD) to restrict the window.
    config:
        Optional `ConditionsConfig`. Currently only controls time grain.

    Returns
    -------
    pd.DataFrame
        One row per (date, hour) with numeric features suitable for
        similarity-based forecasting and downstream demand models.
    """

    if config is None:
        config = ConditionsConfig()

    if config.grain != "hour":
        raise ValueError(f"Unsupported grain {config.grain!r}; only 'hour' is implemented.")

    params = {}
    where_clauses = []
    if start_date is not None:
        where_clauses.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where_clauses.append("date <= :end_date")
        params["end_date"] = end_date

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    # Base fact: sales + weather + calendar signals.
    base_sql = f"""
        SELECT
            swf.date,
            swf.hour,
            SUM(swf.subtotal) AS sales_revenue,
            SUM(swf.quantity) AS sales_units,
            AVG(swf.temp_c) AS temp_c,
            AVG(swf.feels_like_c) AS feels_like_c,
            AVG(swf.precip_mm) AS precip_mm,
            COALESCE(swf.precip_type, 'unknown') AS precip_type,
            AVG(swf.cloud_cover_pct) AS cloud_cover_pct,
            AVG(swf.humidity) AS humidity,
            AVG(swf.wind_kph) AS wind_kph,
            COALESCE(swf.conditions, 'Unknown') AS conditions,
            swf.is_weekend,
            swf.is_evening,
            swf.rain_bucket,
            swf.temp_bucket
        FROM sales_weather_fact AS swf
        {where_sql}
        GROUP BY swf.date, swf.hour, swf.is_weekend, swf.is_evening, swf.rain_bucket, swf.temp_bucket, precip_type, conditions
    """

    base_df = pd.read_sql_query(base_sql, connection, params=params)
    if base_df.empty:
        return base_df

    # Derive simple calendar fields expected by the similarity config.
    # We keep this minimal for now: day-of-week integer and basic flags
    # based on simple rules (weekends as holidays, preholiday as Friday).
    base_df["date"] = pd.to_datetime(base_df["date"]).dt.date
    base_df["dow"] = pd.to_datetime(base_df["date"]).dt.weekday
    # Placeholder calendar features – these can be replaced by a proper
    # calendar/events dimension later.
    base_df["is_holiday"] = 0
    base_df["is_preholiday"] = (base_df["dow"] == 4).astype(int)
    base_df["is_payday_window"] = 0
    base_df["has_home_game"] = 0
    base_df["has_concert"] = 0
    base_df["has_festival"] = 0

    # Mood + behavioral components live at the daily grain.
    mood_sql = """
        SELECT
            date,
            at_home AS local_vibe_at_home_index,
            out_and_about AS local_vibe_out_and_about_index,
            holiday AS local_vibe_holiday_index,
            local_vibe AS local_vibe_overall_index
        FROM behavioral_signals_fact
    """
    mood_df = pd.read_sql_query(mood_sql, connection)

    df = base_df.merge(mood_df, on="date", how="left")

    # Simple dtype normalization: ensure hour is int.
    if "hour" in df.columns:
        df["hour"] = df["hour"].astype(int)

    # Add anomaly flags for down-weighting disrupted periods
    df = add_anomaly_flags_to_df(df, connection, date_col="date")

    return df


def load_category_conditions_df(
    connection: sqlite3.Connection,
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
) -> pd.DataFrame:
    """Load a category-level conditions + outcomes table from SQLite.

    This aggregates `sales_weather_fact` by (date, hour, category_path),
    computing revenue and units per category while retaining the same weather
    and behavioral context as `load_conditions_df`.
    """
    params = {}
    where_clauses = []
    if start_date is not None:
        where_clauses.append("date >= :start_date")
        params["start_date"] = start_date
    if end_date is not None:
        where_clauses.append("date <= :end_date")
        params["end_date"] = end_date

    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)

    sql = f"""
        SELECT
            swf.date,
            swf.hour,
            swf.category_path,
            SUM(swf.subtotal) AS sales_revenue,
            SUM(swf.quantity) AS sales_units,
            AVG(swf.temp_c) AS temp_c,
            AVG(swf.feels_like_c) AS feels_like_c,
            AVG(swf.precip_mm) AS precip_mm,
            COALESCE(swf.precip_type, 'unknown') AS precip_type,
            AVG(swf.cloud_cover_pct) AS cloud_cover_pct,
            AVG(swf.humidity) AS humidity,
            AVG(swf.wind_kph) AS wind_kph,
            COALESCE(swf.conditions, 'Unknown') AS conditions,
            swf.is_weekend,
            swf.is_evening,
            swf.rain_bucket,
            swf.temp_bucket
        FROM sales_weather_fact AS swf
        {where_sql}
        GROUP BY swf.date, swf.hour, swf.category_path,
                 swf.is_weekend, swf.is_evening,
                 swf.rain_bucket, swf.temp_bucket,
                 precip_type, conditions
    """

    df = pd.read_sql_query(sql, connection, params=params)
    if df.empty:
        return df

    # Normalize date/hour and derive calendar fields expected by
    # SimilarityConfig so category-level forecasts can reuse the same
    # vectorization.
    df["date"] = pd.to_datetime(df["date"]).dt.date
    df["hour"] = df["hour"].astype(int)
    df["dow"] = pd.to_datetime(df["date"]).dt.weekday
    df["is_holiday"] = 0
    df["is_preholiday"] = (df["dow"] == 4).astype(int)
    df["is_payday_window"] = 0
    df["has_home_game"] = 0
    df["has_concert"] = 0
    df["has_festival"] = 0

    # Add anomaly flags for down-weighting disrupted periods
    df = add_anomaly_flags_to_df(df, connection, date_col="date")

    return df
