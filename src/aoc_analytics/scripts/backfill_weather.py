#!/usr/bin/env python3
"""
Backfill historical weather data from Open-Meteo.

This script fetches historical weather data for all store locations
and stores it in the SQLite database, allowing us to join sales
data with weather conditions.

Usage:
    python -m aoc_analytics.scripts.backfill_weather --start-date 2021-07-01 --end-date 2024-12-31
    
    # For a specific location
    python -m aoc_analytics.scripts.backfill_weather --location Parksville --start-date 2024-01-01 --end-date 2024-12-31
    
    # Daily only (faster)
    python -m aoc_analytics.scripts.backfill_weather --daily-only --start-date 2021-07-01 --end-date 2024-12-31
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import List, Optional

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

# Default database path
DEFAULT_DB_PATH = Path(__file__).parent.parent.parent.parent / "aoc_sales.db"


def create_weather_tables(conn: sqlite3.Connection) -> None:
    """Create weather tables if they don't exist."""
    
    cursor = conn.cursor()
    
    # Hourly weather observations
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            datetime TEXT NOT NULL,
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            temp_c REAL,
            feels_like_c REAL,
            precip_mm REAL,
            rain_mm REAL,
            snow_mm REAL,
            cloud_cover_pct REAL,
            humidity_pct REAL,
            wind_kph REAL,
            weather_code INTEGER,
            condition TEXT,
            precip_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(location, datetime)
        )
    """)
    
    # Daily weather summaries
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            location TEXT NOT NULL,
            date TEXT NOT NULL,
            temp_max_c REAL,
            temp_min_c REAL,
            temp_avg_c REAL,
            feels_like_max_c REAL,
            feels_like_min_c REAL,
            precip_mm REAL,
            rain_mm REAL,
            snow_mm REAL,
            precip_hours REAL,
            wind_max_kph REAL,
            weather_code INTEGER,
            condition TEXT,
            precip_type TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(location, date)
        )
    """)
    
    # Create indexes for fast lookups
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_hourly_lookup 
        ON weather_hourly(location, date, hour)
    """)
    
    cursor.execute("""
        CREATE INDEX IF NOT EXISTS idx_weather_daily_lookup 
        ON weather_daily(location, date)
    """)
    
    conn.commit()
    logger.info("Weather tables created/verified")


def create_sales_weather_fact_view(conn: sqlite3.Connection) -> None:
    """
    Create a view that joins sales with weather data.
    This is what the conditions.py module expects.
    """
    
    cursor = conn.cursor()
    
    # Drop existing view if it exists
    cursor.execute("DROP VIEW IF EXISTS sales_weather_fact")
    
    # Create the joined view
    cursor.execute("""
        CREATE VIEW sales_weather_fact AS
        SELECT 
            s.id,
            s.date,
            s.time,
            s.datetime,
            s.location,
            s.sku,
            s.product_name,
            s.category,
            s.quantity,
            s.subtotal,
            s.gross_profit,
            
            -- Weather data (hourly)
            wh.temp_c,
            wh.feels_like_c,
            wh.precip_mm,
            wh.precip_type,
            wh.cloud_cover_pct AS cloud_cover,
            wh.humidity_pct AS humidity,
            wh.wind_kph,
            wh.weather_code,
            wh.condition AS weather_condition,
            
            -- Daily weather summaries for broader context
            wd.temp_max_c AS daily_temp_max_c,
            wd.temp_min_c AS daily_temp_min_c,
            wd.precip_mm AS daily_precip_mm,
            
            -- Derived fields
            CAST(strftime('%w', s.date) AS INTEGER) AS day_of_week,
            CAST(strftime('%H', s.time) AS INTEGER) AS hour,
            strftime('%Y-%m', s.date) AS month,
            CAST(strftime('%Y', s.date) AS INTEGER) AS year
            
        FROM sales s
        LEFT JOIN weather_hourly wh 
            ON s.location = wh.location 
            AND s.date = wh.date 
            AND CAST(strftime('%H', s.time) AS INTEGER) = wh.hour
        LEFT JOIN weather_daily wd
            ON s.location = wd.location
            AND s.date = wd.date
    """)
    
    conn.commit()
    logger.info("Created sales_weather_fact view")


def backfill_weather(
    conn: sqlite3.Connection,
    locations: List[str],
    start_date: str,
    end_date: str,
    hourly: bool = True,
    daily: bool = True,
    chunk_days: int = 90,
) -> dict:
    """
    Backfill weather data for given locations and date range.
    
    Args:
        conn: Database connection
        locations: List of location names
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)
        hourly: Whether to fetch hourly data
        daily: Whether to fetch daily data
        chunk_days: Number of days per API request (Open-Meteo limit)
    
    Returns:
        Stats dict with counts
    """
    from aoc_analytics.core.weather import WeatherClient
    
    cursor = conn.cursor()
    stats = {
        "hourly_inserted": 0,
        "hourly_skipped": 0,
        "daily_inserted": 0,
        "daily_skipped": 0,
        "errors": 0,
    }
    
    start = date.fromisoformat(start_date)
    end = date.fromisoformat(end_date)
    
    for location in locations:
        logger.info(f"Processing weather for {location}")
        client = WeatherClient(location=location)
        
        # Process in chunks (Open-Meteo has limits on date ranges)
        current_start = start
        while current_start <= end:
            current_end = min(current_start + timedelta(days=chunk_days - 1), end)
            
            try:
                # Fetch hourly data
                if hourly:
                    logger.info(f"  Fetching hourly: {current_start} to {current_end}")
                    hourly_data = client.get_historical_hourly(
                        current_start.isoformat(),
                        current_end.isoformat()
                    )
                    
                    for obs in hourly_data:
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO weather_hourly (
                                    location, datetime, date, hour,
                                    temp_c, feels_like_c, precip_mm, rain_mm, snow_mm,
                                    cloud_cover_pct, humidity_pct, wind_kph,
                                    weather_code, condition, precip_type
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                location,
                                obs.datetime.isoformat(),
                                obs.datetime.strftime("%Y-%m-%d"),
                                obs.datetime.hour,
                                obs.temp_c,
                                obs.feels_like_c,
                                obs.precip_mm,
                                obs.rain_mm,
                                obs.snow_mm,
                                obs.cloud_cover_pct,
                                obs.humidity_pct,
                                obs.wind_kph,
                                obs.weather_code,
                                obs.condition,
                                obs.precip_type,
                            ))
                            if cursor.rowcount > 0:
                                stats["hourly_inserted"] += 1
                            else:
                                stats["hourly_skipped"] += 1
                        except Exception as e:
                            logger.warning(f"Error inserting hourly: {e}")
                            stats["errors"] += 1
                
                # Fetch daily data
                if daily:
                    logger.info(f"  Fetching daily: {current_start} to {current_end}")
                    daily_data = client.get_historical_daily(
                        current_start.isoformat(),
                        current_end.isoformat()
                    )
                    
                    for obs in daily_data:
                        try:
                            cursor.execute("""
                                INSERT OR IGNORE INTO weather_daily (
                                    location, date,
                                    temp_max_c, temp_min_c, temp_avg_c,
                                    feels_like_max_c, feels_like_min_c,
                                    precip_mm, rain_mm, snow_mm, precip_hours,
                                    wind_max_kph, weather_code, condition, precip_type
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                location,
                                obs.date.isoformat(),
                                obs.temp_max_c,
                                obs.temp_min_c,
                                (obs.temp_max_c + obs.temp_min_c) / 2,  # avg
                                obs.feels_like_max_c,
                                obs.feels_like_min_c,
                                obs.precip_mm,
                                obs.rain_mm,
                                obs.snow_mm,
                                obs.precip_hours,
                                obs.wind_max_kph,
                                obs.weather_code,
                                obs.condition,
                                obs.precip_type,
                            ))
                            if cursor.rowcount > 0:
                                stats["daily_inserted"] += 1
                            else:
                                stats["daily_skipped"] += 1
                        except Exception as e:
                            logger.warning(f"Error inserting daily: {e}")
                            stats["errors"] += 1
                
                conn.commit()
                
            except Exception as e:
                logger.error(f"Error fetching weather for {location} ({current_start} to {current_end}): {e}")
                stats["errors"] += 1
            
            current_start = current_end + timedelta(days=1)
    
    return stats


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical weather data from Open-Meteo"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=DEFAULT_DB_PATH,
        help="Path to SQLite database",
    )
    parser.add_argument(
        "--start-date",
        type=str,
        required=True,
        help="Start date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--end-date",
        type=str,
        required=True,
        help="End date (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--location",
        type=str,
        help="Specific location to backfill (default: all)",
    )
    parser.add_argument(
        "--daily-only",
        action="store_true",
        help="Only fetch daily summaries (much faster)",
    )
    parser.add_argument(
        "--hourly-only",
        action="store_true",
        help="Only fetch hourly data",
    )
    parser.add_argument(
        "--create-view",
        action="store_true",
        help="Create/update sales_weather_fact view after backfill",
    )
    
    args = parser.parse_args()
    
    # Validate dates
    try:
        date.fromisoformat(args.start_date)
        date.fromisoformat(args.end_date)
    except ValueError as e:
        logger.error(f"Invalid date format: {e}")
        sys.exit(1)
    
    # Determine which data to fetch
    fetch_hourly = not args.daily_only
    fetch_daily = not args.hourly_only
    
    if not fetch_hourly and not fetch_daily:
        logger.error("Cannot specify both --daily-only and --hourly-only")
        sys.exit(1)
    
    # Determine locations
    from aoc_analytics.core.weather import STORE_LOCATIONS
    
    if args.location:
        if args.location not in STORE_LOCATIONS:
            logger.error(f"Unknown location: {args.location}")
            logger.info(f"Available locations: {', '.join(STORE_LOCATIONS.keys())}")
            sys.exit(1)
        locations = [args.location]
    else:
        locations = list(STORE_LOCATIONS.keys())
    
    # Connect to database
    logger.info(f"Using database: {args.db}")
    conn = sqlite3.connect(args.db)
    
    # Create tables
    create_weather_tables(conn)
    
    # Backfill weather
    logger.info(f"Backfilling weather for {len(locations)} location(s)")
    logger.info(f"Date range: {args.start_date} to {args.end_date}")
    logger.info(f"Fetching: {'hourly + daily' if fetch_hourly and fetch_daily else 'daily only' if fetch_daily else 'hourly only'}")
    
    stats = backfill_weather(
        conn=conn,
        locations=locations,
        start_date=args.start_date,
        end_date=args.end_date,
        hourly=fetch_hourly,
        daily=fetch_daily,
    )
    
    # Create view if requested
    if args.create_view:
        create_sales_weather_fact_view(conn)
    
    # Summary
    logger.info("=" * 50)
    logger.info("Backfill Complete!")
    logger.info(f"  Hourly records inserted: {stats['hourly_inserted']}")
    logger.info(f"  Hourly records skipped: {stats['hourly_skipped']}")
    logger.info(f"  Daily records inserted: {stats['daily_inserted']}")
    logger.info(f"  Daily records skipped: {stats['daily_skipped']}")
    logger.info(f"  Errors: {stats['errors']}")
    
    conn.close()


if __name__ == "__main__":
    main()
