"""
AOC Scheduled Jobs

Background tasks that keep the AOC system fresh:
1. Hourly: Sync sales data from UpStock
2. Hourly: Fetch weather data for all stores
3. Weekly: Recalculate adaptive weights
4. Daily: Generate mood features for the day

Can be run as:
- Cloud Run Jobs (production)
- Cron jobs (self-hosted)
- Manual CLI invocation
"""

from __future__ import annotations

import logging
import os
import sys
from datetime import datetime, date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


# =============================================================================
# JOB DEFINITIONS
# =============================================================================

def job_sync_sales(
    days_back: int = 7,
    store_ids: Optional[list] = None,
) -> dict:
    """
    Sync recent sales from UpStock to local database.
    
    Run: Hourly
    Purpose: Keep sales data fresh for analysis
    """
    logger.info(f"Starting sales sync job (days_back={days_back})")
    
    try:
        # Import here to avoid circular imports
        from aoc_analytics.scripts.backfill_sales import sync_sales_data
        
        result = sync_sales_data(
            days_back=days_back,
            store_ids=store_ids,
        )
        
        logger.info(f"Sales sync complete: {result}")
        return {"status": "success", "result": result}
    
    except Exception as e:
        logger.error(f"Sales sync failed: {e}")
        return {"status": "error", "error": str(e)}


def job_fetch_weather(
    stores: Optional[list] = None,
) -> dict:
    """
    Fetch current weather for all stores.
    
    Run: Hourly
    Purpose: Keep weather data fresh for regime detection
    """
    logger.info("Starting weather fetch job")
    
    try:
        from aoc_analytics.core.weather import get_all_store_weather, save_weather_to_db
        
        # Get database connection using the shared helper
        conn, db_type = _get_database_connection()
        
        try:
            # Fetch weather for all stores
            weather_clients = get_all_store_weather()
            
            # Save to database
            saved_count = save_weather_to_db(weather_clients, conn)
            
            logger.info(f"Weather fetch complete: {saved_count} records saved")
            return {"status": "success", "saved_count": saved_count, "db_type": db_type}
        finally:
            conn.close()
    
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
        return {"status": "error", "error": str(e)}


def job_backfill_weather(
    start_date: str,
    end_date: str,
    locations: Optional[list] = None,
) -> dict:
    """
    Backfill historical weather data from Open-Meteo.
    
    Run: On-demand (via API)
    Purpose: Populate weather_hourly table with historical data
    
    Args:
        start_date: Start date (YYYY-MM-DD)
        end_date: End date (YYYY-MM-DD)  
        locations: Optional list of locations (defaults to all stores)
    """
    logger.info(f"Starting weather backfill: {start_date} to {end_date}")
    
    try:
        from aoc_analytics.core.weather import backfill_weather
        
        # Get database connection using shared helper
        conn, db_type = _get_database_connection()
        
        try:
            stats = backfill_weather(conn, start_date, end_date, locations)
            stats["db_type"] = db_type
            
            logger.info(f"Weather backfill complete: {stats}")
            return {"status": "success", **stats}
        finally:
            conn.close()
    
    except Exception as e:
        logger.error(f"Weather backfill failed: {e}")
        return {"status": "error", "error": str(e)}


def job_recalculate_weights(
    lookback_days: int = 365,
    activate: bool = True,
) -> dict:
    """
    Recalculate adaptive weights from recent data.
    
    Run: Weekly (Sunday night)
    Purpose: Keep regime weights fresh as shopping patterns evolve
    """
    logger.info(f"Starting weight recalculation job (lookback={lookback_days})")
    
    try:
        from aoc_analytics.core.adaptive_weights import recalculate_and_save_weights
        
        version = recalculate_and_save_weights(
            lookback_days=lookback_days,
            activate=activate,
        )
        
        logger.info(f"Weight recalculation complete: version {version}")
        return {"status": "success", "version": version}
    
    except Exception as e:
        logger.error(f"Weight recalculation failed: {e}")
        return {"status": "error", "error": str(e)}


def _get_database_connection():
    """
    Get a database connection - supports both SQLite and PostgreSQL.
    
    Uses AOC_DATABASE_URL for PostgreSQL (production) or 
    AOC_DATABASE_PATH for SQLite (local dev).
    """
    import sqlite3
    
    # Check for PostgreSQL URL first (production)
    db_url = os.environ.get("AOC_DATABASE_URL")
    if db_url and db_url.startswith("postgresql"):
        try:
            import psycopg2
            from urllib.parse import urlparse, parse_qs
            
            parsed = urlparse(db_url)
            
            # Extract connection params
            user = parsed.username
            password = parsed.password
            database = parsed.path.lstrip('/')
            
            # Handle Cloud SQL socket vs regular host
            query_params = parse_qs(parsed.query)
            if 'host' in query_params:
                # Cloud SQL Unix socket
                host = query_params['host'][0]
                conn = psycopg2.connect(
                    dbname=database,
                    user=user,
                    password=password,
                    host=host,
                )
            else:
                # Regular TCP connection
                host = parsed.hostname or 'localhost'
                port = parsed.port or 5432
                conn = psycopg2.connect(
                    dbname=database,
                    user=user,
                    password=password,
                    host=host,
                    port=port,
                )
            
            logger.info(f"Connected to PostgreSQL database: {database}")
            return conn, "postgresql"
        except Exception as e:
            logger.warning(f"PostgreSQL connection failed: {e}, falling back to SQLite")
    
    # Fall back to SQLite
    db_path = os.environ.get("AOC_DATABASE_PATH", "weather_sales.db")
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    logger.info(f"Connected to SQLite database: {db_path}")
    return conn, "sqlite"


def job_generate_mood_features(
    target_date: Optional[date] = None,
    locations: Optional[list] = None,
) -> dict:
    """
    Generate daily mood features for all stores.
    
    Run: Daily (early morning)
    Purpose: Pre-compute signals for the day
    
    Args:
        target_date: Date to generate features for (default: today)
        locations: List of locations to process (default: all from database)
    """
    if target_date is None:
        target_date = date.today()
    
    logger.info(f"Starting mood features job for {target_date}")
    
    try:
        from aoc_analytics.core.signals.builder import rebuild_behavioral_signals
        
        conn, db_type = _get_database_connection()
        
        try:
            cursor = conn.cursor()
            
            # Get all locations from database if not specified
            if locations is None:
                try:
                    cursor.execute("SELECT DISTINCT location FROM sales WHERE location IS NOT NULL")
                    locations = [row[0] for row in cursor.fetchall()]
                except Exception:
                    # sales table doesn't exist - use default locations
                    # Rollback to clear any transaction errors
                    try:
                        conn.rollback()
                    except Exception:
                        pass
                    locations = ["Kingsway", "Victoria Drive", "Parksville"]
            
            if not locations:
                # Default to known store locations
                locations = ["Kingsway", "Victoria Drive", "Parksville"]
            
            results = {}
            total_records = 0
            
            date_str = target_date.isoformat()
            
            for location in locations:
                try:
                    count = rebuild_behavioral_signals(
                        conn,
                        location=location,
                        start_date=date_str,
                        end_date=date_str,
                    )
                    results[location] = {"status": "success", "records": count}
                    total_records += count
                except Exception as e:
                    logger.error(f"Failed for location {location}: {e}")
                    results[location] = {"status": "error", "error": str(e)}
                    # Rollback the failed transaction to allow subsequent operations
                    try:
                        conn.rollback()
                    except Exception:
                        pass
            
            try:
                conn.commit()
            except Exception:
                pass
            
            logger.info(f"Mood features complete: {total_records} records across {len(locations)} locations")
            return {"status": "success", "total_records": total_records, "locations": results, "db_type": db_type}
        
        finally:
            conn.close()
    
    except Exception as e:
        logger.error(f"Mood features failed: {e}")
        return {"status": "error", "error": str(e)}


# =============================================================================
# ORCHESTRATOR - Run all jobs in sequence
# =============================================================================

def run_hourly_jobs() -> dict:
    """Run all hourly jobs."""
    results = {}
    
    results["weather"] = job_fetch_weather()
    results["sales"] = job_sync_sales(days_back=1)
    
    return results


def run_daily_jobs() -> dict:
    """Run all daily jobs."""
    results = {}
    
    results["mood_features"] = job_generate_mood_features()
    
    return results


def run_weekly_jobs() -> dict:
    """Run all weekly jobs."""
    results = {}
    
    results["weights"] = job_recalculate_weights()
    
    return results


# =============================================================================
# CLOUD RUN JOB ENTRY POINT
# =============================================================================

def cloud_run_handler(job_type: str) -> dict:
    """
    Entry point for Cloud Run Jobs.
    
    Set JOB_TYPE environment variable to:
    - hourly
    - daily
    - weekly
    - sync_sales
    - fetch_weather
    - recalculate_weights
    - mood_features
    """
    logger.info(f"Cloud Run job starting: {job_type}")
    
    if job_type == "hourly":
        return run_hourly_jobs()
    elif job_type == "daily":
        return run_daily_jobs()
    elif job_type == "weekly":
        return run_weekly_jobs()
    elif job_type == "sync_sales":
        return job_sync_sales()
    elif job_type == "fetch_weather":
        return job_fetch_weather()
    elif job_type == "recalculate_weights":
        return job_recalculate_weights()
    elif job_type == "mood_features":
        return job_generate_mood_features()
    else:
        return {"status": "error", "error": f"Unknown job type: {job_type}"}


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
    )
    
    parser = argparse.ArgumentParser(description="AOC Scheduled Jobs")
    parser.add_argument(
        "job", 
        choices=[
            "hourly", "daily", "weekly",
            "sync_sales", "fetch_weather", "recalculate_weights", "mood_features"
        ],
        help="Job to run"
    )
    parser.add_argument("--lookback", type=int, default=365, help="Days of data for weight calculation")
    parser.add_argument("--days-back", type=int, default=7, help="Days of sales to sync")
    
    args = parser.parse_args()
    
    # Check for Cloud Run environment
    job_type = os.environ.get("JOB_TYPE", args.job)
    
    result = cloud_run_handler(job_type)
    
    print(f"\n{'='*60}")
    print(f"Job: {job_type}")
    print(f"Result: {result}")
    print(f"{'='*60}")
    
    # Exit with error code if job failed
    if result.get("status") == "error":
        sys.exit(1)
