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
        
        # Fetch weather for all stores
        weather_data = get_all_store_weather(stores)
        
        # Save to database
        saved_count = save_weather_to_db(weather_data)
        
        logger.info(f"Weather fetch complete: {saved_count} records saved")
        return {"status": "success", "saved_count": saved_count}
    
    except Exception as e:
        logger.error(f"Weather fetch failed: {e}")
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


def job_generate_mood_features(
    target_date: Optional[date] = None,
) -> dict:
    """
    Generate daily mood features for all stores.
    
    Run: Daily (early morning)
    Purpose: Pre-compute signals for the day
    """
    if target_date is None:
        target_date = date.today()
    
    logger.info(f"Starting mood features job for {target_date}")
    
    try:
        from aoc_analytics.core.signals.builder import rebuild_behavioral_signals
        
        # Rebuild signals for target date
        result = rebuild_behavioral_signals(
            start_date=target_date,
            end_date=target_date,
        )
        
        logger.info(f"Mood features complete: {result}")
        return {"status": "success", "result": result}
    
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
