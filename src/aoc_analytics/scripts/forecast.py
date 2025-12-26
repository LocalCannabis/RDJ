#!/usr/bin/env python3
"""
AOC Forecast CLI

Command-line interface for running forecasts and backtests.

Usage:
    # Generate a 7-day forecast for Parksville
    python -m aoc_analytics.scripts.forecast --store Parksville --days 7
    
    # Run a backtest
    python -m aoc_analytics.scripts.forecast backtest --store Parksville --train-days 365
    
    # Generate purchase suggestions
    python -m aoc_analytics.scripts.forecast purchase --store Parksville

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
import sys
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Optional

import pandas as pd

# Add parent to path for local development
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from aoc_analytics.core.forecast_engine import ForecastEngine, ForecastConfig
from aoc_analytics.core.backtesting import (
    BacktestRunner,
    PurchaseOptimizer,
    print_backtest_report,
    run_backtest,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_sales_data(db_path: str, store: Optional[str] = None) -> pd.DataFrame:
    """Load sales data from AOC SQLite database."""
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            date,
            location,
            category,
            product_name,
            sku,
            quantity,
            subtotal,
            unit_price
        FROM sales
    """
    
    if store:
        query += f" WHERE location = '{store}'"
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    logger.info(f"Loaded {len(df):,} sales records")
    return df


def load_weather_data(db_path: str, store: Optional[str] = None) -> pd.DataFrame:
    """Load weather data from AOC SQLite database."""
    conn = sqlite3.connect(db_path)
    
    query = """
        SELECT 
            datetime as date,
            location,
            temperature_2m as temp_c,
            precipitation as precip_mm,
            weather_code
        FROM weather_hourly
    """
    
    if store:
        query += f" WHERE location = '{store}'"
    
    try:
        df = pd.read_sql_query(query, conn)
        logger.info(f"Loaded {len(df):,} weather records")
    except Exception as e:
        logger.warning(f"Could not load weather data: {e}")
        df = pd.DataFrame()
    
    conn.close()
    return df


def cmd_forecast(args):
    """Generate a demand forecast."""
    logger.info(f"Generating {args.days}-day forecast for {args.store}")
    
    # Load data
    sales_df = load_sales_data(args.db, args.store)
    weather_df = load_weather_data(args.db, args.store) if not args.no_weather else None
    
    if len(sales_df) == 0:
        logger.error("No sales data found")
        return 1
    
    # Create engine and load data
    config = ForecastConfig(
        k_neighbors=args.neighbors,
        forecast_days=args.days,
    )
    engine = ForecastEngine(store=args.store, config=config)
    engine.load_historical_data(sales_df, weather_df)
    
    # Generate forecast
    start_date = date.today() + timedelta(days=1)
    forecast = engine.forecast_week(start_date=start_date)
    
    # Output
    if args.output == "json":
        output = {
            "store": forecast.store,
            "forecast_date": str(forecast.forecast_date),
            "week_start": str(forecast.week_start),
            "week_end": str(forecast.week_end),
            "total_expected_revenue": round(forecast.total_expected_revenue, 2),
            "total_expected_units": round(forecast.total_expected_units, 0),
            "avg_confidence": round(forecast.avg_confidence, 2),
            "peak_day": forecast.peak_day,
            "weather_impact": forecast.weather_impact,
            "notable_events": forecast.notable_events,
            "days": [
                {
                    "date": str(d.date),
                    "day": d.dow_name,
                    "temp_high": d.temp_high,
                    "temp_low": d.temp_low,
                    "weather": d.weather_desc,
                    "holiday": d.holiday_name,
                    "expected_revenue": round(d.expected_revenue, 2),
                    "expected_units": round(d.expected_units, 0),
                    "confidence": round(d.confidence, 2),
                    "regime": d.suggested_regime,
                }
                for d in forecast.days
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        # Table output
        df = engine.to_dataframe(forecast)
        print(f"\n{'='*80}")
        print(f"DEMAND FORECAST: {forecast.store}")
        print(f"{'='*80}")
        print(f"Week: {forecast.week_start} to {forecast.week_end}")
        print(f"Generated: {forecast.forecast_date}")
        print()
        print(df.to_string(index=False))
        print()
        print(f"Total Expected Revenue: ${forecast.total_expected_revenue:,.0f}")
        print(f"Total Expected Units:   {forecast.total_expected_units:,.0f}")
        print(f"Average Confidence:     {forecast.avg_confidence:.1%}")
        print(f"Peak Day:               {forecast.peak_day} (${forecast.peak_revenue:,.0f})")
        print()
        if forecast.notable_events:
            print(f"Notable Events: {', '.join(forecast.notable_events)}")
        print(f"Weather Impact: {forecast.weather_impact}")
        print()
    
    return 0


def cmd_backtest(args):
    """Run a backtest."""
    logger.info(f"Running backtest for {args.store}")
    
    # Load data
    sales_df = load_sales_data(args.db, args.store)
    weather_df = load_weather_data(args.db, args.store) if not args.no_weather else None
    
    if len(sales_df) == 0:
        logger.error("No sales data found")
        return 1
    
    # Parse dates
    start_date = None
    end_date = None
    if args.start:
        start_date = datetime.strptime(args.start, "%Y-%m-%d").date()
    if args.end:
        end_date = datetime.strptime(args.end, "%Y-%m-%d").date()
    
    # Run backtest
    result = run_backtest(
        store=args.store,
        sales_df=sales_df,
        weather_df=weather_df,
        train_days=args.train_days,
        test_days=args.test_days,
    )
    
    # Output
    if args.output == "json":
        output = {
            "store": result.store,
            "backtest_start": str(result.backtest_start),
            "backtest_end": str(result.backtest_end),
            "train_window_days": result.train_window_days,
            "test_window_days": result.test_window_days,
            "metrics": {
                "n_days": result.metrics.n_days,
                "revenue_mape": round(result.metrics.revenue_mape, 2),
                "revenue_mae": round(result.metrics.revenue_mae, 2),
                "revenue_bias": round(result.metrics.revenue_bias, 2),
                "units_mape": round(result.metrics.units_mape, 2),
                "overpredict_rate": round(result.metrics.overpredict_rate, 3),
                "severe_miss_rate": round(result.metrics.severe_miss_rate, 3),
                "confidence_correlation": round(result.metrics.confidence_correlation, 3),
            },
            "dow_mape": {str(k): round(v, 2) for k, v in result.metrics.dow_mape.items()},
            "recommendations": result.tuning_recommendations,
        }
        print(json.dumps(output, indent=2))
    else:
        print_backtest_report(result)
    
    # Save detailed results if requested
    if args.save_results:
        result.daily_results.to_csv(args.save_results, index=False)
        logger.info(f"Saved detailed results to {args.save_results}")
    
    return 0


def cmd_purchase(args):
    """Generate purchase suggestions."""
    logger.info(f"Generating purchase suggestions for {args.store}")
    
    # Load data
    sales_df = load_sales_data(args.db, args.store)
    weather_df = load_weather_data(args.db, args.store) if not args.no_weather else None
    
    if len(sales_df) == 0:
        logger.error("No sales data found")
        return 1
    
    # Create forecast
    engine = ForecastEngine(store=args.store)
    engine.load_historical_data(sales_df, weather_df)
    forecast = engine.forecast_week()
    
    # Get current inventory (placeholder - would come from inventory system)
    # In practice, integrate with JFK inventory API
    current_inventory = {}
    for day in forecast.days:
        for cat in day.category_forecasts.keys():
            if cat not in current_inventory:
                # Default to 50% of weekly demand as current stock
                current_inventory[cat] = int(day.category_forecasts[cat] / 25 * 3.5)
    
    # Generate suggestions
    optimizer = PurchaseOptimizer(
        lead_time_days=args.lead_time,
        safety_stock_days=args.safety_days,
    )
    suggestions = optimizer.generate_suggestions(forecast, current_inventory)
    
    # Output
    if args.output == "json":
        output = {
            "store": args.store,
            "forecast_period": f"{forecast.week_start} to {forecast.week_end}",
            "suggestions": [
                {
                    "category": s.category,
                    "current_stock": s.current_stock,
                    "forecast_demand": round(s.forecast_demand_units, 0),
                    "suggested_order": s.suggested_order_qty,
                    "stockout_risk": s.stockout_risk,
                    "overstock_risk": s.overstock_risk,
                    "confidence": round(s.confidence, 2),
                    "rationale": s.rationale,
                }
                for s in suggestions
            ],
        }
        print(json.dumps(output, indent=2))
    else:
        print(f"\n{'='*80}")
        print(f"PURCHASE SUGGESTIONS: {args.store}")
        print(f"{'='*80}")
        print(f"Forecast Period: {forecast.week_start} to {forecast.week_end}")
        print(f"Lead Time: {args.lead_time} days")
        print(f"Safety Stock: {args.safety_days} days")
        print()
        
        df = optimizer.to_dataframe(suggestions)
        print(df.to_string(index=False))
        print()
        
        high_risk = [s for s in suggestions if s.stockout_risk == "high"]
        if high_risk:
            print("⚠️  HIGH STOCKOUT RISK:")
            for s in high_risk:
                print(f"   - {s.category}: Order {s.suggested_order_qty} units urgently")
        print()
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="AOC Forecast CLI - Demand forecasting and backtesting"
    )
    parser.add_argument(
        "--db",
        default="aoc_sales.db",
        help="Path to AOC SQLite database (default: aoc_sales.db)"
    )
    parser.add_argument(
        "--store",
        default="Parksville",
        help="Store name (default: Parksville)"
    )
    parser.add_argument(
        "--output",
        choices=["table", "json"],
        default="table",
        help="Output format (default: table)"
    )
    parser.add_argument(
        "--no-weather",
        action="store_true",
        help="Don't use weather data"
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Verbose logging"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Forecast command
    forecast_parser = subparsers.add_parser("forecast", help="Generate demand forecast")
    forecast_parser.add_argument(
        "--days",
        type=int,
        default=7,
        help="Number of days to forecast (default: 7)"
    )
    forecast_parser.add_argument(
        "--neighbors",
        type=int,
        default=100,
        help="Number of similar days to use (default: 100)"
    )
    forecast_parser.set_defaults(func=cmd_forecast)
    
    # Backtest command
    backtest_parser = subparsers.add_parser("backtest", help="Run forecast backtest")
    backtest_parser.add_argument(
        "--train-days",
        type=int,
        default=365,
        help="Training window size in days (default: 365)"
    )
    backtest_parser.add_argument(
        "--test-days",
        type=int,
        default=7,
        help="Test window size in days (default: 7)"
    )
    backtest_parser.add_argument(
        "--start",
        help="Start date for backtest (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--end",
        help="End date for backtest (YYYY-MM-DD)"
    )
    backtest_parser.add_argument(
        "--save-results",
        help="Save detailed results to CSV file"
    )
    backtest_parser.set_defaults(func=cmd_backtest)
    
    # Purchase command
    purchase_parser = subparsers.add_parser("purchase", help="Generate purchase suggestions")
    purchase_parser.add_argument(
        "--lead-time",
        type=int,
        default=3,
        help="Supplier lead time in days (default: 3)"
    )
    purchase_parser.add_argument(
        "--safety-days",
        type=float,
        default=2.0,
        help="Safety stock in days of demand (default: 2.0)"
    )
    purchase_parser.set_defaults(func=cmd_purchase)
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Default to forecast if no command given
    if not args.command:
        args.command = "forecast"
        args.days = 7
        args.neighbors = 100
        args.func = cmd_forecast
    
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
