"""
AOC Analytics REST API Server.

This module provides a FastAPI application for serving analytics
to external consumers like LocalBot. It bridges the gap between
the Python analytics engine and external inventory management systems.

Endpoints follow the contracts defined in INGEST_DOCS/AOC_API_CONTRACT.md
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import asynccontextmanager
from datetime import date, datetime
from typing import Any, Optional

from fastapi import Depends, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from ..core import (
    Anomaly,
    AnomalyType,
    HeroWeightingConfig,
    SimilarityConfig,
    build_sku_behavior_signals,
    create_anomaly,
    delete_anomaly,
    forecast_demand_for_slot,
    get_anomalies_for_date_range,
    init_anomaly_table,
    list_anomalies,
    load_conditions_df,
    seed_anomalies,
)
from ..core.signals.builder import rebuild_behavioral_signals

logger = logging.getLogger(__name__)

# Database path - configurable via environment
DATABASE_PATH = os.environ.get("AOC_DATABASE_PATH", "weather_sales.db")


# =============================================================================
# Pydantic Models for API
# =============================================================================


class WeatherConditions(BaseModel):
    """Weather conditions for a forecast request."""

    temp_c: float = Field(..., description="Temperature in Celsius")
    feels_like_c: Optional[float] = Field(None, description="Feels-like temperature")
    precip_mm: float = Field(0.0, description="Precipitation in mm")
    precip_type: str = Field("none", description="Type of precipitation")
    cloud_cover_pct: float = Field(0.0, description="Cloud cover percentage")
    humidity: float = Field(50.0, description="Humidity percentage")
    wind_kph: float = Field(0.0, description="Wind speed in km/h")


class CalendarContext(BaseModel):
    """Calendar context for a forecast request."""

    dow: int = Field(..., ge=0, le=6, description="Day of week (0=Monday)")
    hour: int = Field(..., ge=0, le=23, description="Hour of day")
    is_holiday: bool = Field(False, description="Whether it's a holiday")
    is_preholiday: bool = Field(False, description="Day before holiday")
    is_payday_window: bool = Field(False, description="Payday proximity window")
    has_home_game: bool = Field(False, description="Local sports home game")
    has_concert: bool = Field(False, description="Major concert event")
    has_festival: bool = Field(False, description="Festival or large event")


class ForecastRequest(BaseModel):
    """Request for demand forecast."""

    target_date: str = Field(..., description="Target date (YYYY-MM-DD)")
    target_hour: int = Field(..., ge=0, le=23, description="Target hour")
    weather: WeatherConditions
    calendar: CalendarContext
    category_path: Optional[str] = Field(None, description="Filter by category")
    sku: Optional[str] = Field(None, description="Filter by specific SKU")
    top_k: int = Field(10, description="Number of similar days to use")


class ForecastResponse(BaseModel):
    """Response from demand forecast."""

    target_date: str
    target_hour: int
    predicted_revenue: float
    predicted_units: float
    confidence: float
    similar_days_used: int
    weather_context: dict[str, Any]
    methodology: str = "matched-conditions-weighted-average"


class SignalsBuildRequest(BaseModel):
    """Request to rebuild behavioral signals."""

    start_date: Optional[str] = Field(None, description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")


class SignalsBuildResponse(BaseModel):
    """Response from signals rebuild."""

    rows_processed: int
    date_range: dict[str, str]
    duration_seconds: float


class AnomalyRequest(BaseModel):
    """Request to create/update an anomaly."""

    anomaly_type: str = Field(..., description="Type of anomaly")
    name: str = Field(..., description="Human-readable name")
    start_date: str = Field(..., description="Start date (YYYY-MM-DD)")
    end_date: Optional[str] = Field(None, description="End date (YYYY-MM-DD)")
    severity: float = Field(0.5, ge=0.0, le=1.0, description="Trust level 0-1")
    affected_categories: Optional[list[str]] = Field(None)
    affected_skus: Optional[list[str]] = Field(None)
    supply_channel: Optional[str] = Field(None)
    notes: Optional[str] = Field(None)


class AnomalyResponse(BaseModel):
    """Response for anomaly operations."""

    id: int
    anomaly_type: str
    name: str
    start_date: str
    end_date: Optional[str]
    severity: float
    notes: Optional[str]


class HeroScoreRequest(BaseModel):
    """Request for hero SKU scoring."""

    start_date: Optional[str] = Field(None)
    end_date: Optional[str] = Field(None)
    min_observations: int = Field(5)
    category_path: Optional[str] = Field(None)


class HeroScoreResponse(BaseModel):
    """Response with hero scores."""

    skus: list[dict[str, Any]]
    total_skus: int
    config_used: dict[str, Any]


class HealthResponse(BaseModel):
    """Health check response."""

    status: str
    version: str
    database_connected: bool
    database_path: str


# =============================================================================
# Database Connection
# =============================================================================


def get_db_connection() -> sqlite3.Connection:
    """Get a database connection."""
    # check_same_thread=False is required for FastAPI which uses thread pools
    # Each request gets its own connection which is closed after use
    conn = sqlite3.connect(DATABASE_PATH, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    return conn


def get_db():
    """Dependency for database connection."""
    conn = get_db_connection()
    try:
        yield conn
    finally:
        conn.close()


# =============================================================================
# Application Factory
# =============================================================================


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan handler."""
    # Startup
    logger.info(f"AOC Analytics API starting up with database: {DATABASE_PATH}")
    conn = get_db_connection()
    try:
        init_anomaly_table(conn)
        seed_anomalies(conn)
    finally:
        conn.close()
    logger.info("Database initialized")

    yield

    # Shutdown
    logger.info("AOC Analytics API shutting down")


def create_app(db_path: Optional[str] = None) -> FastAPI:
    """Create the FastAPI application.

    Args:
        db_path: Optional database path override.

    Returns:
        Configured FastAPI application.
    """
    global DATABASE_PATH
    if db_path:
        DATABASE_PATH = db_path

    app = FastAPI(
        title="AOC Analytics API",
        description="Weather-aware demand analytics for LocalBot integration",
        version="1.0.0",
        lifespan=lifespan,
    )

    # CORS for browser-based clients
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Register routes
    app.include_router(health_router)
    app.include_router(forecast_router)
    app.include_router(signals_router)
    app.include_router(anomaly_router)
    app.include_router(hero_router)
    
    # JFK integration routes
    try:
        from .jfk_endpoints import jfk_router
        app.include_router(jfk_router)
        logger.info("JFK integration routes registered")
    except ImportError as e:
        logger.warning(f"JFK routes not available: {e}")

    return app


# =============================================================================
# Routers
# =============================================================================

from fastapi import APIRouter

health_router = APIRouter(tags=["health"])
forecast_router = APIRouter(prefix="/api/v1/forecast", tags=["forecast"])
signals_router = APIRouter(prefix="/api/v1/signals", tags=["signals"])
anomaly_router = APIRouter(prefix="/api/v1/anomalies", tags=["anomalies"])
hero_router = APIRouter(prefix="/api/v1/hero", tags=["hero"])


# =============================================================================
# Health Endpoints
# =============================================================================


@health_router.get("/health", response_model=HealthResponse)
def health_check():
    """Health check endpoint."""
    db_ok = False
    try:
        conn = get_db_connection()
        conn.execute("SELECT 1")
        conn.close()
        db_ok = True
    except Exception:
        pass

    return HealthResponse(
        status="healthy" if db_ok else "degraded",
        version="1.0.0",
        database_connected=db_ok,
        database_path=DATABASE_PATH,
    )


# =============================================================================
# Forecast Endpoints
# =============================================================================


@forecast_router.post("/demand", response_model=ForecastResponse)
def forecast_demand(request: ForecastRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Forecast demand for a specific time slot.

    Uses matched-conditions forecasting with weather-aware similarity.
    """
    import time

    start_time = time.time()

    try:
        # Load historical conditions
        conditions_df = load_conditions_df(conn)
        if conditions_df.empty:
            raise HTTPException(
                status_code=503,
                detail="No historical data available for forecasting",
            )

        # Build target conditions dict
        target_conditions = {
            "date": date.fromisoformat(request.target_date),
            "hour": request.target_hour,
            "temp_c": request.weather.temp_c,
            "feels_like_c": request.weather.feels_like_c or request.weather.temp_c,
            "precip_mm": request.weather.precip_mm,
            "precip_type": request.weather.precip_type,
            "cloud_cover_pct": request.weather.cloud_cover_pct,
            "humidity": request.weather.humidity,
            "wind_kph": request.weather.wind_kph,
            "dow": request.calendar.dow,
            "is_holiday": int(request.calendar.is_holiday),
            "is_preholiday": int(request.calendar.is_preholiday),
            "is_payday_window": int(request.calendar.is_payday_window),
            "has_home_game": int(request.calendar.has_home_game),
            "has_concert": int(request.calendar.has_concert),
            "has_festival": int(request.calendar.has_festival),
        }

        # Run forecast
        result = forecast_demand_for_slot(
            target_conditions=target_conditions,
            conditions_df=conditions_df,
            top_k=request.top_k,
        )

        elapsed = time.time() - start_time

        return ForecastResponse(
            target_date=request.target_date,
            target_hour=request.target_hour,
            predicted_revenue=result.get("predicted_revenue", 0.0),
            predicted_units=result.get("predicted_units", 0.0),
            confidence=result.get("confidence", 0.0),
            similar_days_used=result.get("similar_days_used", 0),
            weather_context={
                "temp_c": request.weather.temp_c,
                "precip_mm": request.weather.precip_mm,
                "conditions": request.weather.precip_type,
            },
        )

    except Exception as e:
        logger.error(f"Forecast error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Signals Endpoints
# =============================================================================


@signals_router.post("/rebuild", response_model=SignalsBuildResponse)
def rebuild_signals(request: SignalsBuildRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Rebuild behavioral signals from weather data.

    This recomputes the weather → behavioral propensity transformation.
    """
    import time

    start_time = time.time()

    try:
        result = rebuild_behavioral_signals(
            conn=conn,
            start_date=request.start_date,
            end_date=request.end_date,
        )

        elapsed = time.time() - start_time

        return SignalsBuildResponse(
            rows_processed=result.get("rows_processed", 0),
            date_range={
                "start": result.get("start_date", ""),
                "end": result.get("end_date", ""),
            },
            duration_seconds=elapsed,
        )

    except Exception as e:
        logger.error(f"Signals rebuild error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@signals_router.get("/current")
def get_current_signals(
    date_str: Optional[str] = Query(None, alias="date"),
    conn: sqlite3.Connection = Depends(get_db),
):
    """Get current behavioral signals for a date."""
    target_date = date_str or date.today().isoformat()

    try:
        cur = conn.cursor()
        cur.execute(
            """
            SELECT date, at_home, out_and_about, holiday, local_vibe
            FROM behavioral_signals_fact
            WHERE date = ?
            """,
            (target_date,),
        )
        row = cur.fetchone()

        if row is None:
            raise HTTPException(
                status_code=404,
                detail=f"No signals found for date {target_date}",
            )

        return {
            "date": row[0],
            "at_home": row[1],
            "out_and_about": row[2],
            "holiday": row[3],
            "local_vibe": row[4],
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Get signals error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Anomaly Endpoints
# =============================================================================


@anomaly_router.get("/", response_model=list[AnomalyResponse])
def list_all_anomalies(
    include_expired: bool = Query(True),
    anomaly_type: Optional[str] = Query(None),
    conn: sqlite3.Connection = Depends(get_db),
):
    """List all registered anomalies."""
    try:
        a_type = AnomalyType(anomaly_type) if anomaly_type else None
        anomalies = list_anomalies(conn, include_expired=include_expired, anomaly_type=a_type)

        return [
            AnomalyResponse(
                id=a.id,
                anomaly_type=a.anomaly_type.value,
                name=a.name,
                start_date=a.start_date.isoformat(),
                end_date=a.end_date.isoformat() if a.end_date else None,
                severity=a.severity,
                notes=a.notes,
            )
            for a in anomalies
        ]

    except Exception as e:
        logger.error(f"List anomalies error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@anomaly_router.post("/", response_model=AnomalyResponse)
def create_new_anomaly(request: AnomalyRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Create a new anomaly record."""
    try:
        anomaly = Anomaly(
            anomaly_type=AnomalyType(request.anomaly_type),
            name=request.name,
            start_date=date.fromisoformat(request.start_date),
            end_date=date.fromisoformat(request.end_date) if request.end_date else None,
            severity=request.severity,
            affected_categories=request.affected_categories,
            affected_skus=request.affected_skus,
            supply_channel=request.supply_channel,
            notes=request.notes,
        )

        created = create_anomaly(anomaly, conn)

        return AnomalyResponse(
            id=created.id,
            anomaly_type=created.anomaly_type.value,
            name=created.name,
            start_date=created.start_date.isoformat(),
            end_date=created.end_date.isoformat() if created.end_date else None,
            severity=created.severity,
            notes=created.notes,
        )

    except Exception as e:
        logger.error(f"Create anomaly error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@anomaly_router.delete("/{anomaly_id}")
def delete_existing_anomaly(anomaly_id: int, conn: sqlite3.Connection = Depends(get_db)):
    """Delete an anomaly record."""
    try:
        deleted = delete_anomaly(anomaly_id, conn)
        if not deleted:
            raise HTTPException(status_code=404, detail="Anomaly not found")
        return {"deleted": True, "id": anomaly_id}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Delete anomaly error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Hero Scoring Endpoints
# =============================================================================


@hero_router.post("/scores", response_model=HeroScoreResponse)
def compute_hero_scores(request: HeroScoreRequest, conn: sqlite3.Connection = Depends(get_db)):
    """Compute hero scores for SKUs.

    Returns SKUs ranked by their predictability/stability.
    """
    import pandas as pd

    try:
        # Build query for SKU-level data
        where_clauses = []
        params = {}

        if request.start_date:
            where_clauses.append("date >= :start_date")
            params["start_date"] = request.start_date
        if request.end_date:
            where_clauses.append("date <= :end_date")
            params["end_date"] = request.end_date
        if request.category_path:
            where_clauses.append("category_path = :category_path")
            params["category_path"] = request.category_path

        where_sql = ""
        if where_clauses:
            where_sql = "WHERE " + " AND ".join(where_clauses)

        sql = f"""
            SELECT
                sku,
                datetime(date || ' ' || printf('%02d:00:00', hour)) as datetime_local,
                quantity,
                date(date) as date,
                hour,
                CASE
                    WHEN strftime('%w', date) = '0' THEN 'Sunday'
                    WHEN strftime('%w', date) = '1' THEN 'Monday'
                    WHEN strftime('%w', date) = '2' THEN 'Tuesday'
                    WHEN strftime('%w', date) = '3' THEN 'Wednesday'
                    WHEN strftime('%w', date) = '4' THEN 'Thursday'
                    WHEN strftime('%w', date) = '5' THEN 'Friday'
                    ELSE 'Saturday'
                END as day_of_week,
                temp_c
            FROM sales_weather_fact
            {where_sql}
        """

        df = pd.read_sql_query(sql, conn, params=params)

        if df.empty:
            return HeroScoreResponse(skus=[], total_skus=0, config_used={})

        # Convert datetime column
        df["datetime_local"] = pd.to_datetime(df["datetime_local"])
        df["date"] = pd.to_datetime(df["date"])

        # Compute hero scores
        config = HeroWeightingConfig(min_observations=request.min_observations)
        scores_df = build_sku_behavior_signals(df, config)

        # Convert to response format
        skus = scores_df.sort_values("hero_score", ascending=False).head(100).to_dict("records")

        return HeroScoreResponse(
            skus=skus,
            total_skus=len(scores_df),
            config_used={
                "min_observations": config.min_observations,
                "interval_std_target": config.interval_std_target,
            },
        )

    except Exception as e:
        logger.error(f"Hero scores error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# =============================================================================
# Default Application Instance
# =============================================================================

app = create_app()


# =============================================================================
# CLI Entry Point
# =============================================================================

def main():
    """CLI entry point for aoc-server command."""
    import argparse
    import uvicorn

    parser = argparse.ArgumentParser(description="AOC Analytics API Server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8081, help="Port to listen on")
    parser.add_argument("--db", default=None, help="Path to database file")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    args = parser.parse_args()

    if args.db:
        global DATABASE_PATH
        DATABASE_PATH = args.db

    uvicorn.run(
        "aoc_analytics.api.server:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
    )


if __name__ == "__main__":
    main()
