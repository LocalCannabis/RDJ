"""
AOC Forecast Engine

Predicts demand for the upcoming week using:
- Historical sales patterns
- Weather forecasts (from Open-Meteo)
- Calendar events (holidays, paydays, cannabis culture)
- Matched-conditions similarity scoring

The engine produces forecasts that feed into purchase recommendations.

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
All Rights Reserved. Proprietary and Confidential.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .calendar import (
    get_calendar_context,
    AOCCalendar,
)
from .predictor import (
    SimilarityConfig,
    build_condition_vector,
    compute_similarity,
    forecast_demand_for_slot,
)
from .predictor_fast import (
    FastSimilarityIndex,
    forecast_demand_fast,
    FAISS_AVAILABLE,
)
from .weather import WeatherClient, STORE_LOCATIONS
from .store_profiles import get_store_status, StoreStatus, is_store_active
from .signals.vibe_signals import VibeEngine, get_vibe_for_date_cached, preload_vibe_cache
from .signals.external_calendar import CalendarService

logger = logging.getLogger(__name__)

# Track which stores have already shown warnings (to avoid spam)
_warned_stores: set = set()

# Module-level singletons (reused across ForecastEngine instances)
_VIBE_ENGINE: Optional[VibeEngine] = None
_CALENDAR_SERVICE: Optional[CalendarService] = None


def _get_vibe_engine() -> VibeEngine:
    """Get or create module-level VibeEngine singleton."""
    global _VIBE_ENGINE
    if _VIBE_ENGINE is None:
        _VIBE_ENGINE = VibeEngine()
    return _VIBE_ENGINE


def _get_calendar_service() -> CalendarService:
    """Get or create module-level CalendarService singleton."""
    global _CALENDAR_SERVICE
    if _CALENDAR_SERVICE is None:
        _CALENDAR_SERVICE = CalendarService()
    return _CALENDAR_SERVICE


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class DayForecast:
    """Forecast for a single day."""
    date: date
    dow: int  # 0=Monday, 6=Sunday
    dow_name: str
    
    # Weather forecast
    temp_high: float
    temp_low: float
    temp_avg: float
    precip_mm: float
    weather_code: int
    weather_desc: str
    
    # Calendar context
    is_holiday: bool
    holiday_name: Optional[str]
    is_preholiday: bool
    is_payday_window: bool
    events: List[str]
    
    # Demand predictions (by category/product)
    expected_revenue: float
    expected_units: float
    confidence: float  # 0-1 based on match quality
    
    # Comparable days used
    similar_days_count: int
    similarity_score: float
    
    # Category-level forecasts
    category_forecasts: Dict[str, float] = field(default_factory=dict)
    
    # Regime suggestion
    suggested_regime: str = ""
    regime_rationale: str = ""


@dataclass
class WeekForecast:
    """Forecast for an upcoming week."""
    store: str
    forecast_date: date  # When this forecast was generated
    week_start: date
    week_end: date
    
    days: List[DayForecast]
    
    # Week totals
    total_expected_revenue: float
    total_expected_units: float
    avg_confidence: float
    
    # Week-over-week comparison (if available)
    last_week_revenue: Optional[float] = None
    yoy_comparison: Optional[float] = None  # vs same week last year
    
    # Key insights
    peak_day: Optional[str] = None
    peak_revenue: float = 0.0
    notable_events: List[str] = field(default_factory=list)
    weather_impact: str = ""


@dataclass 
class ForecastConfig:
    """Configuration for the forecast engine."""
    
    # How many historical neighbors to consider
    k_neighbors: int = 100
    
    # Minimum confidence threshold for forecasts
    min_confidence: float = 0.3
    
    # Weather forecast days ahead (Open-Meteo provides up to 16)
    forecast_days: int = 7
    
    # Historical lookback for building conditions dataset
    lookback_days: int = 365 * 3  # 3 years
    
    # Similarity config
    similarity_config: SimilarityConfig = field(default_factory=SimilarityConfig)
    
    # Category aggregation level
    category_level: str = "category"  # or "subcategory"
    
    # Skip category-level forecasting (faster for backtesting)
    skip_categories: bool = False


# =============================================================================
# FORECAST ENGINE
# =============================================================================

class ForecastEngine:
    """
    Main forecasting engine that combines all signals to predict demand.
    
    Usage:
        engine = ForecastEngine(store="Parksville")
        engine.load_historical_data(sales_df, weather_df)
        forecast = engine.forecast_week(start_date=date.today())
    """
    
    def __init__(
        self,
        store: str,
        config: Optional[ForecastConfig] = None,
    ):
        self.store = store
        self.config = config or ForecastConfig()
        
        # Check store status and warn if not active (only once per store)
        if store not in _warned_stores:
            status = get_store_status(store)
            if status == StoreStatus.FAILED:
                logger.warning(
                    f"⚠️  CAUTION: '{store}' is a FAILED store. "
                    "Forecasts based on this store's historical data may not be reliable. "
                    "Consider this data for cautionary analysis only."
                )
                _warned_stores.add(store)
            elif status == StoreStatus.CLOSED:
                logger.warning(f"Note: '{store}' is a closed store.")
                _warned_stores.add(store)
        
        # Get store coordinates for weather
        if store in STORE_LOCATIONS:
            self.lat, self.lon = STORE_LOCATIONS[store]
        else:
            # Default to Vancouver area
            self.lat, self.lon = 49.2827, -123.1207
            logger.warning(f"Unknown store '{store}', using Vancouver coordinates")
        
        self.weather_client = WeatherClient(latitude=self.lat, longitude=self.lon)
        
        # Data storage
        self._conditions_df: Optional[pd.DataFrame] = None
        self._sales_df: Optional[pd.DataFrame] = None
        self._category_sales: Optional[pd.DataFrame] = None
        self._fast_index: Optional[FastSimilarityIndex] = None  # Cache for fast similarity search
    
    def load_historical_data(
        self,
        sales_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
    ) -> None:
        """
        Load historical sales and weather data for forecasting.
        
        Parameters
        ----------
        sales_df : pd.DataFrame
            Historical sales with columns: date, category, quantity, revenue, etc.
        weather_df : pd.DataFrame, optional
            Historical weather. If not provided, will be fetched.
        """
        self._sales_df = sales_df.copy()
        
        # Build conditions dataframe from sales + weather
        self._build_conditions_df(weather_df)
        
        # Pre-aggregate category-level sales
        self._aggregate_category_sales()
        
        # Build fast similarity index (this is the speedup!)
        # Try GPU first, then FAISS, then fall back to slow path
        from .predictor_fast import TORCH_AVAILABLE, GPU_NAME
        
        if TORCH_AVAILABLE:
            self._fast_index = FastSimilarityIndex(
                self._conditions_df,
                self.config.similarity_config,
                use_gpu=True,
            )
            logger.info(f"Built GPU similarity index on {GPU_NAME}")
        elif FAISS_AVAILABLE:
            self._fast_index = FastSimilarityIndex(
                self._conditions_df,
                self.config.similarity_config,
                use_gpu=False,
            )
            logger.info(f"Built FAISS CPU index for similarity search")
        else:
            self._fast_index = None
            logger.warning("No GPU or FAISS available - using slow similarity search")
        
        logger.info(
            f"Loaded {len(self._conditions_df)} historical days for {self.store}"
        )
    
    def _build_conditions_df(self, weather_df: Optional[pd.DataFrame]) -> None:
        """Build the conditions dataframe used for similarity matching."""
        
        # Aggregate sales by date
        daily_sales = (
            self._sales_df
            .groupby("date")
            .agg({
                "quantity": "sum",
                "subtotal": "sum",
            })
            .reset_index()
            .rename(columns={"quantity": "sales_units", "subtotal": "sales_revenue"})
        )
        
        # Ensure date is date type
        daily_sales["date"] = pd.to_datetime(daily_sales["date"]).dt.date
        
        # Add calendar features
        daily_sales["dow"] = daily_sales["date"].apply(lambda d: d.weekday())
        daily_sales["hour"] = 12  # Default to noon for daily aggregates
        
        # NEW: Add seasonality features (critical for accuracy!)
        daily_sales["month"] = daily_sales["date"].apply(lambda d: d.month)
        daily_sales["day_of_year"] = daily_sales["date"].apply(lambda d: d.timetuple().tm_yday)
        
        # Add calendar context
        calendar = AOCCalendar()
        vibe_engine = _get_vibe_engine()
        ext_calendar = _get_calendar_service()
        
        for idx, row in daily_sales.iterrows():
            d = row["date"]
            ctx = get_calendar_context(d)
            
            daily_sales.loc[idx, "is_holiday"] = 1 if ctx.get("is_holiday") else 0
            daily_sales.loc[idx, "is_preholiday"] = 1 if ctx.get("is_preholiday") else 0
            daily_sales.loc[idx, "is_payday_window"] = 1 if ctx.get("is_payday") else 0
            daily_sales.loc[idx, "has_home_game"] = 0  # TODO: integrate sports calendar
            daily_sales.loc[idx, "has_concert"] = 0
            daily_sales.loc[idx, "has_festival"] = 1 if ctx.get("is_cannabis_event") else 0
            # Sunday-specific features
            daily_sales.loc[idx, "is_sunday"] = 1 if ctx.get("is_sunday") else 0
            daily_sales.loc[idx, "is_nfl_sunday"] = 1 if (ctx.get("is_sunday") and ctx.get("is_nfl_season")) else 0
            
            # Vibe signal features
            # Get weather data for this day if available
            temp_c = row.get("temp_c", 15.0) if "temp_c" in daily_sales.columns else 15.0
            precip_mm = row.get("precip_mm", 0.0) if "precip_mm" in daily_sales.columns else 0.0
            weather_data = {"temp_c": temp_c, "precip_mm": precip_mm}
            
            day_vibe = vibe_engine.get_day_vibe(d, weather_data=weather_data)
            daily_sales.loc[idx, "couch_index"] = day_vibe.couch_index
            daily_sales.loc[idx, "party_index"] = day_vibe.party_index
            daily_sales.loc[idx, "stress_index"] = day_vibe.stress_index
            
            # Check for major events from external calendar
            ext_data = ext_calendar.get_calendar_data(d)
            daily_sales.loc[idx, "has_major_event"] = 1 if ext_data.get("has_events") else 0
        
        # Merge weather if provided
        if weather_df is not None:
            weather_daily = (
                weather_df
                .groupby("date")
                .agg({
                    "temp_c": "mean",
                    "precip_mm": "sum",
                })
                .reset_index()
            )
            weather_daily["date"] = pd.to_datetime(weather_daily["date"]).dt.date
            daily_sales = daily_sales.merge(weather_daily, on="date", how="left")
        else:
            # Use placeholder weather
            daily_sales["temp_c"] = 15.0
            daily_sales["precip_mm"] = 0.0
        
        # Fill any missing weather with defaults
        daily_sales["temp_c"] = daily_sales["temp_c"].fillna(15.0)
        daily_sales["precip_mm"] = daily_sales["precip_mm"].fillna(0.0)
        
        # Add vibe indices (placeholder - would come from mood data)
        daily_sales["local_vibe_at_home_index"] = 0.5
        daily_sales["local_vibe_out_and_about_index"] = 0.5
        
        self._conditions_df = daily_sales
    
    def _aggregate_category_sales(self) -> None:
        """Pre-aggregate sales by date and category for category-level forecasts."""
        if self._sales_df is None:
            return
        
        cat_col = self.config.category_level
        if cat_col not in self._sales_df.columns:
            cat_col = "category"
        
        self._category_sales = (
            self._sales_df
            .groupby(["date", cat_col])
            .agg({
                "quantity": "sum",
                "subtotal": "sum",
            })
            .reset_index()
            .rename(columns={
                "quantity": "units",
                "subtotal": "revenue",
                cat_col: "category",
            })
        )
    
    def forecast_day(
        self,
        target_date: date,
        weather_forecast: Optional[Dict[str, Any]] = None,
    ) -> DayForecast:
        """
        Generate demand forecast for a single day.
        
        Parameters
        ----------
        target_date : date
            The day to forecast
        weather_forecast : dict, optional
            Weather data for the day. If not provided, will fetch from API.
        """
        if self._conditions_df is None:
            raise ValueError("Must call load_historical_data() first")
        
        # Get calendar context
        ctx = get_calendar_context(target_date)
        
        # Get weather forecast if not provided
        if weather_forecast is None:
            weather_forecast = self._get_weather_for_date(target_date)
        
        # Get vibe signals for the target date (use singletons)
        vibe_engine = _get_vibe_engine()
        ext_calendar = _get_calendar_service()
        weather_data = {
            "temp_c": weather_forecast.get("temp_avg", 15.0),
            "precip_mm": weather_forecast.get("precip_sum", 0.0),
        }
        day_vibe = vibe_engine.get_day_vibe(target_date, weather_data=weather_data)
        ext_data = ext_calendar.get_calendar_data(target_date)
        
        # Build future condition row for similarity matching
        future_row = {
            "dow": target_date.weekday(),
            "hour": 12,  # Midday for daily
            # NEW: Seasonality features (critical for accuracy!)
            "month": target_date.month,
            "day_of_year": target_date.timetuple().tm_yday,
            # Weather
            "temp_c": weather_forecast.get("temp_avg", 15.0),
            "precip_mm": weather_forecast.get("precip_sum", 0.0),
            # Calendar/events
            "is_holiday": 1 if ctx.get("is_holiday") else 0,
            "is_preholiday": 1 if ctx.get("is_preholiday") else 0,
            "is_payday_window": 1 if ctx.get("is_payday") else 0,
            "has_home_game": 0,
            "has_concert": 0,
            "has_festival": 1 if ctx.get("is_cannabis_event") else 0,
            "local_vibe_at_home_index": 0.5,
            "local_vibe_out_and_about_index": 0.5,
            # Sunday-specific features
            "is_sunday": 1 if ctx.get("is_sunday") else 0,
            "is_nfl_sunday": 1 if (ctx.get("is_sunday") and ctx.get("is_nfl_season")) else 0,
            # NEW: Vibe signal features
            "couch_index": day_vibe.couch_index,
            "party_index": day_vibe.party_index,
            "stress_index": day_vibe.stress_index,
            "has_major_event": 1 if ext_data.get("has_events") else 0,
        }
        
        # Run similarity forecast (use fast version if available)
        if self._fast_index is not None:
            forecast_result = forecast_demand_fast(
                conditions_df=self._conditions_df,
                future_row=future_row,
                k_neighbors=self.config.k_neighbors,
                outcome_cols=["sales_units", "sales_revenue"],
                cfg=self.config.similarity_config,
                index=self._fast_index,
            )
        else:
            forecast_result = forecast_demand_for_slot(
                conditions_df=self._conditions_df,
                future_row=future_row,
                k_neighbors=self.config.k_neighbors,
                outcome_cols=["sales_units", "sales_revenue"],
                cfg=self.config.similarity_config,
            )
        
        # Calculate confidence from similarity scores
        sim_range = forecast_result["similarity_max"] - forecast_result["similarity_min"]
        sim_mean = forecast_result["similarity_mean"]
        # Higher mean similarity = more confidence, narrower range = more confidence
        confidence = min(1.0, max(0.0, (sim_mean + 10) / 20))  # Normalize to 0-1
        
        # Get category forecasts (skip if configured for speed)
        if self.config.skip_categories:
            category_forecasts = {}
        else:
            category_forecasts = self._forecast_categories(target_date, future_row)
        
        # Get events from context
        events = ctx.get("events", [])
        
        # Determine suggested regime
        regime, rationale = self._suggest_regime(target_date, ctx, weather_forecast, events)
        
        dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
        
        return DayForecast(
            date=target_date,
            dow=target_date.weekday(),
            dow_name=dow_names[target_date.weekday()],
            temp_high=weather_forecast.get("temp_max", 20.0),
            temp_low=weather_forecast.get("temp_min", 10.0),
            temp_avg=weather_forecast.get("temp_avg", 15.0),
            precip_mm=weather_forecast.get("precip_sum", 0.0),
            weather_code=weather_forecast.get("weather_code", 0),
            weather_desc=weather_forecast.get("weather_desc", "Unknown"),
            is_holiday=ctx.get("is_holiday", False),
            holiday_name=ctx.get("holiday_name"),
            is_preholiday=ctx.get("is_preholiday", False),
            is_payday_window=ctx.get("is_payday", False),
            events=[str(e) for e in events],
            expected_revenue=forecast_result.get("expected_sales_revenue", 0.0),
            expected_units=forecast_result.get("expected_sales_units", 0.0),
            confidence=confidence,
            similar_days_count=int(forecast_result.get("neighbors_used", 0)),
            similarity_score=sim_mean,
            category_forecasts=category_forecasts,
            suggested_regime=regime,
            regime_rationale=rationale,
        )
    
    def _forecast_categories(
        self,
        target_date: date,
        future_row: Dict[str, Any],
    ) -> Dict[str, float]:
        """Forecast revenue by category."""
        if self._category_sales is None:
            return {}
        
        category_forecasts = {}
        
        # Get unique categories
        categories = self._category_sales["category"].unique()
        
        for cat in categories:
            # Filter conditions to this category
            cat_history = self._category_sales[
                self._category_sales["category"] == cat
            ].copy()
            
            if len(cat_history) < 10:
                continue
            
            # Build conditions with category sales
            cat_conditions = self._conditions_df.merge(
                cat_history[["date", "revenue"]],
                on="date",
                how="inner",
            )
            
            if len(cat_conditions) < 10:
                continue
            
            try:
                result = forecast_demand_for_slot(
                    conditions_df=cat_conditions,
                    future_row=future_row,
                    k_neighbors=min(50, len(cat_conditions)),
                    outcome_cols=["revenue"],
                    cfg=self.config.similarity_config,
                )
                category_forecasts[cat] = result.get("expected_revenue", 0.0)
            except Exception as e:
                logger.debug(f"Could not forecast category {cat}: {e}")
        
        return category_forecasts
    
    def _get_weather_for_date(self, target_date: date) -> Dict[str, Any]:
        """Get weather forecast for a specific date."""
        try:
            days_ahead = (target_date - date.today()).days
            if days_ahead < 0:
                # Historical - would need to fetch from archive
                return {"temp_avg": 15.0, "precip_sum": 0.0, "weather_code": 0, "weather_desc": "Unknown"}
            
            forecast = self.weather_client.get_forecast(days=max(1, days_ahead + 1))
            
            # Find the matching day
            for day in forecast.get("daily", []):
                if day.get("date") == str(target_date):
                    return {
                        "temp_max": day.get("temp_max", 20.0),
                        "temp_min": day.get("temp_min", 10.0),
                        "temp_avg": (day.get("temp_max", 20.0) + day.get("temp_min", 10.0)) / 2,
                        "precip_sum": day.get("precip_sum", 0.0),
                        "weather_code": day.get("weather_code", 0),
                        "weather_desc": day.get("weather_desc", "Unknown"),
                    }
        except Exception as e:
            logger.warning(f"Could not fetch weather for {target_date}: {e}")
        
        return {"temp_avg": 15.0, "precip_sum": 0.0, "weather_code": 0, "weather_desc": "Unknown"}
    
    def _suggest_regime(
        self,
        target_date: date,
        ctx: Dict[str, Any],
        weather: Dict[str, Any],
        events: List,
    ) -> Tuple[str, str]:
        """Suggest the appropriate sales regime for the day."""
        reasons = []
        
        # Check for special events first
        if any("420" in str(e) for e in events):
            return "420_celebration", "4/20 cannabis culture celebration day"
        
        is_holiday = ctx.get("is_holiday", False)
        holiday_name = ctx.get("holiday_name")
        is_preholiday = ctx.get("is_preholiday", False)
        
        if is_holiday:
            if holiday_name in ["Christmas Day", "Boxing Day"]:
                return "holiday_gifting", f"{holiday_name} - gifting focus"
            if holiday_name in ["Canada Day", "BC Day", "Victoria Day"]:
                return "holiday_party", f"{holiday_name} - celebration/party focus"
            return "holiday_general", f"Holiday: {holiday_name}"
        
        if is_preholiday:
            reasons.append("pre-holiday stock-up period")
        
        # Weather impact
        temp_avg = weather.get("temp_avg", 15.0)
        precip = weather.get("precip_sum", 0.0)
        
        if precip > 10:
            reasons.append("rainy day - cozy indoor products")
            return "rainy_cozy", "; ".join(reasons)
        
        if temp_avg > 25:
            reasons.append("hot day - refreshing products, beverages")
            return "hot_summer", "; ".join(reasons)
        
        if temp_avg < 5:
            reasons.append("cold day - warming, indica focus")
            return "cold_winter", "; ".join(reasons)
        
        # Default by day of week
        dow = target_date.weekday()
        if dow == 4:  # Friday
            return "friday_rush", "Friday - end of work week rush"
        if dow == 5:  # Saturday
            return "weekend_peak", "Saturday - weekend peak"
        if dow == 6:  # Sunday
            return "sunday_chill", "Sunday - relaxed shopping"
        
        return "standard", "Standard business day"
    
    def forecast_week(
        self,
        start_date: Optional[date] = None,
    ) -> WeekForecast:
        """
        Generate demand forecast for an entire week.
        
        Parameters
        ----------
        start_date : date, optional
            First day of the week to forecast. Defaults to tomorrow.
        """
        if start_date is None:
            start_date = date.today() + timedelta(days=1)
        
        week_end = start_date + timedelta(days=6)
        
        # Fetch weather for the whole week at once
        try:
            weather_data = self.weather_client.get_forecast(days=14)
            weather_by_date = {}
            for day in weather_data.get("daily", []):
                d = day.get("date")
                weather_by_date[d] = {
                    "temp_max": day.get("temp_max", 20.0),
                    "temp_min": day.get("temp_min", 10.0),
                    "temp_avg": (day.get("temp_max", 20.0) + day.get("temp_min", 10.0)) / 2,
                    "precip_sum": day.get("precip_sum", 0.0),
                    "weather_code": day.get("weather_code", 0),
                    "weather_desc": day.get("weather_desc", "Unknown"),
                }
        except Exception as e:
            logger.warning(f"Could not fetch week weather: {e}")
            weather_by_date = {}
        
        # Forecast each day
        days = []
        total_revenue = 0.0
        total_units = 0.0
        total_confidence = 0.0
        notable_events = []
        peak_day = None
        peak_revenue = 0.0
        
        for i in range(7):
            day_date = start_date + timedelta(days=i)
            weather = weather_by_date.get(str(day_date), {})
            
            day_forecast = self.forecast_day(day_date, weather)
            days.append(day_forecast)
            
            total_revenue += day_forecast.expected_revenue
            total_units += day_forecast.expected_units
            total_confidence += day_forecast.confidence
            
            if day_forecast.expected_revenue > peak_revenue:
                peak_revenue = day_forecast.expected_revenue
                peak_day = day_forecast.dow_name
            
            if day_forecast.is_holiday or day_forecast.events:
                notable_events.extend(day_forecast.events)
                if day_forecast.holiday_name:
                    notable_events.append(day_forecast.holiday_name)
        
        # Weather summary
        avg_temp = np.mean([d.temp_avg for d in days])
        total_precip = sum(d.precip_mm for d in days)
        if total_precip > 50:
            weather_impact = "Very rainy week expected - indoor/cozy product focus"
        elif total_precip > 20:
            weather_impact = "Some rain expected - mixed indoor/outdoor"
        elif avg_temp > 25:
            weather_impact = "Hot week - refreshing products, beverages will perform well"
        elif avg_temp < 5:
            weather_impact = "Cold week - warming products, indica focus"
        else:
            weather_impact = "Moderate weather - standard product mix"
        
        return WeekForecast(
            store=self.store,
            forecast_date=date.today(),
            week_start=start_date,
            week_end=week_end,
            days=days,
            total_expected_revenue=total_revenue,
            total_expected_units=total_units,
            avg_confidence=total_confidence / 7,
            peak_day=peak_day,
            peak_revenue=peak_revenue,
            notable_events=list(set(notable_events)),
            weather_impact=weather_impact,
        )
    
    def to_dataframe(self, forecast: WeekForecast) -> pd.DataFrame:
        """Convert a week forecast to a DataFrame for easy viewing."""
        rows = []
        for day in forecast.days:
            rows.append({
                "date": day.date,
                "day": day.dow_name,
                "temp_high": day.temp_high,
                "temp_low": day.temp_low,
                "precip_mm": day.precip_mm,
                "weather": day.weather_desc,
                "holiday": day.holiday_name or "",
                "events": ", ".join(day.events),
                "expected_revenue": round(day.expected_revenue, 2),
                "expected_units": round(day.expected_units, 0),
                "confidence": round(day.confidence, 2),
                "regime": day.suggested_regime,
            })
        return pd.DataFrame(rows)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def quick_forecast(
    store: str,
    sales_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    days: int = 7,
) -> WeekForecast:
    """
    Quick forecast for a store without manual engine setup.
    
    Parameters
    ----------
    store : str
        Store name (e.g., "Parksville")
    sales_df : pd.DataFrame
        Historical sales data
    weather_df : pd.DataFrame, optional
        Historical weather data
    days : int
        Number of days to forecast (default 7)
    """
    engine = ForecastEngine(store=store)
    engine.load_historical_data(sales_df, weather_df)
    return engine.forecast_week()
