"""
AOC Backtesting Framework

Walk-forward validation for forecast accuracy and purchase recommendation tuning.

The backtester:
1. Splits historical data into train/test windows
2. Generates forecasts using only past data (no lookahead)
3. Compares predictions vs actual outcomes
4. Computes accuracy metrics at day, week, and category levels
5. Tunes recommendation parameters based on error patterns

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

from .forecast_engine import ForecastEngine, ForecastConfig, DayForecast, WeekForecast
from .store_profiles import get_store_status, StoreStatus, is_store_active
from .signals.vibe_signals import preload_vibe_cache

logger = logging.getLogger(__name__)


# =============================================================================
# METRICS
# =============================================================================

@dataclass
class ForecastMetrics:
    """Metrics for evaluating forecast accuracy."""
    
    # Count
    n_days: int
    n_weeks: int
    
    # Revenue prediction accuracy
    revenue_mae: float  # Mean Absolute Error
    revenue_mape: float  # Mean Absolute Percentage Error
    revenue_rmse: float  # Root Mean Square Error
    revenue_bias: float  # Average over/under prediction
    
    # Units prediction accuracy
    units_mae: float
    units_mape: float
    units_rmse: float
    units_bias: float
    
    # Day-of-week accuracy (some days might be harder to predict)
    dow_mape: Dict[int, float] = field(default_factory=dict)
    
    # Category-level accuracy
    category_mape: Dict[str, float] = field(default_factory=dict)
    
    # Confidence calibration (are high-confidence predictions more accurate?)
    confidence_correlation: float = 0.0
    
    # Over/under prediction patterns
    overpredict_rate: float = 0.0  # % of days we overpredict
    severe_miss_rate: float = 0.0  # % of days with >50% error


@dataclass
class BacktestResult:
    """Results from a backtest run."""
    
    store: str
    backtest_start: date
    backtest_end: date
    train_window_days: int
    test_window_days: int
    
    # Overall metrics
    metrics: ForecastMetrics
    
    # Per-week results
    weekly_results: List[Dict[str, Any]]
    
    # Raw predictions vs actuals
    daily_results: pd.DataFrame
    
    # Identified patterns
    worst_days: List[Tuple[date, float, str]]  # date, error%, reason
    best_days: List[Tuple[date, float]]
    
    # Recommendations for tuning
    tuning_recommendations: List[str]


# =============================================================================
# BACKTESTER
# =============================================================================

class BacktestRunner:
    """
    Walk-forward backtesting for forecast validation.
    
    Uses expanding or rolling windows to ensure no lookahead bias.
    
    Usage:
        runner = BacktestRunner(
            store="Parksville",
            train_window_days=365,  # Use 1 year of history
            test_window_days=7,     # Forecast 1 week at a time
        )
        result = runner.run(sales_df, weather_df, start_date, end_date)
    """
    
    def __init__(
        self,
        store: str,
        train_window_days: int = 365,
        test_window_days: int = 7,
        rolling: bool = False,  # If True, use rolling window. If False, expanding.
        config: Optional[ForecastConfig] = None,
    ):
        self.store = store
        self.train_window_days = train_window_days
        self.test_window_days = test_window_days
        self.rolling = rolling
        
        # Check store status and warn if not active
        status = get_store_status(store)
        if status == StoreStatus.FAILED:
            logger.warning(
                f"⚠️  CAUTION: '{store}' is a FAILED store. "
                "Results may not be representative of healthy store patterns. "
                "Consider using this data only for cautionary analysis."
            )
        elif status == StoreStatus.CLOSED:
            logger.warning(f"Note: '{store}' is a closed store.")
        elif status is None:
            logger.debug(f"Store '{store}' not found in registry.")
        
        # Use provided config or create default with skip_categories=True for speed
        if config:
            self.config = config
        else:
            self.config = ForecastConfig(skip_categories=True)
    
    def run(
        self,
        sales_df: pd.DataFrame,
        weather_df: Optional[pd.DataFrame] = None,
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
    ) -> BacktestResult:
        """
        Run walk-forward backtest.
        
        Parameters
        ----------
        sales_df : pd.DataFrame
            Full historical sales data
        weather_df : pd.DataFrame, optional
            Historical weather data
        start_date : date, optional
            Start of test period. Defaults to train_window_days after data start.
        end_date : date, optional
            End of test period. Defaults to most recent data.
        """
        # Ensure date column is proper date type
        sales_df = sales_df.copy()
        sales_df["date"] = pd.to_datetime(sales_df["date"]).dt.date
        
        # Determine date range
        data_start = sales_df["date"].min()
        data_end = sales_df["date"].max()
        
        if start_date is None:
            start_date = data_start + timedelta(days=self.train_window_days)
        
        if end_date is None:
            end_date = data_end
        
        logger.info(
            f"Running backtest for {self.store}: {start_date} to {end_date} "
            f"(train={self.train_window_days}d, test={self.test_window_days}d)"
        )
        
        # Preload vibe cache for the entire date range (prevents repeated API calls)
        preload_vibe_cache(data_start, end_date)
        
        # Collect predictions and actuals
        daily_results = []
        weekly_results = []
        
        current_date = start_date
        
        while current_date <= end_date - timedelta(days=self.test_window_days - 1):
            # Define train period
            if self.rolling:
                train_start = current_date - timedelta(days=self.train_window_days)
            else:
                train_start = data_start
            train_end = current_date - timedelta(days=1)
            
            # Filter training data
            train_sales = sales_df[
                (sales_df["date"] >= train_start) & 
                (sales_df["date"] <= train_end)
            ]
            
            train_weather = None
            if weather_df is not None and len(weather_df) > 0:
                weather_df_copy = weather_df.copy()
                weather_df_copy["date"] = pd.to_datetime(weather_df_copy["date"]).dt.date
                train_weather = weather_df_copy[
                    (weather_df_copy["date"] >= train_start) &
                    (weather_df_copy["date"] <= train_end)
                ]
            
            if len(train_sales) < 30:
                logger.warning(f"Insufficient training data at {current_date}, skipping")
                current_date += timedelta(days=self.test_window_days)
                continue
            
            # Build forecast engine with training data only
            engine = ForecastEngine(store=self.store, config=self.config)
            engine.load_historical_data(train_sales, train_weather)
            
            # Generate forecast for test period
            test_end = current_date + timedelta(days=self.test_window_days - 1)
            
            week_predicted_revenue = 0.0
            week_actual_revenue = 0.0
            week_predicted_units = 0.0
            week_actual_units = 0.0
            
            for i in range(self.test_window_days):
                forecast_date = current_date + timedelta(days=i)
                
                if forecast_date > data_end:
                    break
                
                # Get actual weather for this date (simulate having forecast)
                actual_weather = {}
                if weather_df is not None:
                    day_weather = weather_df_copy[weather_df_copy["date"] == forecast_date]
                    if not day_weather.empty:
                        row = day_weather.iloc[0]
                        actual_weather = {
                            "temp_max": row.get("temp_max", 20.0),
                            "temp_min": row.get("temp_min", 10.0),
                            "temp_avg": row.get("temp_c", 15.0),
                            "precip_sum": row.get("precip_mm", 0.0),
                            "weather_code": row.get("weather_code", 0),
                            "weather_desc": row.get("weather_desc", "Unknown"),
                        }
                
                # Generate prediction
                try:
                    day_forecast = engine.forecast_day(forecast_date, actual_weather)
                except Exception as e:
                    logger.warning(f"Forecast failed for {forecast_date}: {e}")
                    continue
                
                # Get actual sales for this date
                actual_sales = sales_df[sales_df["date"] == forecast_date]
                if actual_sales.empty:
                    continue
                
                actual_revenue = actual_sales["subtotal"].sum()
                actual_units = actual_sales["quantity"].sum()
                
                # Record result
                daily_results.append({
                    "date": forecast_date,
                    "dow": forecast_date.weekday(),
                    "predicted_revenue": day_forecast.expected_revenue,
                    "actual_revenue": actual_revenue,
                    "predicted_units": day_forecast.expected_units,
                    "actual_units": actual_units,
                    "confidence": day_forecast.confidence,
                    "is_holiday": day_forecast.is_holiday,
                    "is_payday": day_forecast.is_payday_window,
                    "suggested_regime": day_forecast.suggested_regime,
                    "revenue_error": day_forecast.expected_revenue - actual_revenue,
                    "revenue_error_pct": (
                        (day_forecast.expected_revenue - actual_revenue) / actual_revenue * 100
                        if actual_revenue > 0 else 0
                    ),
                    "units_error": day_forecast.expected_units - actual_units,
                    "units_error_pct": (
                        (day_forecast.expected_units - actual_units) / actual_units * 100
                        if actual_units > 0 else 0
                    ),
                })
                
                week_predicted_revenue += day_forecast.expected_revenue
                week_actual_revenue += actual_revenue
                week_predicted_units += day_forecast.expected_units
                week_actual_units += actual_units
            
            # Record weekly result
            if week_actual_revenue > 0:
                weekly_results.append({
                    "week_start": current_date,
                    "predicted_revenue": week_predicted_revenue,
                    "actual_revenue": week_actual_revenue,
                    "predicted_units": week_predicted_units,
                    "actual_units": week_actual_units,
                    "revenue_error_pct": (
                        (week_predicted_revenue - week_actual_revenue) / week_actual_revenue * 100
                    ),
                })
            
            # Move to next test window
            current_date += timedelta(days=self.test_window_days)
        
        # Compute metrics
        daily_df = pd.DataFrame(daily_results)
        metrics = self._compute_metrics(daily_df)
        
        # Identify worst/best days
        worst_days = self._find_worst_days(daily_df)
        best_days = self._find_best_days(daily_df)
        
        # Generate tuning recommendations
        recommendations = self._generate_recommendations(daily_df, metrics)
        
        return BacktestResult(
            store=self.store,
            backtest_start=start_date,
            backtest_end=end_date,
            train_window_days=self.train_window_days,
            test_window_days=self.test_window_days,
            metrics=metrics,
            weekly_results=weekly_results,
            daily_results=daily_df,
            worst_days=worst_days,
            best_days=best_days,
            tuning_recommendations=recommendations,
        )
    
    def _compute_metrics(self, daily_df: pd.DataFrame) -> ForecastMetrics:
        """Compute accuracy metrics from daily results."""
        if daily_df.empty:
            return ForecastMetrics(
                n_days=0, n_weeks=0,
                revenue_mae=0, revenue_mape=0, revenue_rmse=0, revenue_bias=0,
                units_mae=0, units_mape=0, units_rmse=0, units_bias=0,
            )
        
        n_days = len(daily_df)
        n_weeks = n_days // 7
        
        # Revenue metrics
        revenue_errors = daily_df["revenue_error"].values
        revenue_pct_errors = daily_df["revenue_error_pct"].abs().values
        
        revenue_mae = np.abs(revenue_errors).mean()
        revenue_mape = revenue_pct_errors.mean()
        revenue_rmse = np.sqrt((revenue_errors ** 2).mean())
        revenue_bias = revenue_errors.mean()
        
        # Units metrics
        units_errors = daily_df["units_error"].values
        units_pct_errors = daily_df["units_error_pct"].abs().values
        
        units_mae = np.abs(units_errors).mean()
        units_mape = units_pct_errors.mean()
        units_rmse = np.sqrt((units_errors ** 2).mean())
        units_bias = units_errors.mean()
        
        # Day-of-week breakdown
        dow_mape = {}
        for dow in range(7):
            dow_data = daily_df[daily_df["dow"] == dow]
            if len(dow_data) > 0:
                dow_mape[dow] = dow_data["revenue_error_pct"].abs().mean()
        
        # Confidence correlation
        confidence_correlation = 0.0
        if len(daily_df) > 10:
            abs_errors = daily_df["revenue_error_pct"].abs()
            confidence = daily_df["confidence"]
            if abs_errors.std() > 0 and confidence.std() > 0:
                confidence_correlation = -np.corrcoef(abs_errors, confidence)[0, 1]
        
        # Over/under prediction patterns
        overpredict_rate = (daily_df["revenue_error"] > 0).mean()
        severe_miss_rate = (daily_df["revenue_error_pct"].abs() > 50).mean()
        
        return ForecastMetrics(
            n_days=n_days,
            n_weeks=n_weeks,
            revenue_mae=float(revenue_mae),
            revenue_mape=float(revenue_mape),
            revenue_rmse=float(revenue_rmse),
            revenue_bias=float(revenue_bias),
            units_mae=float(units_mae),
            units_mape=float(units_mape),
            units_rmse=float(units_rmse),
            units_bias=float(units_bias),
            dow_mape=dow_mape,
            confidence_correlation=float(confidence_correlation),
            overpredict_rate=float(overpredict_rate),
            severe_miss_rate=float(severe_miss_rate),
        )
    
    def _find_worst_days(
        self,
        daily_df: pd.DataFrame,
        n: int = 10,
    ) -> List[Tuple[date, float, str]]:
        """Find the days with worst prediction errors."""
        if daily_df.empty:
            return []
        
        df = daily_df.copy()
        df["abs_error_pct"] = df["revenue_error_pct"].abs()
        worst = df.nlargest(n, "abs_error_pct")
        
        results = []
        for _, row in worst.iterrows():
            reason = []
            if row.get("is_holiday"):
                reason.append("holiday")
            if row.get("is_payday"):
                reason.append("payday")
            reason.append(f"regime={row.get('suggested_regime', 'unknown')}")
            
            results.append((
                row["date"],
                row["abs_error_pct"],
                "; ".join(reason),
            ))
        
        return results
    
    def _find_best_days(
        self,
        daily_df: pd.DataFrame,
        n: int = 10,
    ) -> List[Tuple[date, float]]:
        """Find the days with best prediction accuracy."""
        if daily_df.empty:
            return []
        
        df = daily_df.copy()
        df["abs_error_pct"] = df["revenue_error_pct"].abs()
        best = df.nsmallest(n, "abs_error_pct")
        
        return [(row["date"], row["abs_error_pct"]) for _, row in best.iterrows()]
    
    def _generate_recommendations(
        self,
        daily_df: pd.DataFrame,
        metrics: ForecastMetrics,
    ) -> List[str]:
        """Generate recommendations for improving forecast accuracy."""
        recommendations = []
        
        if metrics.n_days == 0:
            return ["Insufficient data for analysis"]
        
        # Overall accuracy assessment
        if metrics.revenue_mape > 30:
            recommendations.append(
                f"High overall error ({metrics.revenue_mape:.1f}% MAPE). "
                "Consider: more training data, better weather integration, "
                "or different similarity weights."
            )
        elif metrics.revenue_mape > 20:
            recommendations.append(
                f"Moderate error ({metrics.revenue_mape:.1f}% MAPE). "
                "Room for improvement with tuning."
            )
        else:
            recommendations.append(
                f"Good accuracy ({metrics.revenue_mape:.1f}% MAPE). "
                "Focus on edge cases."
            )
        
        # Bias detection
        if metrics.revenue_bias > 0:
            pct_over = (metrics.revenue_bias / daily_df["actual_revenue"].mean()) * 100
            recommendations.append(
                f"Systematic overprediction by ~{pct_over:.1f}%. "
                "Consider reducing base forecast or adjusting similarity weights."
            )
        elif metrics.revenue_bias < 0:
            pct_under = (-metrics.revenue_bias / daily_df["actual_revenue"].mean()) * 100
            recommendations.append(
                f"Systematic underprediction by ~{pct_under:.1f}%. "
                "Consider increasing base forecast."
            )
        
        # Day-of-week analysis
        if metrics.dow_mape:
            worst_dow = max(metrics.dow_mape.items(), key=lambda x: x[1])
            best_dow = min(metrics.dow_mape.items(), key=lambda x: x[1])
            dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            
            if worst_dow[1] > best_dow[1] * 1.5:
                recommendations.append(
                    f"{dow_names[worst_dow[0]]} is hardest to predict "
                    f"({worst_dow[1]:.1f}% MAPE vs {best_dow[1]:.1f}% on {dow_names[best_dow[0]]}). "
                    "Consider day-specific adjustments."
                )
        
        # Confidence calibration
        if metrics.confidence_correlation < 0.1:
            recommendations.append(
                "Confidence scores not well calibrated with accuracy. "
                "Consider recalibrating confidence calculation."
            )
        elif metrics.confidence_correlation > 0.3:
            recommendations.append(
                f"Good confidence calibration (r={metrics.confidence_correlation:.2f}). "
                "High-confidence forecasts are more reliable."
            )
        
        # Severe miss analysis
        if metrics.severe_miss_rate > 0.1:
            recommendations.append(
                f"{metrics.severe_miss_rate*100:.1f}% of days have >50% error. "
                "Focus on identifying these edge cases (holidays, events, weather extremes)."
            )
        
        return recommendations


# =============================================================================
# PURCHASE OPTIMIZER
# =============================================================================

@dataclass
class PurchaseSuggestion:
    """Suggested purchase quantity for a product/category."""
    category: str
    product_sku: Optional[str]
    product_name: Optional[str]
    
    # Current inventory
    current_stock: int
    
    # Forecast demand
    forecast_demand_units: float
    forecast_period_days: int
    
    # Suggested order
    suggested_order_qty: int
    safety_stock: int
    
    # Rationale
    confidence: float
    rationale: str
    
    # Risk assessment
    stockout_risk: str  # low, medium, high
    overstock_risk: str


class PurchaseOptimizer:
    """
    Converts demand forecasts into purchase recommendations.
    
    Considers:
    - Forecast demand + uncertainty
    - Current inventory levels
    - Lead times
    - Minimum order quantities
    - Safety stock policies
    - Shelf life constraints
    """
    
    def __init__(
        self,
        lead_time_days: int = 3,
        safety_stock_days: float = 2.0,
        service_level: float = 0.95,  # Target in-stock rate
    ):
        self.lead_time_days = lead_time_days
        self.safety_stock_days = safety_stock_days
        self.service_level = service_level
    
    def generate_suggestions(
        self,
        forecast: WeekForecast,
        current_inventory: Dict[str, int],
        category_velocity: Optional[Dict[str, float]] = None,
    ) -> List[PurchaseSuggestion]:
        """
        Generate purchase suggestions from a week forecast.
        
        Parameters
        ----------
        forecast : WeekForecast
            Demand forecast for the upcoming week
        current_inventory : dict
            Current stock levels by category/SKU
        category_velocity : dict, optional
            Historical daily sales velocity by category
        """
        suggestions = []
        
        # Aggregate forecast demand by category
        category_demand = {}
        for day in forecast.days:
            for cat, revenue in day.category_forecasts.items():
                if cat not in category_demand:
                    category_demand[cat] = 0.0
                category_demand[cat] += revenue
        
        # Convert revenue to units (rough approximation)
        # In practice, you'd use category-specific price data
        avg_price_per_unit = 25.0  # Placeholder
        
        for category, total_revenue in category_demand.items():
            forecast_units = total_revenue / avg_price_per_unit
            current_stock = current_inventory.get(category, 0)
            
            # Calculate safety stock
            daily_demand = forecast_units / 7
            safety_stock = int(np.ceil(daily_demand * self.safety_stock_days))
            
            # Calculate order quantity
            # Need: forecast_demand + safety_stock - current_stock
            # But also account for lead time demand
            lead_time_demand = daily_demand * self.lead_time_days
            
            reorder_point = lead_time_demand + safety_stock
            target_stock = forecast_units + safety_stock
            
            order_qty = max(0, int(np.ceil(target_stock - current_stock)))
            
            # Assess risks
            days_of_stock = current_stock / daily_demand if daily_demand > 0 else float("inf")
            
            if days_of_stock < self.lead_time_days:
                stockout_risk = "high"
            elif days_of_stock < self.lead_time_days + self.safety_stock_days:
                stockout_risk = "medium"
            else:
                stockout_risk = "low"
            
            weeks_of_stock_after_order = (current_stock + order_qty) / (daily_demand * 7) if daily_demand > 0 else float("inf")
            if weeks_of_stock_after_order > 4:
                overstock_risk = "high"
            elif weeks_of_stock_after_order > 2:
                overstock_risk = "medium"
            else:
                overstock_risk = "low"
            
            # Build rationale
            rationale_parts = [
                f"Forecast: {forecast_units:.0f} units over 7 days",
                f"Current stock: {current_stock}",
                f"Days of stock: {days_of_stock:.1f}",
            ]
            if stockout_risk == "high":
                rationale_parts.append("⚠️ High stockout risk - order urgently")
            
            suggestions.append(PurchaseSuggestion(
                category=category,
                product_sku=None,
                product_name=None,
                current_stock=current_stock,
                forecast_demand_units=forecast_units,
                forecast_period_days=7,
                suggested_order_qty=order_qty,
                safety_stock=safety_stock,
                confidence=forecast.avg_confidence,
                rationale="; ".join(rationale_parts),
                stockout_risk=stockout_risk,
                overstock_risk=overstock_risk,
            ))
        
        # Sort by stockout risk
        risk_order = {"high": 0, "medium": 1, "low": 2}
        suggestions.sort(key=lambda s: (risk_order[s.stockout_risk], -s.forecast_demand_units))
        
        return suggestions
    
    def to_dataframe(self, suggestions: List[PurchaseSuggestion]) -> pd.DataFrame:
        """Convert suggestions to a DataFrame."""
        rows = []
        for s in suggestions:
            rows.append({
                "category": s.category,
                "current_stock": s.current_stock,
                "forecast_demand": round(s.forecast_demand_units, 0),
                "suggested_order": s.suggested_order_qty,
                "safety_stock": s.safety_stock,
                "stockout_risk": s.stockout_risk,
                "overstock_risk": s.overstock_risk,
                "confidence": round(s.confidence, 2),
            })
        return pd.DataFrame(rows)


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def run_backtest(
    store: str,
    sales_df: pd.DataFrame,
    weather_df: Optional[pd.DataFrame] = None,
    train_days: int = 365,
    test_days: int = 7,
) -> BacktestResult:
    """
    Convenience function to run a backtest.
    
    Parameters
    ----------
    store : str
        Store name
    sales_df : pd.DataFrame
        Historical sales data
    weather_df : pd.DataFrame, optional
        Historical weather data
    train_days : int
        Training window size (default 365)
    test_days : int
        Test window size (default 7)
    """
    runner = BacktestRunner(
        store=store,
        train_window_days=train_days,
        test_window_days=test_days,
    )
    return runner.run(sales_df, weather_df)


def print_backtest_report(result: BacktestResult) -> None:
    """Print a human-readable backtest report."""
    m = result.metrics
    
    # Check if this is a failed/closed store and show warning banner
    status = get_store_status(result.store)
    if status == StoreStatus.FAILED:
        print(f"\n{'!'*60}")
        print(f"⚠️  WARNING: {result.store} is a FAILED STORE")
        print(f"This data represents a business that failed.")
        print(f"Use for cautionary analysis only - not for forecasting.")
        print(f"{'!'*60}")
    elif status == StoreStatus.CLOSED:
        print(f"\n{'~'*60}")
        print(f"Note: {result.store} is a CLOSED store.")
        print(f"{'~'*60}")
    
    print(f"\n{'='*60}")
    print(f"BACKTEST REPORT: {result.store}")
    print(f"{'='*60}")
    print(f"Period: {result.backtest_start} to {result.backtest_end}")
    print(f"Training window: {result.train_window_days} days")
    print(f"Test window: {result.test_window_days} days")
    print(f"Days tested: {m.n_days}")
    print()
    
    print("ACCURACY METRICS")
    print("-" * 40)
    print(f"Revenue MAPE: {m.revenue_mape:.1f}%")
    print(f"Revenue MAE:  ${m.revenue_mae:,.0f}")
    print(f"Revenue Bias: ${m.revenue_bias:+,.0f}")
    print(f"Units MAPE:   {m.units_mape:.1f}%")
    print()
    
    print("DAY-OF-WEEK ACCURACY")
    print("-" * 40)
    dow_names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    for dow, mape in sorted(m.dow_mape.items()):
        bar = "█" * int(mape / 5)
        print(f"{dow_names[dow]}: {mape:5.1f}% {bar}")
    print()
    
    print("PATTERNS")
    print("-" * 40)
    print(f"Overprediction rate: {m.overpredict_rate*100:.1f}%")
    print(f"Severe miss rate (>50% error): {m.severe_miss_rate*100:.1f}%")
    print(f"Confidence calibration: {m.confidence_correlation:.2f}")
    print()
    
    print("WORST PREDICTIONS")
    print("-" * 40)
    for dt, error, reason in result.worst_days[:5]:
        print(f"{dt}: {error:+.1f}% ({reason})")
    print()
    
    print("RECOMMENDATIONS")
    print("-" * 40)
    for rec in result.tuning_recommendations:
        print(f"• {rec}")
    print()
