"""
Decomposed Demand Forecasting

Splits demand into two components:
1. REGULAR BASELINE: Predictable demand from repeat customers
   - Not weather-sensitive
   - Highly consistent day-to-day
   - Can be forecast with simple moving averages
   
2. WALK-IN DEMAND: Weather/signal-sensitive random traffic
   - Responds to weather, events, paydays
   - Uses the full similarity-matching engine
   - Higher variance, harder to predict

This decomposition improves forecast accuracy by:
- Not applying weather adjustments to regular demand
- Focusing signal-matching on genuinely responsive demand
- Providing separate confidence intervals for each component
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd


@dataclass
class RegularDemandEstimate:
    """Estimated regular customer demand for a SKU or category."""
    
    sku: Optional[str]
    category: Optional[str]
    
    # Daily baseline from regulars
    daily_regular_qty: float
    daily_regular_revenue: float
    
    # Confidence based on pattern consistency
    confidence: float  # 0-1
    
    # Pattern details
    same_day_repeat_rate: float
    consistency_score: float  # 1 - CV of daily purchases
    
    # Day-of-week adjustments (regulars may still have weekly patterns)
    dow_factors: Dict[int, float] = field(default_factory=dict)


@dataclass
class DecomposedForecast:
    """Forecast with regular vs walk-in decomposition."""
    
    date: str
    sku: Optional[str] = None
    category: Optional[str] = None
    location: Optional[str] = None
    
    # Component forecasts
    regular_qty: float = 0.0
    walkin_qty: float = 0.0
    total_qty: float = 0.0
    
    # Confidence intervals
    regular_qty_low: float = 0.0
    regular_qty_high: float = 0.0
    walkin_qty_low: float = 0.0
    walkin_qty_high: float = 0.0
    
    # Overall forecast range
    total_qty_low: float = 0.0
    total_qty_high: float = 0.0
    
    # Attribution
    regular_share: float = 0.0  # % of forecast from regulars
    
    # Drivers (for walk-in component)
    walkin_drivers: List[str] = field(default_factory=list)
    
    def to_dict(self) -> dict:
        return {
            "date": self.date,
            "sku": self.sku,
            "category": self.category,
            "location": self.location,
            "regular_qty": round(self.regular_qty, 1),
            "walkin_qty": round(self.walkin_qty, 1),
            "total_qty": round(self.total_qty, 1),
            "forecast_range": [round(self.total_qty_low, 1), round(self.total_qty_high, 1)],
            "regular_share": round(self.regular_share, 2),
            "walkin_drivers": self.walkin_drivers,
        }


class DemandDecomposer:
    """
    Decompose historical demand into regular vs walk-in components.
    
    Uses outlier detection patterns to identify which portion of demand
    is driven by predictable repeat customers vs responsive walk-ins.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            candidates = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent / "aoc_sales.db",
                Path.home() / "Projects" / "aoc-analytics" / "aoc_sales.db",
            ]
            for p in candidates:
                if p.exists():
                    db_path = str(p)
                    break
            else:
                raise FileNotFoundError("Could not find aoc_sales.db")
        self.db_path = str(db_path)
        
        # Cache regular demand estimates
        self._regular_cache: Dict[str, RegularDemandEstimate] = {}
        self._cache_time: Optional[datetime] = None
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def estimate_regular_demand(self, sku: str = None, category: str = None, 
                                 location: str = None, lookback_days: int = 90) -> RegularDemandEstimate:
        """
        Estimate the regular customer component of demand for a SKU or category.
        
        Uses same-day repeat patterns as a proxy for regular customers.
        """
        
        cache_key = f"{sku}:{category}:{location}"
        
        # Check cache
        now = datetime.now()
        if (self._cache_time and (now - self._cache_time) < timedelta(hours=1) and
            cache_key in self._regular_cache):
            return self._regular_cache[cache_key]
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Build WHERE clause
        conditions = ["date >= date('now', ?)"]
        params = [f'-{lookback_days} days']
        
        if sku:
            conditions.append("sku = ?")
            params.append(sku)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if location:
            conditions.append("location = ?")
            params.append(location)
        
        where_clause = " AND ".join(conditions)
        
        # Get daily transaction patterns
        cursor.execute(f"""
            WITH daily_stats AS (
                SELECT 
                    date,
                    CAST(strftime('%w', date) AS INTEGER) as dow,
                    COUNT(*) as transactions,
                    SUM(quantity) as qty,
                    SUM(subtotal) as revenue
                FROM sales
                WHERE {where_clause}
                GROUP BY date
            ),
            -- Identify days with same-day repeats (proxy for regular activity)
            repeat_days AS (
                SELECT 
                    date,
                    COUNT(*) as txns_that_day,
                    SUM(quantity) as qty_that_day
                FROM sales
                WHERE {where_clause}
                GROUP BY date
                HAVING txns_that_day > 1
            )
            SELECT 
                ds.date,
                ds.dow,
                ds.transactions,
                ds.qty,
                ds.revenue,
                COALESCE(rd.qty_that_day, 0) as repeat_qty
            FROM daily_stats ds
            LEFT JOIN repeat_days rd ON ds.date = rd.date
            ORDER BY ds.date
        """, params * 2)  # Params used twice in subqueries
        
        rows = cursor.fetchall()
        conn.close()
        
        if not rows:
            return RegularDemandEstimate(
                sku=sku, category=category,
                daily_regular_qty=0, daily_regular_revenue=0,
                confidence=0, same_day_repeat_rate=0, consistency_score=0
            )
        
        # Analyze patterns
        total_qty = sum(r["qty"] for r in rows)
        total_revenue = sum(r["revenue"] for r in rows)
        total_repeat_qty = sum(r["repeat_qty"] for r in rows)
        num_days = len(rows)
        
        # Same-day repeat rate
        repeat_rate = total_repeat_qty / total_qty if total_qty > 0 else 0
        
        # Consistency score (inverse of CV)
        daily_qtys = [r["qty"] for r in rows]
        cv = np.std(daily_qtys) / np.mean(daily_qtys) if np.mean(daily_qtys) > 0 else 1
        consistency = max(0, 1 - cv)
        
        # Estimate regular demand as the portion with same-day repeats
        # This is conservative - actual regular demand is likely higher
        daily_regular = total_repeat_qty / num_days
        daily_regular_rev = (total_revenue / total_qty * total_repeat_qty / num_days) if total_qty > 0 else 0
        
        # Confidence based on data volume and pattern consistency
        confidence = min(1.0, (num_days / 60) * (1 - cv / 2))
        
        # Day-of-week factors for regulars
        dow_totals = defaultdict(float)
        dow_counts = defaultdict(int)
        for r in rows:
            dow_totals[r["dow"]] += r["repeat_qty"]
            dow_counts[r["dow"]] += 1
        
        avg_daily = total_repeat_qty / num_days if num_days > 0 else 1
        dow_factors = {}
        for dow in range(7):
            if dow_counts[dow] > 0:
                dow_avg = dow_totals[dow] / dow_counts[dow]
                dow_factors[dow] = dow_avg / avg_daily if avg_daily > 0 else 1.0
            else:
                dow_factors[dow] = 1.0
        
        estimate = RegularDemandEstimate(
            sku=sku,
            category=category,
            daily_regular_qty=daily_regular,
            daily_regular_revenue=daily_regular_rev,
            confidence=confidence,
            same_day_repeat_rate=repeat_rate,
            consistency_score=consistency,
            dow_factors=dow_factors,
        )
        
        self._regular_cache[cache_key] = estimate
        self._cache_time = now
        
        return estimate
    
    def decompose_historical(self, sku: str = None, category: str = None,
                             location: str = None, days: int = 90) -> pd.DataFrame:
        """
        Decompose historical daily demand into regular vs walk-in.
        
        Returns DataFrame with columns:
        - date
        - total_qty
        - regular_qty (estimated)
        - walkin_qty (total - regular)
        """
        
        regular_est = self.estimate_regular_demand(sku, category, location, days)
        
        conn = self.get_connection()
        
        # Build query
        conditions = ["date >= date('now', ?)"]
        params = [f'-{days} days']
        
        if sku:
            conditions.append("sku = ?")
            params.append(sku)
        if category:
            conditions.append("category = ?")
            params.append(category)
        if location:
            conditions.append("location = ?")
            params.append(location)
        
        where_clause = " AND ".join(conditions)
        
        query = f"""
            SELECT 
                date,
                CAST(strftime('%w', date) AS INTEGER) as dow,
                SUM(quantity) as total_qty,
                SUM(subtotal) as total_revenue
            FROM sales
            WHERE {where_clause}
            GROUP BY date
            ORDER BY date
        """
        
        df = pd.read_sql_query(query, conn, params=params)
        conn.close()
        
        if df.empty:
            return df
        
        # Apply decomposition
        def calc_regular(row):
            dow = int(row['dow'])
            factor = regular_est.dow_factors.get(dow, 1.0)
            return regular_est.daily_regular_qty * factor
        
        df['regular_qty'] = df.apply(calc_regular, axis=1)
        df['walkin_qty'] = (df['total_qty'] - df['regular_qty']).clip(lower=0)
        df['regular_share'] = df['regular_qty'] / df['total_qty'].replace(0, 1)
        
        return df


class DecomposedForecaster:
    """
    Forecast demand with separate regular vs walk-in components.
    
    Regular demand: Simple day-of-week adjusted baseline
    Walk-in demand: Signal-aware similarity matching
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            candidates = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent / "aoc_sales.db",
            ]
            for p in candidates:
                if p.exists():
                    db_path = str(p)
                    break
        self.db_path = db_path
        self.decomposer = DemandDecomposer(db_path)
    
    def forecast(
        self,
        date: str,
        sku: str = None,
        category: str = None,
        location: str = None,
        weather: Dict = None,
        signals: Dict = None,
    ) -> DecomposedForecast:
        """
        Generate decomposed forecast for a specific date.
        
        Args:
            date: Target date (YYYY-MM-DD)
            sku: Optional SKU filter
            category: Optional category filter
            location: Store location
            weather: Weather context dict
            signals: Signal context dict
        """
        
        # Parse date
        target_date = datetime.strptime(date, "%Y-%m-%d")
        dow = target_date.weekday()  # 0=Mon
        # Adjust to match SQLite %w (0=Sun)
        sqlite_dow = (dow + 1) % 7
        
        # === REGULAR COMPONENT ===
        regular_est = self.decomposer.estimate_regular_demand(sku, category, location)
        
        dow_factor = regular_est.dow_factors.get(sqlite_dow, 1.0)
        regular_qty = regular_est.daily_regular_qty * dow_factor
        
        # Regular demand has low variance (use CV-based CI)
        regular_cv = 1 - regular_est.consistency_score
        regular_std = regular_qty * regular_cv
        regular_low = max(0, regular_qty - 1.96 * regular_std)
        regular_high = regular_qty + 1.96 * regular_std
        
        # === WALK-IN COMPONENT ===
        # Get historical walk-in for this day of week
        historical = self.decomposer.decompose_historical(sku, category, location, days=180)
        
        if not historical.empty:
            # Filter to same day of week
            dow_history = historical[historical['dow'] == sqlite_dow]
            
            if not dow_history.empty:
                walkin_mean = dow_history['walkin_qty'].mean()
                walkin_std = dow_history['walkin_qty'].std()
                
                # Apply weather/signal adjustments
                walkin_adj = 1.0
                drivers = []
                
                if weather:
                    # Rain penalty
                    if weather.get('precip_mm', 0) > 5:
                        walkin_adj *= 0.95
                        drivers.append("Rain (-5%)")
                    
                    # Cold boost (people stay home, order delivery)
                    if weather.get('temp_c', 15) < 5:
                        walkin_adj *= 1.08
                        drivers.append("Cold weather (+8%)")
                
                if signals:
                    # Payday boost
                    if signals.get('is_payday_window'):
                        walkin_adj *= 1.15
                        drivers.append("Payday window (+15%)")
                    
                    # Holiday boost
                    if signals.get('is_preholiday'):
                        walkin_adj *= 1.10
                        drivers.append("Pre-holiday (+10%)")
                    
                    # Sports event
                    if signals.get('has_home_game'):
                        walkin_adj *= 1.05
                        drivers.append("Home game (+5%)")
                
                walkin_qty = walkin_mean * walkin_adj
                walkin_low = max(0, (walkin_mean - 1.96 * walkin_std) * walkin_adj)
                walkin_high = (walkin_mean + 1.96 * walkin_std) * walkin_adj
            else:
                walkin_qty = 0
                walkin_low = 0
                walkin_high = 0
                drivers = []
        else:
            walkin_qty = 0
            walkin_low = 0
            walkin_high = 0
            drivers = []
        
        # === COMBINE ===
        total_qty = regular_qty + walkin_qty
        total_low = regular_low + walkin_low
        total_high = regular_high + walkin_high
        
        regular_share = regular_qty / total_qty if total_qty > 0 else 0
        
        return DecomposedForecast(
            date=date,
            sku=sku,
            category=category,
            location=location,
            regular_qty=regular_qty,
            walkin_qty=walkin_qty,
            total_qty=total_qty,
            regular_qty_low=regular_low,
            regular_qty_high=regular_high,
            walkin_qty_low=walkin_low,
            walkin_qty_high=walkin_high,
            total_qty_low=total_low,
            total_qty_high=total_high,
            regular_share=regular_share,
            walkin_drivers=drivers,
        )
    
    def forecast_range(
        self,
        start_date: str,
        days: int,
        sku: str = None,
        category: str = None,
        location: str = None,
        weather_forecast: List[Dict] = None,
        signals_forecast: List[Dict] = None,
    ) -> List[DecomposedForecast]:
        """Forecast multiple days."""
        
        start = datetime.strptime(start_date, "%Y-%m-%d")
        forecasts = []
        
        for i in range(days):
            date = (start + timedelta(days=i)).strftime("%Y-%m-%d")
            weather = weather_forecast[i] if weather_forecast and i < len(weather_forecast) else None
            signals = signals_forecast[i] if signals_forecast and i < len(signals_forecast) else None
            
            forecast = self.forecast(date, sku, category, location, weather, signals)
            forecasts.append(forecast)
        
        return forecasts


def demo():
    """Demonstrate decomposed forecasting."""
    
    print("=" * 70)
    print("📊 DECOMPOSED DEMAND FORECASTING")
    print("   Separating regular customers from walk-in traffic")
    print("=" * 70)
    
    decomposer = DemandDecomposer()
    forecaster = DecomposedForecaster()
    
    # Analyze Hash category (where we know there's a regular)
    print("\n\n### HASH CATEGORY ANALYSIS ###\n")
    
    regular_est = decomposer.estimate_regular_demand(category="Hash")
    print(f"Estimated daily regular demand: {regular_est.daily_regular_qty:.1f} units")
    print(f"Regular revenue: ${regular_est.daily_regular_revenue:.2f}/day")
    print(f"Same-day repeat rate: {regular_est.same_day_repeat_rate:.1%}")
    print(f"Consistency score: {regular_est.consistency_score:.1%}")
    print(f"Confidence: {regular_est.confidence:.1%}")
    
    print("\nDay-of-week factors for regulars:")
    dow_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
    for dow, factor in regular_est.dow_factors.items():
        print(f"  {dow_names[dow]}: {factor:.2f}x")
    
    # Historical decomposition
    print("\n\n### HISTORICAL DECOMPOSITION (last 30 days) ###\n")
    
    history = decomposer.decompose_historical(category="Hash", days=30)
    if not history.empty:
        print(history[['date', 'total_qty', 'regular_qty', 'walkin_qty', 'regular_share']].tail(10).to_string())
        
        print(f"\nAverage regular share: {history['regular_share'].mean():.1%}")
        print(f"Regular demand std dev: {history['regular_qty'].std():.1f} (low = predictable)")
        print(f"Walk-in demand std dev: {history['walkin_qty'].std():.1f} (higher = responsive)")
    
    # Forecast comparison
    print("\n\n### 7-DAY FORECAST ###\n")
    
    from datetime import datetime
    today = datetime.now().strftime("%Y-%m-%d")
    
    forecasts = forecaster.forecast_range(
        start_date=today,
        days=7,
        category="Hash"
    )
    
    print(f"{'Date':<12} {'Regular':>8} {'Walk-in':>8} {'Total':>8} {'Reg %':>8} {'Range':>15}")
    print("-" * 70)
    for f in forecasts:
        range_str = f"[{f.total_qty_low:.0f}-{f.total_qty_high:.0f}]"
        print(f"{f.date:<12} {f.regular_qty:>8.1f} {f.walkin_qty:>8.1f} {f.total_qty:>8.1f} {f.regular_share:>7.0%} {range_str:>15}")


if __name__ == "__main__":
    demo()
