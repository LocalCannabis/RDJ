"""
Brain Integration - Connect AI brain to existing AOC signal system.

This module piggybacks on the existing signals:
- Weather → weather_daily table (temp, precip, weather_code)
- Events → vibe_signals.py (sports, concerts, holidays)
- Calendar → external_calendar.py (holidays, BC events)
- Predictor → predictor.py (23-dim feature vectors)

The brain doesn't reinvent this - it:
1. READS these signals
2. FORMS hypotheses about which matter
3. VALIDATES against actual sales
4. SUGGESTS weight adjustments
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional, Dict, List, Any, Tuple
import logging

import numpy as np

from aoc_analytics.core.db_adapter import get_connection

# Import existing signal infrastructure
try:
    from aoc_analytics.core.signals.vibe_signals import (
        VibeEngine,
        DayVibe,
        get_vibe_for_date,
        SportsSchedule,
        WeatherVibe,
    )
    from aoc_analytics.core.signals.external_calendar import (
        CalendarService,
        fetch_holidays,
        is_holiday,
    )
    from aoc_analytics.core.signals.builder import (
        _score_at_home,
        _score_out_and_about,
        _score_local_vibe,
    )
    from aoc_analytics.core.predictor import SimilarityConfig
    SIGNALS_AVAILABLE = True
except ImportError:
    SIGNALS_AVAILABLE = False
    VibeEngine = None
    CalendarService = None
    SimilarityConfig = None

logger = logging.getLogger(__name__)


@dataclass
class SignalSnapshot:
    """Snapshot of all signals for a given date."""
    date: date
    
    # From behavioral_signals_fact
    at_home: float = 0.5
    out_and_about: float = 0.5
    holiday: float = 0.0
    payday: float = 0.0
    sports: float = 0.0
    concert: float = 0.0
    local_vibe: float = 0.5
    
    # From weather_daily
    temp_avg: Optional[float] = None
    precip_mm: Optional[float] = None
    weather_code: Optional[int] = None
    
    # From vibe engine
    couch_index: float = 0.5
    party_index: float = 0.0
    stress_index: float = 0.0
    has_major_event: bool = False
    dominant_vibe: Optional[str] = None
    
    # Derived
    is_weekend: bool = False
    is_friday: bool = False
    day_of_week: int = 0
    month: int = 1
    
    # Actual outcome (filled after the fact)
    actual_revenue: Optional[float] = None
    actual_transactions: Optional[int] = None
    expected_revenue: Optional[float] = None  # What we predicted


@dataclass
class SignalHypothesis:
    """A hypothesis about a signal's impact."""
    signal_name: str
    condition: str  # e.g., "at_home > 0.7"
    expected_effect: str  # e.g., "revenue +15%"
    tests: int = 0
    correct: int = 0
    
    @property
    def accuracy(self) -> float:
        return self.correct / self.tests if self.tests > 0 else 0.0
    
    @property
    def is_validated(self) -> bool:
        return self.tests >= 5 and self.accuracy >= 0.6


class SignalAnalyzer:
    """
    Analyzes existing signals against actual outcomes.
    
    Doesn't compute new signals - reads what's already there
    and finds what actually correlates with sales.
    
    Uses:
    - weather_daily table (already populated)
    - sales table (transaction history)
    - vibe_signals.py (sports, events, calendar)
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        try:
            self.vibe_engine = VibeEngine()
            self.calendar = CalendarService()
        except Exception:
            self.vibe_engine = None
            self.calendar = None
    
    def get_signal_snapshot(self, target_date: date) -> SignalSnapshot:
        """Get all signals for a specific date."""
        conn = get_connection(self.db_path)
        
        snapshot = SignalSnapshot(date=target_date)
        snapshot.day_of_week = target_date.weekday()
        snapshot.is_weekend = target_date.weekday() >= 5
        snapshot.is_friday = target_date.weekday() == 4
        snapshot.month = target_date.month
        
        date_str = target_date.isoformat()
        
        # Get behavioral signals (already computed)
        row = conn.execute("""
            SELECT 
                AVG(at_home) as at_home,
                AVG(out_and_about) as out_and_about,
                AVG(holiday) as holiday,
                AVG(payday) as payday,
                AVG(sports) as sports,
                AVG(concert) as concert,
                AVG(local_vibe) as local_vibe
            FROM behavioral_signals_fact
            WHERE date = ?
        """, (date_str,)).fetchone()
        
        if row and row[0] is not None:
            snapshot.at_home = row[0]
            snapshot.out_and_about = row[1]
            snapshot.holiday = row[2]
            snapshot.payday = row[3]
            snapshot.sports = row[4]
            snapshot.concert = row[5]
            snapshot.local_vibe = row[6]
        
        # Get weather (already stored)
        weather = conn.execute("""
            SELECT temperature_2m_mean, precipitation_sum, weather_code
            FROM weather_daily
            WHERE date = ?
        """, (date_str,)).fetchone()
        
        if weather:
            snapshot.temp_avg = weather[0]
            snapshot.precip_mm = weather[1]
            snapshot.weather_code = weather[2]
        
        # Get vibe from existing engine
        try:
            vibe = get_vibe_for_date(target_date, self.vibe_engine)
            if vibe:
                snapshot.couch_index = vibe.couch_index
                snapshot.party_index = vibe.party_index
                snapshot.stress_index = vibe.stress_index
                snapshot.has_major_event = vibe.has_major_event
                snapshot.dominant_vibe = vibe.dominant_vibe.value if vibe.dominant_vibe else None
        except Exception:
            pass
        
        # Get actual sales (for validation)
        sales = conn.execute("""
            SELECT SUM(subtotal), COUNT(DISTINCT invoice_id)
            FROM sales
            WHERE date = ?
        """, (date_str,)).fetchone()
        
        if sales and sales[0]:
            snapshot.actual_revenue = sales[0]
            snapshot.actual_transactions = sales[1]
        
        conn.close()
        return snapshot
    
    def analyze_signal_correlations(
        self, 
        start_date: str, 
        end_date: str
    ) -> Dict[str, Dict[str, float]]:
        """
        Analyze which signals actually correlate with sales deviations.
        
        Returns dict of signal -> correlation stats
        """
        conn = get_connection(self.db_path)
        
        # Get daily data with signals and sales
        df_query = """
            WITH daily_sales AS (
                SELECT 
                    date,
                    SUM(subtotal) as revenue,
                    COUNT(DISTINCT invoice_id) as txns
                FROM sales
                WHERE date BETWEEN ? AND ?
                GROUP BY date
            ),
            daily_signals AS (
                SELECT 
                    date,
                    AVG(at_home) as at_home,
                    AVG(out_and_about) as out_and_about,
                    AVG(holiday) as holiday,
                    AVG(payday) as payday,
                    AVG(sports) as sports,
                    AVG(concert) as concert,
                    AVG(local_vibe) as local_vibe
                FROM behavioral_signals_fact
                WHERE date BETWEEN ? AND ?
                GROUP BY date
            ),
            daily_weather AS (
                SELECT 
                    date,
                    temperature_2m_mean as temp,
                    precipitation_sum as precip
                FROM weather_daily
                WHERE date BETWEEN ? AND ?
            )
            SELECT 
                s.date,
                strftime('%w', s.date) as dow,
                s.revenue,
                s.txns,
                COALESCE(sig.at_home, 0.5) as at_home,
                COALESCE(sig.out_and_about, 0.5) as out_and_about,
                COALESCE(sig.holiday, 0) as holiday,
                COALESCE(sig.payday, 0) as payday,
                COALESCE(sig.sports, 0) as sports,
                COALESCE(sig.concert, 0) as concert,
                COALESCE(sig.local_vibe, 0.5) as local_vibe,
                w.temp,
                w.precip
            FROM daily_sales s
            LEFT JOIN daily_signals sig ON s.date = sig.date
            LEFT JOIN daily_weather w ON s.date = w.date
            ORDER BY s.date
        """
        
        rows = conn.execute(
            df_query, 
            (start_date, end_date) * 3
        ).fetchall()
        
        if not rows:
            conn.close()
            return {}
        
        # Build arrays for correlation analysis
        import numpy as np
        
        data = {
            'revenue': [],
            'dow': [],
            'at_home': [],
            'out_and_about': [],
            'holiday': [],
            'payday': [],
            'sports': [],
            'concert': [],
            'local_vibe': [],
            'temp': [],
            'precip': [],
        }
        
        for row in rows:
            data['revenue'].append(row[2])
            data['dow'].append(int(row[1]))
            data['at_home'].append(row[4])
            data['out_and_about'].append(row[5])
            data['holiday'].append(row[6])
            data['payday'].append(row[7])
            data['sports'].append(row[8])
            data['concert'].append(row[9])
            data['local_vibe'].append(row[10])
            data['temp'].append(row[11] if row[11] else 15.0)
            data['precip'].append(row[12] if row[12] else 0.0)
        
        # Normalize revenue by DOW (remove known DOW effect)
        revenue = np.array(data['revenue'])
        dow = np.array(data['dow'])
        
        dow_means = {}
        for d in range(7):
            mask = dow == d
            if mask.sum() > 0:
                dow_means[d] = revenue[mask].mean()
        
        overall_mean = revenue.mean()
        normalized_revenue = np.zeros_like(revenue)
        for i, (r, d) in enumerate(zip(revenue, dow)):
            dow_mean = dow_means.get(d, overall_mean)
            normalized_revenue[i] = (r - dow_mean) / dow_mean * 100  # % deviation from DOW norm
        
        # Compute correlations with normalized revenue
        results = {}
        signals = ['at_home', 'out_and_about', 'holiday', 'payday', 
                   'sports', 'concert', 'local_vibe', 'temp', 'precip']
        
        for signal in signals:
            signal_arr = np.array(data[signal])
            
            # Skip if no variance
            if signal_arr.std() < 0.001:
                continue
            
            # Pearson correlation
            corr = np.corrcoef(normalized_revenue, signal_arr)[0, 1]
            
            # Effect size: average revenue deviation when signal is high vs low
            median = np.median(signal_arr)
            high_mask = signal_arr > median
            low_mask = signal_arr <= median
            
            high_effect = normalized_revenue[high_mask].mean() if high_mask.sum() > 0 else 0
            low_effect = normalized_revenue[low_mask].mean() if low_mask.sum() > 0 else 0
            
            results[signal] = {
                'correlation': round(corr, 3) if not np.isnan(corr) else 0,
                'high_signal_effect': round(high_effect, 1),
                'low_signal_effect': round(low_effect, 1),
                'effect_spread': round(high_effect - low_effect, 1),
                'samples': len(revenue),
            }
        
        conn.close()
        return results
    
    def suggest_weight_adjustments(
        self,
        correlations: Dict[str, Dict[str, float]],
        current_config: Optional[SimilarityConfig] = None
    ) -> Dict[str, float]:
        """
        Based on actual correlations, suggest weight adjustments
        for the predictor's similarity scoring.
        """
        config = current_config or SimilarityConfig()
        current_weights = config.weights
        
        suggestions = {}
        
        # Map signal names to weight keys
        signal_to_weight = {
            'at_home': 'at_home',
            'out_and_about': 'out_and_about',
            'holiday': 'holiday',
            'payday': 'payday',
            'sports': 'home_game',
            'concert': 'concert',
            'temp': 'temp',
            'precip': 'precip',
        }
        
        for signal, stats in correlations.items():
            weight_key = signal_to_weight.get(signal)
            if not weight_key:
                continue
            
            current = current_weights.get(weight_key, 1.0)
            
            # Strong correlation (>0.15) → increase weight
            # Weak/no correlation → decrease weight
            corr = abs(stats.get('correlation', 0))
            effect = abs(stats.get('effect_spread', 0))
            
            if corr > 0.2 and effect > 5:
                # Signal matters - suggest higher weight
                suggested = min(current * 1.5, 5.0)
            elif corr < 0.05 or effect < 2:
                # Signal doesn't matter much - suggest lower weight
                suggested = max(current * 0.5, 0.1)
            else:
                suggested = current
            
            if abs(suggested - current) > 0.1:
                suggestions[weight_key] = {
                    'current': current,
                    'suggested': round(suggested, 1),
                    'reason': f"corr={stats['correlation']:.2f}, effect={stats['effect_spread']:.1f}%"
                }
        
        return suggestions
    
    def find_anomaly_explanations(
        self,
        target_date: date,
        threshold_pct: float = 15.0
    ) -> Optional[Dict[str, Any]]:
        """
        If a day's sales were anomalous, explain why using signals.
        """
        snapshot = self.get_signal_snapshot(target_date)
        
        if snapshot.actual_revenue is None:
            return None
        
        # Get expected revenue (same DOW average)
        conn = get_connection(self.db_path)
        dow = target_date.weekday()
        # SQLite weekday: 0=Sunday, but Python weekday: 0=Monday
        sqlite_dow = (dow + 1) % 7
        
        expected = conn.execute("""
            SELECT AVG(revenue) FROM (
                SELECT SUM(subtotal) as revenue
                FROM sales
                WHERE strftime('%w', date) = ?
                AND date >= date(?, '-90 days')
                AND date < ?
                GROUP BY date
            )
        """, (str(sqlite_dow), target_date.isoformat(), target_date.isoformat())).fetchone()[0]
        
        conn.close()
        
        if not expected:
            return None
        
        deviation_pct = ((snapshot.actual_revenue - expected) / expected) * 100
        
        if abs(deviation_pct) < threshold_pct:
            return None  # Not anomalous
        
        # Find likely explanations from signals
        explanations = []
        
        if snapshot.holiday > 0.5:
            explanations.append(f"Holiday effect (holiday={snapshot.holiday:.2f})")
        
        if snapshot.payday > 0.5:
            explanations.append(f"Payday effect (payday={snapshot.payday:.2f})")
        
        if snapshot.at_home > 0.7:
            explanations.append(f"High stay-at-home index ({snapshot.at_home:.2f}) - likely bad weather")
        elif snapshot.at_home < 0.3:
            explanations.append(f"Low stay-at-home index ({snapshot.at_home:.2f}) - nice weather")
        
        if snapshot.sports > 0.5:
            explanations.append(f"Sports event (sports={snapshot.sports:.2f})")
        
        if snapshot.concert > 0.5:
            explanations.append(f"Concert/event (concert={snapshot.concert:.2f})")
        
        if snapshot.has_major_event:
            explanations.append(f"Major vibe event: {snapshot.dominant_vibe}")
        
        if snapshot.precip_mm and snapshot.precip_mm > 10:
            explanations.append(f"Heavy precipitation ({snapshot.precip_mm:.1f}mm)")
        
        if snapshot.temp_avg and snapshot.temp_avg < 5:
            explanations.append(f"Very cold ({snapshot.temp_avg:.1f}°C)")
        elif snapshot.temp_avg and snapshot.temp_avg > 25:
            explanations.append(f"Hot weather ({snapshot.temp_avg:.1f}°C)")
        
        return {
            'date': target_date.isoformat(),
            'actual_revenue': round(snapshot.actual_revenue, 2),
            'expected_revenue': round(expected, 2),
            'deviation_pct': round(deviation_pct, 1),
            'direction': 'above' if deviation_pct > 0 else 'below',
            'explanations': explanations if explanations else ['No clear signal explanation'],
            'signals': {
                'at_home': snapshot.at_home,
                'out_and_about': snapshot.out_and_about,
                'holiday': snapshot.holiday,
                'payday': snapshot.payday,
                'sports': snapshot.sports,
                'temp': snapshot.temp_avg,
                'precip': snapshot.precip_mm,
            }
        }


def analyze_signals_demo(db_path: str = "aoc_sales.db"):
    """Demo: Analyze which signals actually matter using raw weather + sales data."""
    
    print("=" * 70)
    print("AOC SIGNAL ANALYSIS - Finding What Actually Matters")
    print("=" * 70)
    
    conn = get_connection(db_path)
    
    # Join weather with daily sales
    query = """
        WITH daily_sales AS (
            SELECT 
                date,
                strftime('%w', date) as dow,
                strftime('%m', date) as month,
                SUM(subtotal) as revenue,
                COUNT(DISTINCT invoice_id) as txns
            FROM sales
            WHERE date >= '2024-01-01' AND date <= '2024-11-30'
            GROUP BY date
        )
        SELECT 
            s.date, s.dow, s.month, s.revenue, s.txns,
            w.temp_avg_c, w.precip_mm, w.rain_mm, w.snow_mm,
            w.wind_max_kph, w.weather_code
        FROM daily_sales s
        LEFT JOIN weather_daily w 
            ON s.date = w.date AND w.location = 'Parksville'
        WHERE w.date IS NOT NULL
        ORDER BY s.date
    """
    
    rows = conn.execute(query).fetchall()
    print(f"\nAnalyzing {len(rows)} days with weather data")
    
    if not rows:
        print("No data found!")
        conn.close()
        return
    
    # Build arrays
    revenue = np.array([r[3] for r in rows])
    dow = np.array([int(r[1]) for r in rows])
    month = np.array([int(r[2]) for r in rows])
    temp = np.array([r[5] if r[5] else 15.0 for r in rows])
    precip = np.array([r[6] if r[6] else 0.0 for r in rows])
    rain = np.array([r[7] if r[7] else 0.0 for r in rows])
    code = np.array([r[10] if r[10] else 0 for r in rows])
    
    # Normalize by DOW
    dow_means = {d: revenue[dow == d].mean() for d in range(7)}
    normalized = np.array([(r - dow_means[d]) / dow_means[d] * 100 
                           for r, d in zip(revenue, dow)])
    
    print("\n" + "=" * 70)
    print("📊 SIGNAL CORRELATIONS (DOW-normalized)")
    print("=" * 70)
    
    signals = {
        'Temperature': temp,
        'Precipitation': precip,
        'Rain': rain,
        'Month': month,
    }
    
    for name, signal in signals.items():
        if signal.std() < 0.001:
            continue
        corr = np.corrcoef(normalized, signal)[0, 1]
        median = np.median(signal)
        high_effect = normalized[signal > median].mean()
        low_effect = normalized[signal <= median].mean()
        effect = high_effect - low_effect
        marker = "🔥" if abs(effect) > 3 else "  "
        print(f"  {marker} {name:15} corr={corr:+.3f}  effect={effect:+.1f}%")
    
    # Rain analysis
    print("\n🌧️ RAIN EFFECT:")
    no_rain = normalized[rain < 0.1]
    heavy_rain = normalized[rain >= 5]
    print(f"  No rain:    {no_rain.mean():+.1f}% ({len(no_rain)} days)")
    print(f"  Heavy rain: {heavy_rain.mean():+.1f}% ({len(heavy_rain)} days)")
    
    # Temperature analysis
    print("\n🌡️ TEMPERATURE EFFECT:")
    cold = normalized[temp < 10]
    mild = normalized[(temp >= 15) & (temp < 22)]
    print(f"  Cold (<10°C):      {cold.mean():+.1f}% ({len(cold)} days)")
    print(f"  Mild (15-22°C):    {mild.mean():+.1f}% ({len(mild)} days)")
    
    # Monthly patterns
    print("\n📅 MONTHLY PATTERNS:")
    month_names = ['', 'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    for m in range(1, 13):
        mask = month == m
        if mask.sum() > 0:
            effect = normalized[mask].mean()
            marker = "📈" if effect > 5 else "📉" if effect < -5 else "  "
            print(f"  {marker} {month_names[m]}: {effect:+.1f}%")
    
    conn.close()
    print("\n" + "=" * 70)


if __name__ == "__main__":
    analyze_signals_demo("/home/macklemoron/Projects/aoc-analytics/aoc_sales.db")
