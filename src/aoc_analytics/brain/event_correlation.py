"""
Event Impact Correlation

Correlates detected events (Reddit, vibe_signals, etc.) with
actual sales anomalies to learn which event types matter most.

This answers questions like:
- Do road closures downtown hurt our Kingsway store?
- Do concerts at Rogers Arena help or hurt?
- What Reddit keywords correlate with sales spikes?
"""

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


@dataclass
class SalesAnomaly:
    """A detected sales anomaly (unexpected high or low)."""
    date: str
    location: str
    actual_units: float
    expected_units: float
    deviation: float  # as percentage
    deviation_sigma: float  # in standard deviations
    is_positive: bool  # True = higher than expected
    
    def __str__(self) -> str:
        direction = "📈" if self.is_positive else "📉"
        return (
            f"{direction} {self.date} @ {self.location}: "
            f"{self.actual_units:.0f} units ({self.deviation:+.1%}, {self.deviation_sigma:.1f}σ)"
        )


@dataclass
class EventCorrelation:
    """Correlation between an event type and sales impact."""
    event_type: str
    avg_impact: float  # as percentage
    sample_size: int
    std_dev: float
    p_value: float
    confidence: float
    example_events: List[str]
    
    def __str__(self) -> str:
        direction = "📈" if self.avg_impact > 0 else "📉"
        sig = "✓" if self.p_value < 0.1 else "~"
        return (
            f"{sig} {self.event_type}: {self.avg_impact:+.1%} {direction}\n"
            f"   (n={self.sample_size}, p={self.p_value:.3f})"
        )


class EventImpactAnalyzer:
    """
    Analyzes correlation between events and sales anomalies.
    """
    
    DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            possible_paths = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent.parent / "aoc_sales.db",
            ]
            for p in possible_paths:
                if p.exists() and p.stat().st_size > 0:
                    db_path = str(p)
                    break
        self.db_path = str(db_path) if db_path else None
        
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def _get_daily_sales(self, location: str = None, days: int = 365) -> Dict[str, Dict]:
        """Get daily sales data."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        query = """
            SELECT 
                date,
                location,
                SUM(quantity) as qty,
                SUM(subtotal) as rev
            FROM sales
            WHERE date >= ?
        """
        params = [cutoff_str]
        
        if location:
            query += " AND location = ?"
            params.append(location)
        
        query += " GROUP BY date, location ORDER BY date"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            key = f"{row[0]}|{row[1]}"
            dt = datetime.strptime(row[0], "%Y-%m-%d")
            result[key] = {
                "date": row[0],
                "location": row[1],
                "dow": dt.weekday(),
                "qty": row[2] or 0,
                "rev": row[3] or 0,
            }
        
        return result
    
    def _calculate_baselines(self, daily_sales: Dict[str, Dict]) -> Dict[Tuple[str, int], Tuple[float, float]]:
        """Calculate baseline (mean, std) by location and day of week."""
        
        # Group by (location, dow)
        grouped = defaultdict(list)
        for key, data in daily_sales.items():
            group_key = (data["location"], data["dow"])
            grouped[group_key].append(data["qty"])
        
        # Calculate mean and std for each group
        baselines = {}
        for group_key, values in grouped.items():
            if len(values) >= 5:
                baselines[group_key] = (np.mean(values), np.std(values))
        
        return baselines
    
    def detect_anomalies(
        self, 
        location: str = None,
        threshold_sigma: float = 1.5,
        days: int = 365
    ) -> List[SalesAnomaly]:
        """Detect sales anomalies (days significantly above/below expected)."""
        
        daily_sales = self._get_daily_sales(location, days)
        baselines = self._calculate_baselines(daily_sales)
        
        anomalies = []
        
        for key, data in daily_sales.items():
            group_key = (data["location"], data["dow"])
            if group_key not in baselines:
                continue
            
            mean, std = baselines[group_key]
            if std < 1:  # Avoid division by near-zero
                continue
            
            deviation = (data["qty"] - mean) / mean if mean > 0 else 0
            deviation_sigma = (data["qty"] - mean) / std
            
            if abs(deviation_sigma) >= threshold_sigma:
                anomalies.append(SalesAnomaly(
                    date=data["date"],
                    location=data["location"],
                    actual_units=data["qty"],
                    expected_units=mean,
                    deviation=deviation,
                    deviation_sigma=deviation_sigma,
                    is_positive=deviation > 0,
                ))
        
        # Sort by absolute deviation
        anomalies.sort(key=lambda x: abs(x.deviation_sigma), reverse=True)
        
        return anomalies
    
    def load_reddit_events(self) -> Dict[str, List[Dict]]:
        """Load Reddit events indexed by date."""
        
        events_file = Path(__file__).parent / "data" / "reddit_events.json"
        if not events_file.exists():
            return {}
        
        with open(events_file) as f:
            data = json.load(f)
        
        # Index by event_date
        by_date = defaultdict(list)
        for event in data.get("events", []):
            if event.get("event_date"):
                by_date[event["event_date"]].append(event)
        
        return dict(by_date)
    
    def load_signal_events(self, days: int = 365) -> Dict[str, List[str]]:
        """Load events from vibe_signals (cruise ships, games, etc.) indexed by date."""
        
        # Try to import vibe_signals
        try:
            from aoc_analytics.core.signals.vibe_signals import (
                CruiseSchedule,
                SportsSchedule,
                AcademicCalendar,
            )
        except ImportError:
            return {}
        
        events_by_date = defaultdict(list)
        
        today = date.today()
        start = today - timedelta(days=days)
        
        current = start
        while current <= today:
            date_str = current.strftime("%Y-%m-%d")
            
            # Check cruise ships
            if CruiseSchedule.is_cruise_season(current):
                likelihood = CruiseSchedule.estimate_cruise_likelihood(current)
                if likelihood > 0.3:
                    events_by_date[date_str].append(f"cruise_ships:{likelihood:.1f}")
            
            # Check Canucks
            is_game, prob = SportsSchedule.is_likely_canucks_home_game(current)
            if is_game and prob > 0.25:
                events_by_date[date_str].append("canucks_home_game")
            
            # Check NFL Sunday
            is_nfl, game_type = SportsSchedule.is_nfl_game_window(current)
            if is_nfl and game_type:
                events_by_date[date_str].append(f"nfl:{game_type}")
            
            # Check academic calendar
            period, stress = AcademicCalendar.get_academic_period(current)
            if stress > 0.5:  # High stress periods
                events_by_date[date_str].append(f"academic:{period}")
            
            current += timedelta(days=1)
        
        return dict(events_by_date)
    
    def correlate_events_with_anomalies(
        self,
        anomalies: List[SalesAnomaly],
        signal_events: Dict[str, List[str]],
    ) -> Dict[str, List[float]]:
        """Find which event types correlate with anomalies."""
        
        # Map anomalies by date
        anomaly_by_date = defaultdict(list)
        for a in anomalies:
            anomaly_by_date[a.date].append(a)
        
        # Track impacts for each event type
        event_impacts = defaultdict(list)
        
        for date_str, events in signal_events.items():
            if date_str in anomaly_by_date:
                for event in events:
                    # Extract event type (before colon if any)
                    event_type = event.split(":")[0]
                    
                    for anomaly in anomaly_by_date[date_str]:
                        event_impacts[event_type].append(anomaly.deviation)
        
        return dict(event_impacts)
    
    def analyze_event_correlations(
        self,
        event_impacts: Dict[str, List[float]],
        min_samples: int = 5,
    ) -> List[EventCorrelation]:
        """Analyze statistical significance of event correlations."""
        
        if stats is None:
            return []
        
        correlations = []
        
        for event_type, impacts in event_impacts.items():
            if len(impacts) < min_samples:
                continue
            
            avg_impact = np.mean(impacts)
            std_dev = np.std(impacts)
            
            # One-sample t-test against 0 (no impact)
            t_stat, p_value = stats.ttest_1samp(impacts, 0)
            
            confidence = 1 - p_value if p_value < 0.5 else 0.5
            
            correlations.append(EventCorrelation(
                event_type=event_type,
                avg_impact=float(avg_impact),
                sample_size=len(impacts),
                std_dev=float(std_dev),
                p_value=float(p_value),
                confidence=float(confidence),
                example_events=[],
            ))
        
        # Sort by significance
        correlations.sort(key=lambda x: x.p_value)
        
        return correlations
    
    def get_anomaly_explanations(
        self,
        anomalies: List[SalesAnomaly],
        signal_events: Dict[str, List[str]],
        reddit_events: Dict[str, List[Dict]],
    ) -> List[Tuple[SalesAnomaly, List[str]]]:
        """Try to explain each anomaly with detected events."""
        
        explanations = []
        
        for anomaly in anomalies[:20]:  # Top 20 anomalies
            reasons = []
            
            # Check signal events
            if anomaly.date in signal_events:
                for event in signal_events[anomaly.date]:
                    reasons.append(f"[signal] {event}")
            
            # Check Reddit events
            if anomaly.date in reddit_events:
                for event in reddit_events[anomaly.date]:
                    reasons.append(f"[reddit] {event.get('type')}: {event.get('title', '')[:40]}")
            
            explanations.append((anomaly, reasons))
        
        return explanations
    
    def save_analysis(
        self,
        anomalies: List[SalesAnomaly],
        correlations: List[EventCorrelation],
        explanations: List[Tuple[SalesAnomaly, List[str]]],
    ) -> str:
        """Save analysis results to JSON."""
        
        output = {
            "generated": str(datetime.now()),
            "anomalies": [
                {
                    "date": a.date,
                    "location": a.location,
                    "actual": float(a.actual_units),
                    "expected": float(a.expected_units),
                    "deviation_pct": float(round(a.deviation * 100, 1)),
                    "deviation_sigma": float(round(a.deviation_sigma, 2)),
                    "is_positive": bool(a.is_positive),
                }
                for a in anomalies[:50]
            ],
            "correlations": [
                {
                    "event_type": c.event_type,
                    "avg_impact_pct": float(round(c.avg_impact * 100, 1)),
                    "sample_size": int(c.sample_size),
                    "p_value": float(round(c.p_value, 4)),
                    "significant": bool(c.p_value < 0.1),
                }
                for c in correlations
            ],
            "explained_anomalies": [
                {
                    "anomaly": str(a),
                    "reasons": r,
                }
                for a, r in explanations if r
            ],
        }
        
        # Save
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "event_impact_analysis.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate event impact analysis."""
    
    print("=" * 70)
    print("🔗 EVENT IMPACT CORRELATION")
    print("   Which events explain sales anomalies?")
    print("=" * 70)
    print()
    
    analyzer = EventImpactAnalyzer()
    
    # Detect anomalies
    print("Detecting sales anomalies (±1.5σ from baseline)...")
    anomalies = analyzer.detect_anomalies(threshold_sigma=1.5, days=365)
    
    positive = [a for a in anomalies if a.is_positive]
    negative = [a for a in anomalies if not a.is_positive]
    
    print(f"  Found {len(anomalies)} anomalies: {len(positive)} high, {len(negative)} low\n")
    
    # Load events
    print("Loading event data...")
    signal_events = analyzer.load_signal_events(days=365)
    reddit_events = analyzer.load_reddit_events()
    
    print(f"  Signal events: {sum(len(v) for v in signal_events.values())} across {len(signal_events)} days")
    print(f"  Reddit events: {sum(len(v) for v in reddit_events.values())} across {len(reddit_events)} days\n")
    
    # Correlate
    print("Correlating events with anomalies...")
    event_impacts = analyzer.correlate_events_with_anomalies(anomalies, signal_events)
    
    print("\n" + "=" * 70)
    print("📊 EVENT TYPE IMPACT ANALYSIS")
    print("=" * 70 + "\n")
    
    correlations = analyzer.analyze_event_correlations(event_impacts, min_samples=5)
    
    if correlations:
        for corr in correlations:
            print(corr)
    else:
        print("Not enough data to calculate correlations.")
        print("Need more days with both events and anomalies.")
    
    # Try to explain anomalies
    explanations = analyzer.get_anomaly_explanations(anomalies, signal_events, reddit_events)
    
    explained = [e for e in explanations if e[1]]
    unexplained = [e for e in explanations if not e[1]]
    
    print("\n" + "=" * 70)
    print("🔍 TOP ANOMALIES WITH EXPLANATIONS")
    print("=" * 70 + "\n")
    
    for anomaly, reasons in explained[:10]:
        print(anomaly)
        for reason in reasons:
            print(f"      → {reason}")
        print()
    
    if unexplained:
        print(f"\n⚠️ {len(unexplained)} anomalies without detected explanations")
        print("   These might be:")
        print("   - Local events we didn't capture")
        print("   - Random variation")
        print("   - Internal factors (staffing, inventory)")
    
    # Save
    output_file = analyzer.save_analysis(anomalies, correlations, explanations)
    print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    demo()
