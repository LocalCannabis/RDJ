"""
Signal Magnitude Learning

Connects the brain to existing vibe_signals to learn the actual
magnitude of impact from different signal types.

vibe_signals provides the DETECTION (when is there a cruise ship?)
This module learns the MAGNITUDE (how much does it actually affect sales?)

Signals we can learn magnitude for:
- Cruise ships (April-October)
- Canucks home games (NHL season)
- NFL Sundays (Seahawks watching)
- Whitecaps/Lions home games
- Academic calendar (UBC/SFU)
- Weather patterns (rain streak, first nice day)
- Full moons (yes, really)
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

# Import vibe signals - handle both module and script contexts
try:
    from aoc_analytics.core.signals.vibe_signals import (
        CruiseSchedule,
        SportsSchedule,
        AcademicCalendar,
        WeatherVibe,
        VibeEngine,
    )
except ImportError:
    from ..core.signals.vibe_signals import (
        CruiseSchedule,
        SportsSchedule,
        AcademicCalendar,
        WeatherVibe,
        VibeEngine,
    )


@dataclass
class SignalImpact:
    """Measured impact of a signal on sales."""
    
    signal_name: str
    signal_type: str  # cruise, sports, academic, weather, etc.
    
    # Measured impact
    avg_sales_lift: float  # % change vs baseline (e.g., 0.15 = +15%)
    avg_revenue_lift: float
    
    # Statistical confidence
    sample_size: int  # Number of days with this signal
    std_dev: float    # Standard deviation of lift
    confidence: float  # Statistical confidence (0-1)
    p_value: float    # Statistical significance
    
    # Category-specific impacts
    category_impacts: Dict[str, float] = field(default_factory=dict)
    
    # Time patterns
    hour_impacts: Dict[int, float] = field(default_factory=dict)
    
    def __str__(self):
        direction = "📈" if self.avg_sales_lift > 0 else "📉"
        return (
            f"\n{direction} {self.signal_name} ({self.signal_type})\n"
            f"   Sales lift: {self.avg_sales_lift:+.1%} ± {self.std_dev:.1%}\n"
            f"   Revenue lift: {self.avg_revenue_lift:+.1%}\n"
            f"   Sample size: {self.sample_size} days\n"
            f"   Confidence: {self.confidence:.0%} (p={self.p_value:.3f})\n"
        )


class SignalMagnitudeLearner:
    """
    Learn the actual magnitude of impact from different signals.
    
    Uses historical sales data correlated with signal presence
    to measure real-world lift.
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
        self.vibe_engine = VibeEngine()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def _get_daily_sales(self, location: str = None, days: int = 365) -> Dict[str, Dict]:
        """Get daily sales totals for analysis."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        location_clause = "AND location = ?" if location else ""
        params = [f'-{days} days']
        if location:
            params.append(location)
        
        cursor.execute(f"""
            SELECT 
                date,
                CAST(strftime('%w', date) AS INTEGER) as dow,
                CAST(strftime('%m', date) AS INTEGER) as month,
                SUM(quantity) as total_qty,
                SUM(subtotal) as total_revenue,
                COUNT(DISTINCT invoice_id) as transactions
            FROM sales
            WHERE date >= date('now', ?)
            {location_clause}
            GROUP BY date
            ORDER BY date
        """, params)
        
        daily = {}
        for row in cursor.fetchall():
            daily[row["date"]] = {
                "qty": row["total_qty"],
                "revenue": row["total_revenue"],
                "transactions": row["transactions"],
                "dow": row["dow"],
                "month": row["month"],
            }
        
        conn.close()
        return daily
    
    def _calculate_baseline(self, daily_sales: Dict[str, Dict]) -> Dict[int, float]:
        """Calculate baseline sales by day of week."""
        
        dow_totals = defaultdict(list)
        for date_str, data in daily_sales.items():
            dow_totals[data["dow"]].append(data["qty"])
        
        baselines = {}
        for dow, qtys in dow_totals.items():
            baselines[dow] = np.median(qtys)  # Use median for robustness
        
        return baselines
    
    def learn_cruise_ship_impact(self, location: str = None) -> SignalImpact:
        """Learn the actual impact of cruise ship days on sales."""
        
        daily_sales = self._get_daily_sales(location, days=365)
        baselines = self._calculate_baseline(daily_sales)
        
        cruise_days = []
        non_cruise_days = []
        
        for date_str, data in daily_sales.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            likelihood = CruiseSchedule.estimate_cruise_likelihood(dt)
            
            baseline = baselines.get(data["dow"], np.mean(list(baselines.values())))
            lift = (data["qty"] - baseline) / baseline if baseline > 0 else 0
            
            if likelihood > 0.5:  # Likely cruise ship day
                cruise_days.append({
                    "date": date_str,
                    "qty": data["qty"],
                    "lift": lift,
                    "likelihood": likelihood,
                })
            elif likelihood < 0.1:  # Definitely no cruise
                non_cruise_days.append(lift)
        
        if len(cruise_days) < 10:
            return SignalImpact(
                signal_name="Cruise Ship Day",
                signal_type="cruise",
                avg_sales_lift=0,
                avg_revenue_lift=0,
                sample_size=len(cruise_days),
                std_dev=0,
                confidence=0,
                p_value=1.0,
            )
        
        cruise_lifts = [d["lift"] for d in cruise_days]
        
        avg_lift = np.mean(cruise_lifts)
        std_dev = np.std(cruise_lifts)
        
        # T-test against non-cruise days
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(cruise_lifts, non_cruise_days)
        
        confidence = 1 - p_value if p_value < 0.5 else 0.5
        
        return SignalImpact(
            signal_name="Cruise Ship Day",
            signal_type="cruise",
            avg_sales_lift=avg_lift,
            avg_revenue_lift=avg_lift,  # Assume revenue scales with qty
            sample_size=len(cruise_days),
            std_dev=std_dev,
            confidence=confidence,
            p_value=p_value,
        )
    
    def learn_sports_impact(self, sport: str = "nhl", location: str = None) -> SignalImpact:
        """Learn impact of sports events (Canucks, NFL, Whitecaps, Lions)."""
        
        daily_sales = self._get_daily_sales(location, days=365)
        baselines = self._calculate_baseline(daily_sales)
        
        game_days = []
        non_game_days = []
        
        for date_str, data in daily_sales.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            baseline = baselines.get(data["dow"], np.mean(list(baselines.values())))
            lift = (data["qty"] - baseline) / baseline if baseline > 0 else 0
            
            if sport == "nhl":
                is_game, prob = SportsSchedule.is_likely_canucks_home_game(dt)
                if is_game and prob > 0.2:
                    game_days.append({"lift": lift, "prob": prob})
                elif not SportsSchedule.is_nhl_season(dt):
                    non_game_days.append(lift)
            
            elif sport == "nfl":
                is_game, game_type = SportsSchedule.is_nfl_game_window(dt)
                if is_game and game_type in ("sunday_football", "super_bowl"):
                    game_days.append({"lift": lift, "type": game_type})
                elif data["dow"] == 6 and not is_game:  # Non-NFL Sunday
                    non_game_days.append(lift)
            
            elif sport == "mls":
                # Whitecaps: MLS season Mar-Oct, mostly weekends
                if SportsSchedule.is_mls_season(dt):
                    # Higher probability weekends (Sat/Sun)
                    if data["dow"] in (5, 6):  # Weekend
                        game_days.append({"lift": lift, "prob": 0.35})
                    elif data["dow"] == 2:  # Wednesday (midweek games)
                        game_days.append({"lift": lift, "prob": 0.15})
                else:
                    if data["dow"] in (5, 6):
                        non_game_days.append(lift)
            
            elif sport == "cfl":
                # BC Lions: CFL season Jun-Nov, mostly weekends
                if SportsSchedule.is_cfl_season(dt):
                    if data["dow"] in (4, 5, 6):  # Fri/Sat/Sun
                        game_days.append({"lift": lift, "prob": 0.25})
                else:
                    if data["dow"] in (5, 6):
                        non_game_days.append(lift)
        
        if len(game_days) < 10:
            return SignalImpact(
                signal_name=f"{sport.upper()} Game Day",
                signal_type="sports",
                avg_sales_lift=0,
                avg_revenue_lift=0,
                sample_size=len(game_days),
                std_dev=0,
                confidence=0,
                p_value=1.0,
            )
        
        game_lifts = [d["lift"] for d in game_days]
        
        avg_lift = np.mean(game_lifts)
        std_dev = np.std(game_lifts)
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(game_lifts, non_game_days) if non_game_days else (0, 1.0)
        
        confidence = 1 - p_value if p_value < 0.5 else 0.5
        
        sport_names = {
            "nhl": "Canucks Home Game", 
            "nfl": "NFL Sunday", 
            "mls": "Whitecaps Home Game",
            "cfl": "BC Lions Home Game"
        }
        
        return SignalImpact(
            signal_name=sport_names.get(sport, f"{sport.upper()} Game"),
            signal_type="sports",
            avg_sales_lift=avg_lift,
            avg_revenue_lift=avg_lift,
            sample_size=len(game_days),
            std_dev=std_dev,
            confidence=confidence,
            p_value=p_value,
        )
    
    def learn_academic_impact(self, event_type: str = "finals", location: str = None) -> SignalImpact:
        """Learn impact of academic calendar events."""
        
        daily_sales = self._get_daily_sales(location, days=365)
        baselines = self._calculate_baseline(daily_sales)
        
        event_days = []
        non_event_days = []
        
        for date_str, data in daily_sales.items():
            dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            
            baseline = baselines.get(data["dow"], np.mean(list(baselines.values())))
            lift = (data["qty"] - baseline) / baseline if baseline > 0 else 0
            
            # Check for academic events using get_academic_period
            period, stress = AcademicCalendar.get_academic_period(dt)
            
            if event_type == "finals" and period in ("winter_finals", "spring_finals"):
                event_days.append(lift)
            elif event_type == "reading_break" and period == "reading_break":
                event_days.append(lift)
            elif event_type == "move_in" and period in ("fall_start", "winter_start"):
                event_days.append(lift)
            elif period in ("semester", "between_terms", "summer_break"):
                non_event_days.append(lift)
        
        if len(event_days) < 5:
            return SignalImpact(
                signal_name=f"Academic {event_type.replace('_', ' ').title()}",
                signal_type="academic",
                avg_sales_lift=0,
                avg_revenue_lift=0,
                sample_size=len(event_days),
                std_dev=0,
                confidence=0,
                p_value=1.0,
            )
        
        avg_lift = np.mean(event_days)
        std_dev = np.std(event_days)
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(event_days, non_event_days) if non_event_days else (0, 1.0)
        
        confidence = 1 - p_value if p_value < 0.5 else 0.5
        
        return SignalImpact(
            signal_name=f"Academic {event_type.replace('_', ' ').title()}",
            signal_type="academic",
            avg_sales_lift=avg_lift,
            avg_revenue_lift=avg_lift,
            sample_size=len(event_days),
            std_dev=std_dev,
            confidence=confidence,
            p_value=p_value,
        )
    
    def learn_weather_pattern_impact(self, pattern: str = "rain_streak", 
                                      location: str = None) -> SignalImpact:
        """Learn impact of weather patterns (rain streak, first nice day, etc.)."""
        
        daily_sales = self._get_daily_sales(location, days=365)
        baselines = self._calculate_baseline(daily_sales)
        
        # Get weather data
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT date, precip_mm, temp_max_c, temp_min_c
            FROM weather_daily
            WHERE date >= date('now', '-365 days')
            ORDER BY date
        """)
        
        weather = {row["date"]: row for row in cursor.fetchall()}
        conn.close()
        
        pattern_days = []
        normal_days = []
        
        # Build 7-day precip history for each day
        sorted_dates = sorted(weather.keys())
        
        for i, date_str in enumerate(sorted_dates):
            if date_str not in daily_sales:
                continue
            
            data = daily_sales[date_str]
            baseline = baselines.get(data["dow"], np.mean(list(baselines.values())))
            lift = (data["qty"] - baseline) / baseline if baseline > 0 else 0
            
            # Get 7-day precip history
            precip_history = []
            for j in range(max(0, i-6), i+1):
                if j < len(sorted_dates):
                    hist_date = sorted_dates[j]
                    if hist_date in weather:
                        precip_history.append(weather[hist_date]["precip_mm"] or 0)
            
            if len(precip_history) >= 7:
                streak, is_first_nice = WeatherVibe.calculate_rain_streak(precip_history)
                
                if pattern == "rain_streak" and streak >= 3:
                    pattern_days.append({"lift": lift, "streak": streak})
                elif pattern == "first_nice_day" and is_first_nice:
                    pattern_days.append({"lift": lift, "streak": streak})
                elif streak < 2 and not is_first_nice:
                    normal_days.append(lift)
        
        if len(pattern_days) < 5:
            return SignalImpact(
                signal_name=pattern.replace("_", " ").title(),
                signal_type="weather",
                avg_sales_lift=0,
                avg_revenue_lift=0,
                sample_size=len(pattern_days),
                std_dev=0,
                confidence=0,
                p_value=1.0,
            )
        
        pattern_lifts = [d["lift"] for d in pattern_days]
        
        avg_lift = np.mean(pattern_lifts)
        std_dev = np.std(pattern_lifts)
        
        from scipy import stats
        t_stat, p_value = stats.ttest_ind(pattern_lifts, normal_days) if normal_days else (0, 1.0)
        
        confidence = 1 - p_value if p_value < 0.5 else 0.5
        
        return SignalImpact(
            signal_name=pattern.replace("_", " ").title(),
            signal_type="weather",
            avg_sales_lift=avg_lift,
            avg_revenue_lift=avg_lift,
            sample_size=len(pattern_days),
            std_dev=std_dev,
            confidence=confidence,
            p_value=p_value,
        )
    
    def learn_all_signals(self, location: str = None) -> Dict[str, SignalImpact]:
        """Learn magnitude for all signal types."""
        
        results = {}
        
        # Cruise ships
        print("Learning cruise ship impact...")
        results["cruise"] = self.learn_cruise_ship_impact(location)
        
        # Sports - Vancouver teams
        print("Learning NHL (Canucks) impact...")
        results["nhl"] = self.learn_sports_impact("nhl", location)
        
        print("Learning NFL Sunday impact...")
        results["nfl"] = self.learn_sports_impact("nfl", location)
        
        print("Learning MLS (Whitecaps) impact...")
        results["mls"] = self.learn_sports_impact("mls", location)
        
        print("Learning CFL (BC Lions) impact...")
        results["cfl"] = self.learn_sports_impact("cfl", location)
        results["nfl"] = self.learn_sports_impact("nfl", location)
        
        # Academic
        print("Learning finals week impact...")
        results["finals"] = self.learn_academic_impact("finals", location)
        
        print("Learning reading break impact...")
        results["reading_break"] = self.learn_academic_impact("reading_break", location)
        
        print("Learning move-in week impact...")
        results["move_in"] = self.learn_academic_impact("move_in", location)
        
        # Weather patterns
        print("Learning rain streak impact...")
        results["rain_streak"] = self.learn_weather_pattern_impact("rain_streak", location)
        
        print("Learning first nice day impact...")
        results["first_nice_day"] = self.learn_weather_pattern_impact("first_nice_day", location)
        
        return results
    
    def save_learned_magnitudes(self, location: str = None) -> Dict[str, Any]:
        """
        Learn all signals and save to a JSON file for predictor use.
        
        Returns dict suitable for predictor weight tuning:
        {
            "signal_name": {
                "lift": 0.058,  # as decimal
                "confidence": 0.99,
                "predictor_weight": 2.5,  # suggested weight for predictor
                "actionable": True  # p < 0.1 and sample >= 10
            }
        }
        """
        import json
        from pathlib import Path
        
        results = self.learn_all_signals(location)
        
        # Convert to predictor-friendly format
        output = {}
        for key, impact in results.items():
            is_actionable = impact.p_value < 0.1 and impact.sample_size >= 10
            
            # Calculate suggested predictor weight
            # Higher impact + higher confidence = higher weight
            base_weight = abs(impact.avg_sales_lift) * 10  # Scale to ~0-1
            confidence_factor = impact.confidence
            suggested_weight = round(float(base_weight * confidence_factor * 3), 1)  # Scale up for predictor
            
            output[key] = {
                "signal_name": impact.signal_name,
                "signal_type": impact.signal_type,
                "lift": float(round(impact.avg_sales_lift, 4)),
                "lift_pct": f"{impact.avg_sales_lift:+.1%}",
                "std_dev": float(round(impact.std_dev, 4)),
                "p_value": float(round(impact.p_value, 4)),
                "confidence": float(round(impact.confidence, 2)),
                "sample_size": int(impact.sample_size),
                "predictor_weight": float(suggested_weight) if is_actionable else 0.0,
                "actionable": bool(is_actionable),
            }
        
        # Save to brain data directory
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "learned_signal_magnitudes.json"
        with open(output_file, "w") as f:
            json.dump({
                "generated": str(datetime.now()),
                "location": location or "all",
                "signals": output
            }, f, indent=2)
        
        print(f"\n💾 Saved learned magnitudes to: {output_file}")
        
        return output
    
    @staticmethod
    def load_learned_magnitudes() -> Dict[str, Any]:
        """Load previously learned magnitudes for predictor use."""
        import json
        from pathlib import Path
        
        brain_dir = Path(__file__).parent / "data"
        magnitude_file = brain_dir / "learned_signal_magnitudes.json"
        
        if not magnitude_file.exists():
            return {}
        
        with open(magnitude_file) as f:
            return json.load(f)
    
    @staticmethod
    def get_predictor_weights() -> Dict[str, float]:
        """Get weights suitable for predictor SimilarityConfig."""
        data = SignalMagnitudeLearner.load_learned_magnitudes()
        
        if not data or "signals" not in data:
            return {}
        
        # Map signal keys to predictor weight names
        weight_mapping = {
            "cruise": "cruise_ship",
            "nhl": "home_game",  # Canucks
            "mls": "whitecaps",
            "cfl": "lions",
            "nfl": "nfl_sunday",
            "finals": "academic_finals",
            "reading_break": "academic_reading",
            "move_in": "academic_movein",
            "rain_streak": "rain_streak",
            "first_nice_day": "first_nice_day",
        }
        
        weights = {}
        for key, signal in data.get("signals", {}).items():
            if signal.get("actionable"):
                weight_name = weight_mapping.get(key, key)
                weights[weight_name] = signal.get("predictor_weight", 0.0)
        
        return weights


def demo():
    """Demonstrate signal magnitude learning."""
    
    print("=" * 70)
    print("📊 SIGNAL MAGNITUDE LEARNING")
    print("   Measuring actual impact of vibe signals on sales")
    print("=" * 70)
    
    learner = SignalMagnitudeLearner()
    
    results = learner.learn_all_signals()
    
    print("\n\n### LEARNED SIGNAL IMPACTS ###\n")
    
    # Sort by absolute impact
    sorted_results = sorted(
        results.items(), 
        key=lambda x: abs(x[1].avg_sales_lift) if x[1].sample_size > 5 else 0,
        reverse=True
    )
    
    for name, impact in sorted_results:
        if impact.sample_size >= 5:
            print(impact)
    
    # Summary
    print("\n" + "=" * 70)
    print("📈 ACTIONABLE SIGNALS (statistically significant)")
    print("=" * 70)
    
    significant = [
        (name, impact) for name, impact in results.items()
        if impact.p_value < 0.1 and impact.sample_size >= 10
    ]
    
    if significant:
        for name, impact in sorted(significant, key=lambda x: -abs(x[1].avg_sales_lift)):
            direction = "↑" if impact.avg_sales_lift > 0 else "↓"
            print(f"   {impact.signal_name}: {impact.avg_sales_lift:+.1%} {direction}")
    else:
        print("   No signals reached statistical significance with current data.")
        print("   This might mean:")
        print("   - Signals have minimal real impact")
        print("   - Need more data (run again in a few months)")
        print("   - Signal detection needs refinement")
    
    # Save for predictor use
    print("\n" + "=" * 70)
    print("💾 SAVING LEARNED MAGNITUDES")
    print("=" * 70)
    
    output = learner.save_learned_magnitudes()
    
    # Show suggested predictor weights
    actionable = {k: v for k, v in output.items() if v.get("actionable")}
    if actionable:
        print("\n📊 Suggested predictor weights (for actionable signals):")
        for key, data in sorted(actionable.items(), key=lambda x: -x[1]["predictor_weight"]):
            print(f"   {data['signal_name']}: weight={data['predictor_weight']} (lift={data['lift_pct']})")


if __name__ == "__main__":
    demo()
