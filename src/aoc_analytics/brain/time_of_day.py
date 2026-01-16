"""
Time-of-Day Pattern Learning

Learn hourly sales patterns:
- Morning rush (9-10am?)
- Lunch bump (12-1pm?)
- After-work spike (5-7pm?)
- Evening peak (8-10pm?)
- Late night (10pm-close?)

Also learns day-of-week x hour interactions:
- Friday evening vs Monday evening
- Weekend afternoons vs weekday

Store Hours: 9am - 11pm (hours 9-22)
Orders outside store hours are online orders and not significant for staffing.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

from aoc_analytics.core.db_adapter import get_connection as db_get_connection

try:
    from scipy import stats
except ImportError:
    stats = None


# Store operating hours
STORE_OPEN_HOUR = 9   # 9am
STORE_CLOSE_HOUR = 23  # 11pm (hour 22 is 10pm-11pm, so we include up to 22)

def is_store_hours(hour: int) -> bool:
    """Check if an hour falls within store operating hours (9am-11pm)."""
    return STORE_OPEN_HOUR <= hour < STORE_CLOSE_HOUR


@dataclass
class HourlyPattern:
    """Sales pattern for a specific hour."""
    hour: int
    avg_units: float
    avg_revenue: float
    std_dev: float
    pct_of_daily: float  # What % of daily sales happen in this hour
    peak_rank: int  # 1 = highest hour, 2 = second, etc.
    
    def __str__(self) -> str:
        bar_len = int(self.pct_of_daily * 50)  # Scale to ~50 chars
        bar = "█" * bar_len
        return f"{self.hour:02d}:00 │ {bar} {self.pct_of_daily:.1%} ({self.avg_units:.1f} units)"


@dataclass
class DayHourPattern:
    """Sales pattern for a specific day-hour combination."""
    day_name: str  # "Monday", "Tuesday", etc.
    hour: int
    avg_units: float
    lift_vs_baseline: float  # vs same-hour average across all days
    sample_size: int


class TimeOfDayLearner:
    """
    Learn when sales happen throughout the day.
    
    Note: Store hours are 9am-11pm. Orders outside these hours
    are online orders and excluded from staffing analysis.
    """
    
    DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
    # Time periods for analysis (within store hours 9am-11pm)
    TIME_PERIODS = {
        "morning_open": (9, 11),    # 9am-11am (opening rush)
        "midday": (11, 14),         # 11am-2pm (lunch)
        "afternoon": (14, 17),      # 2pm-5pm (slow period)
        "evening_rush": (17, 20),   # 5pm-8pm (after-work peak)
        "night": (20, 23),          # 8pm-11pm (evening until close)
    }
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            possible_paths = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent.parent / "aoc_sales.db",
                Path.home() / "Projects" / "aoc-analytics" / "aoc_sales.db",
            ]
            for p in possible_paths:
                if p.exists() and p.stat().st_size > 0:
                    db_path = str(p)
                    break
        self.db_path = str(db_path) if db_path else None
        
    def get_connection(self):
        return db_get_connection(self.db_path)
    
    def _get_hourly_sales(self, location: str = None, days: int = 365) -> Dict[int, List[float]]:
        """Get sales by hour across all days (store hours only: 9am-11pm)."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        # Only include store hours (9am-11pm = hours 9-22)
        query = """
            SELECT 
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(quantity) as qty,
                SUM(subtotal) as rev,
                date
            FROM sales
            WHERE date >= ?
              AND time IS NOT NULL
              AND CAST(strftime('%H', time) AS INTEGER) >= ?
              AND CAST(strftime('%H', time) AS INTEGER) < ?
        """
        params = [cutoff_str, STORE_OPEN_HOUR, STORE_CLOSE_HOUR]
        
        if location:
            query += " AND location = ?"
            params.append(location)
        
        query += " GROUP BY date, hour ORDER BY date, hour"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Group by hour
        hourly = defaultdict(list)
        for row in rows:
            hour = row[0]
            qty = row[1] or 0
            hourly[hour].append(qty)
        
        return dict(hourly)
    
    def _get_day_hour_sales(self, location: str = None, days: int = 365) -> Dict[Tuple[int, int], List[float]]:
        """Get sales by (day_of_week, hour) combination (store hours only: 9am-11pm)."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        # Only include store hours (9am-11pm = hours 9-22)
        query = """
            SELECT 
                CAST(strftime('%w', date) AS INTEGER) as dow,
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(quantity) as qty,
                date
            FROM sales
            WHERE date >= ?
              AND time IS NOT NULL
              AND CAST(strftime('%H', time) AS INTEGER) >= ?
              AND CAST(strftime('%H', time) AS INTEGER) < ?
        """
        params = [cutoff_str, STORE_OPEN_HOUR, STORE_CLOSE_HOUR]
        
        if location:
            query += " AND location = ?"
            params.append(location)
        
        query += " GROUP BY date, dow, hour ORDER BY date"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        # Group by (dow, hour)
        day_hour = defaultdict(list)
        for row in rows:
            dow = row[0]
            hour = row[1]
            qty = row[2] or 0
            day_hour[(dow, hour)].append(qty)
        
        return dict(day_hour)
    
    def learn_hourly_patterns(self, location: str = None) -> List[HourlyPattern]:
        """Learn the basic hourly sales pattern (store hours only: 9am-11pm)."""
        
        hourly_data = self._get_hourly_sales(location)
        
        # Calculate stats for each hour (store hours only)
        patterns = []
        total_avg = sum(np.mean(v) for v in hourly_data.values())
        
        for hour in range(STORE_OPEN_HOUR, STORE_CLOSE_HOUR):
            if hour not in hourly_data:
                continue
            
            values = hourly_data[hour]
            avg_units = np.mean(values)
            std_dev = np.std(values)
            pct_of_daily = avg_units / total_avg if total_avg > 0 else 0
            
            patterns.append(HourlyPattern(
                hour=hour,
                avg_units=float(avg_units),
                avg_revenue=0,  # TODO: calculate from revenue data
                std_dev=float(std_dev),
                pct_of_daily=float(pct_of_daily),
                peak_rank=0,  # Set below
            ))
        
        # Set peak ranks
        patterns.sort(key=lambda x: x.avg_units, reverse=True)
        for i, p in enumerate(patterns):
            p.peak_rank = i + 1
        
        # Sort by hour for display
        patterns.sort(key=lambda x: x.hour)
        
        return patterns
    
    def learn_day_hour_interactions(self, location: str = None) -> Dict[str, List[DayHourPattern]]:
        """Learn how hourly patterns differ by day of week."""
        
        day_hour_data = self._get_day_hour_sales(location)
        hourly_data = self._get_hourly_sales(location)
        
        # Calculate baseline for each hour
        hour_baselines = {}
        for hour in range(24):
            if hour in hourly_data:
                hour_baselines[hour] = np.mean(hourly_data[hour])
            else:
                hour_baselines[hour] = 0
        
        # Calculate day-hour patterns
        results = {day: [] for day in self.DAY_NAMES}
        
        for dow in range(7):
            day_name = self.DAY_NAMES[dow]
            
            for hour in range(24):
                key = (dow, hour)
                if key not in day_hour_data:
                    continue
                
                values = day_hour_data[key]
                avg_units = np.mean(values)
                baseline = hour_baselines.get(hour, 0)
                
                if baseline > 0:
                    lift = (avg_units - baseline) / baseline
                else:
                    lift = 0
                
                results[day_name].append(DayHourPattern(
                    day_name=day_name,
                    hour=hour,
                    avg_units=float(avg_units),
                    lift_vs_baseline=float(lift),
                    sample_size=len(values),
                ))
        
        return results
    
    def find_peak_hours(self, patterns: List[HourlyPattern], top_n: int = 5) -> List[HourlyPattern]:
        """Find the highest-volume hours."""
        return sorted(patterns, key=lambda x: x.avg_units, reverse=True)[:top_n]
    
    def find_rush_hours(self, day_hour: Dict[str, List[DayHourPattern]]) -> Dict[str, List[Tuple[int, float]]]:
        """Find rush hours for each day (hours with significant lift)."""
        
        rush_hours = {}
        
        for day_name, patterns in day_hour.items():
            significant = [
                (p.hour, p.lift_vs_baseline) 
                for p in patterns 
                if p.lift_vs_baseline > 0.1 and p.sample_size >= 10
            ]
            rush_hours[day_name] = sorted(significant, key=lambda x: x[1], reverse=True)
        
        return rush_hours
    
    def get_staffing_recommendations(self, patterns: List[HourlyPattern]) -> Dict[str, List[int]]:
        """Generate staffing recommendations based on patterns."""
        
        # Categorize hours by volume
        sorted_patterns = sorted(patterns, key=lambda x: x.avg_units, reverse=True)
        n = len(sorted_patterns)
        
        high_vol = [p.hour for p in sorted_patterns[:n//3]]
        med_vol = [p.hour for p in sorted_patterns[n//3:2*n//3]]
        low_vol = [p.hour for p in sorted_patterns[2*n//3:]]
        
        return {
            "peak_staffing": sorted(high_vol),
            "normal_staffing": sorted(med_vol),
            "reduced_staffing": sorted(low_vol),
        }
    
    def save_patterns(
        self, 
        hourly: List[HourlyPattern], 
        day_hour: Dict[str, List[DayHourPattern]]
    ) -> str:
        """Save patterns to JSON."""
        import json
        
        output = {
            "generated": str(datetime.now()),
            "store_hours": {
                "open": STORE_OPEN_HOUR,
                "close": STORE_CLOSE_HOUR,
                "note": "Only in-store hours analyzed. After-hours orders are online."
            },
            "hourly_patterns": [
                {
                    "hour": p.hour,
                    "avg_units": round(p.avg_units, 2),
                    "pct_of_daily": round(p.pct_of_daily, 4),
                    "peak_rank": p.peak_rank,
                }
                for p in hourly
            ],
            "day_hour_interactions": {
                day: [
                    {
                        "hour": p.hour,
                        "avg_units": round(p.avg_units, 2),
                        "lift_vs_baseline": round(p.lift_vs_baseline, 4),
                    }
                    for p in patterns
                ]
                for day, patterns in day_hour.items()
            }
        }
        
        # Save
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "time_of_day_patterns.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate time-of-day learning."""
    
    print("=" * 70)
    print("🕐 TIME-OF-DAY PATTERN LEARNING")
    print("   When do sales peak throughout the day?")
    print(f"   Store hours: {STORE_OPEN_HOUR}am - {STORE_CLOSE_HOUR % 12 or 12}pm")
    print("   (After-hours orders are online - excluded from analysis)")
    print("=" * 70)
    print()
    
    learner = TimeOfDayLearner()
    
    # Learn hourly patterns
    print("Learning hourly patterns...")
    hourly = learner.learn_hourly_patterns()
    
    if not hourly:
        print("No hourly data found. Make sure sales have time data.")
        return
    
    print("\n📊 HOURLY SALES DISTRIBUTION")
    print("-" * 60)
    
    for pattern in hourly:
        print(pattern)
    
    # Find peak hours
    peaks = learner.find_peak_hours(hourly, top_n=5)
    print(f"\n🔥 TOP 5 PEAK HOURS:")
    for i, p in enumerate(peaks, 1):
        print(f"   {i}. {p.hour:02d}:00 - {p.avg_units:.1f} units/day ({p.pct_of_daily:.1%} of daily)")
    
    # Learn day-hour interactions
    print("\n\nLearning day-hour interactions...")
    day_hour = learner.learn_day_hour_interactions()
    
    # Find rush hours by day
    rush_hours = learner.find_rush_hours(day_hour)
    
    print("\n📅 RUSH HOURS BY DAY (>10% above baseline)")
    print("-" * 60)
    
    for day in learner.DAY_NAMES:
        rushes = rush_hours.get(day, [])
        if rushes:
            hours_str = ", ".join([f"{h}:00 (+{lift:.0%})" for h, lift in rushes[:3]])
            print(f"   {day:9s}: {hours_str}")
        else:
            print(f"   {day:9s}: No significant rush periods")
    
    # Staffing recommendations
    staffing = learner.get_staffing_recommendations(hourly)
    
    print("\n\n👥 STAFFING RECOMMENDATIONS")
    print("-" * 60)
    print(f"   🔴 Peak staffing (busy):    {staffing['peak_staffing']}")
    print(f"   🟡 Normal staffing:         {staffing['normal_staffing']}")
    print(f"   🟢 Reduced staffing (slow): {staffing['reduced_staffing']}")
    
    # Friday vs Monday evening comparison
    print("\n\n📈 FRIDAY vs MONDAY EVENING (17:00-21:00)")
    print("-" * 60)
    
    for day in ["Friday", "Monday"]:
        patterns = day_hour.get(day, [])
        evening = [p for p in patterns if 17 <= p.hour <= 21]
        if evening:
            avg_lift = np.mean([p.lift_vs_baseline for p in evening])
            avg_units = np.mean([p.avg_units for p in evening])
            direction = "📈" if avg_lift > 0 else "📉"
            print(f"   {day:9s} evening: {avg_units:.1f} units/hr ({avg_lift:+.1%} vs baseline) {direction}")
    
    # Save results
    output_file = learner.save_patterns(hourly, day_hour)
    print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    demo()
