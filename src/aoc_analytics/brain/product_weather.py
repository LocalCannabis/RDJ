"""
Product-Level Weather Impact Learning

Learns which product CATEGORIES respond differently to weather:
- Do edibles sell better on rainy days (indoor vibe)?
- Do pre-rolls spike on nice days (outdoor vibe)?
- Do accessories/lighters spike before rain (preparedness)?

This extends signal_magnitude.py to be product-aware.
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

# Import scipy for statistics
try:
    from scipy import stats
except ImportError:
    stats = None


@dataclass
class CategoryWeatherImpact:
    """Impact of weather on a specific category."""
    category: str
    weather_condition: str  # "rain", "sunny", "cold", "hot"
    avg_lift: float
    std_dev: float
    sample_size: int
    p_value: float
    confidence: float
    
    def __str__(self) -> str:
        direction = "📈" if self.avg_lift > 0 else "📉"
        return (
            f"{direction} {self.category} on {self.weather_condition} days\n"
            f"   Sales lift: {self.avg_lift:+.1%} ± {self.std_dev:.1%}\n"
            f"   Sample: {self.sample_size} days, Confidence: {self.confidence:.0%}"
        )


class ProductWeatherLearner:
    """
    Learn how different product categories respond to weather.
    
    Uses weather_daily table joined with sales data.
    """
    
    # Main categories to analyze
    CATEGORIES = [
        "Flower",
        "Pre-Rolls", 
        "Vapes",
        "Edibles",
        "Beverages",
        "Concentrates",
        "Topicals",
        "Accessories",
        "Hash",
        "Capsules",
    ]
    
    # Weather conditions to test
    WEATHER_CONDITIONS = {
        "rainy": {"precip_min": 5.0},  # > 5mm rain
        "dry": {"precip_max": 0.5},    # < 0.5mm
        "cold": {"temp_max": 5.0},     # < 5°C
        "mild": {"temp_min": 10.0, "temp_max": 20.0},  # 10-20°C
        "warm": {"temp_min": 20.0},    # > 20°C
    }
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Search for database in common locations
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
        
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def _get_category_daily_sales(self, category: str, location: str = None, days: int = 365) -> Dict[str, Dict]:
        """Get daily sales for a specific category."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        query = """
            SELECT 
                DATE(date) as sale_day,
                strftime('%w', date) as dow,
                SUM(quantity) as total_qty,
                SUM(subtotal) as total_rev
            FROM sales
            WHERE category = ?
              AND date >= ?
        """
        params = [category, cutoff_str]
        
        if location:
            query += " AND location = ?"
            params.append(location)
        
        query += " GROUP BY sale_day ORDER BY sale_day"
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            result[row[0]] = {
                "dow": int(row[1]),
                "qty": row[2] or 0,
                "rev": row[3] or 0.0,
            }
        
        return result
    
    def _get_weather_data(self, days: int = 365) -> Dict[str, Dict]:
        """Get weather data indexed by date."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT 
                date,
                temp_max_c,
                temp_min_c,
                temp_avg_c,
                precip_mm,
                condition
            FROM weather_daily
            WHERE date >= ?
            ORDER BY date
        """, (cutoff_str,))
        
        rows = cursor.fetchall()
        conn.close()
        
        result = {}
        for row in rows:
            temp_avg = row[3] if row[3] else ((row[1] + row[2]) / 2 if row[1] and row[2] else 15.0)
            result[row[0]] = {
                "temp_high": row[1] or 15.0,
                "temp_low": row[2] or 10.0,
                "temp_avg": temp_avg,
                "precip": row[4] or 0.0,
                "conditions": row[5] or "",
            }
        
        return result
    
    def _classify_weather_day(self, weather: Dict) -> List[str]:
        """Classify a weather day into conditions."""
        
        conditions = []
        
        # Rain check
        if weather["precip"] >= 5.0:
            conditions.append("rainy")
        elif weather["precip"] <= 0.5:
            conditions.append("dry")
        
        # Temperature check
        if weather["temp_avg"] <= 5.0:
            conditions.append("cold")
        elif 10.0 <= weather["temp_avg"] <= 20.0:
            conditions.append("mild")
        elif weather["temp_avg"] >= 20.0:
            conditions.append("warm")
        
        return conditions
    
    def _calculate_category_baseline(self, daily_data: Dict[str, Dict]) -> Dict[int, float]:
        """Calculate baseline sales by day of week for a category."""
        
        dow_totals = defaultdict(list)
        for date_str, data in daily_data.items():
            dow_totals[data["dow"]].append(data["qty"])
        
        baselines = {}
        for dow, values in dow_totals.items():
            baselines[dow] = np.mean(values) if values else 0
        
        return baselines
    
    def learn_category_weather_impact(
        self, 
        category: str,
        weather_condition: str,
        location: str = None
    ) -> CategoryWeatherImpact:
        """Learn how a category responds to a specific weather condition."""
        
        if stats is None:
            return CategoryWeatherImpact(
                category=category,
                weather_condition=weather_condition,
                avg_lift=0,
                std_dev=0,
                sample_size=0,
                p_value=1.0,
                confidence=0,
            )
        
        # Get data
        daily_sales = self._get_category_daily_sales(category, location)
        weather_data = self._get_weather_data()
        baselines = self._calculate_category_baseline(daily_sales)
        
        condition_days = []
        normal_days = []
        
        for date_str, sales in daily_sales.items():
            weather = weather_data.get(date_str)
            if not weather:
                continue
            
            baseline = baselines.get(sales["dow"], np.mean(list(baselines.values())))
            if baseline <= 0:
                continue
            
            lift = (sales["qty"] - baseline) / baseline
            
            # Classify this day's weather
            day_conditions = self._classify_weather_day(weather)
            
            if weather_condition in day_conditions:
                condition_days.append(lift)
            else:
                normal_days.append(lift)
        
        if len(condition_days) < 5 or len(normal_days) < 10:
            return CategoryWeatherImpact(
                category=category,
                weather_condition=weather_condition,
                avg_lift=0,
                std_dev=0,
                sample_size=len(condition_days),
                p_value=1.0,
                confidence=0,
            )
        
        avg_lift = float(np.mean(condition_days))
        std_dev = float(np.std(condition_days))
        
        # T-test
        t_stat, p_value = stats.ttest_ind(condition_days, normal_days)
        confidence = 1 - p_value if p_value < 0.5 else 0.5
        
        return CategoryWeatherImpact(
            category=category,
            weather_condition=weather_condition,
            avg_lift=avg_lift,
            std_dev=std_dev,
            sample_size=len(condition_days),
            p_value=float(p_value),
            confidence=float(confidence),
        )
    
    def learn_all_category_weather_impacts(
        self,
        location: str = None
    ) -> Dict[str, Dict[str, CategoryWeatherImpact]]:
        """Learn weather impact for all category/condition combinations."""
        
        results = {}
        
        for category in self.CATEGORIES:
            print(f"  Analyzing {category}...")
            results[category] = {}
            
            for condition in self.WEATHER_CONDITIONS.keys():
                impact = self.learn_category_weather_impact(category, condition, location)
                results[category][condition] = impact
        
        return results
    
    def get_actionable_insights(
        self,
        results: Dict[str, Dict[str, CategoryWeatherImpact]],
        p_threshold: float = 0.1
    ) -> List[Tuple[str, CategoryWeatherImpact]]:
        """Extract statistically significant insights."""
        
        actionable = []
        
        for category, conditions in results.items():
            for condition_name, impact in conditions.items():
                if impact.p_value < p_threshold and impact.sample_size >= 10:
                    actionable.append((f"{category}/{condition_name}", impact))
        
        # Sort by absolute impact
        actionable.sort(key=lambda x: abs(x[1].avg_lift), reverse=True)
        
        return actionable
    
    def save_results(self, results: Dict[str, Dict[str, CategoryWeatherImpact]]) -> str:
        """Save results to JSON file."""
        import json
        
        output = {
            "generated": str(datetime.now()),
            "categories": {}
        }
        
        for category, conditions in results.items():
            output["categories"][category] = {}
            for condition_name, impact in conditions.items():
                output["categories"][category][condition_name] = {
                    "lift": float(round(impact.avg_lift, 4)),
                    "lift_pct": f"{impact.avg_lift:+.1%}",
                    "std_dev": float(round(impact.std_dev, 4)),
                    "sample_size": int(impact.sample_size),
                    "p_value": float(round(impact.p_value, 4)),
                    "confidence": float(round(impact.confidence, 2)),
                    "actionable": bool(impact.p_value < 0.1 and impact.sample_size >= 10),
                }
        
        # Save
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "category_weather_impacts.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate category-weather learning."""
    
    print("=" * 70)
    print("🌦️  PRODUCT-LEVEL WEATHER IMPACT LEARNING")
    print("   Which categories respond to weather differently?")
    print("=" * 70)
    print()
    
    learner = ProductWeatherLearner()
    
    print("Learning category-weather impacts...")
    results = learner.learn_all_category_weather_impacts()
    
    print("\n" + "=" * 70)
    print("📊 ACTIONABLE CATEGORY-WEATHER INSIGHTS")
    print("   (statistically significant, p < 0.1)")
    print("=" * 70 + "\n")
    
    actionable = learner.get_actionable_insights(results)
    
    if actionable:
        for name, impact in actionable[:15]:  # Top 15
            direction = "☀️" if impact.avg_lift > 0 else "🌧️"
            print(f"{direction} {impact.category} on {impact.weather_condition} days: "
                  f"{impact.avg_lift:+.1%} (p={impact.p_value:.3f}, n={impact.sample_size})")
    else:
        print("No statistically significant category-weather relationships found.")
        print("This might mean:")
        print("  - Weather doesn't strongly affect category mix")
        print("  - Need more data (try running with more history)")
    
    # Save results
    output_file = learner.save_results(results)
    print(f"\n💾 Saved to: {output_file}")
    
    # Category summary
    print("\n" + "=" * 70)
    print("📋 CATEGORY WEATHER SENSITIVITY SUMMARY")
    print("=" * 70 + "\n")
    
    for category in learner.CATEGORIES:
        cat_results = results.get(category, {})
        significant = [
            (cond, impact) for cond, impact in cat_results.items()
            if impact.p_value < 0.2 and impact.sample_size >= 5
        ]
        
        if significant:
            print(f"\n{category}:")
            for cond, impact in sorted(significant, key=lambda x: x[1].p_value):
                marker = "✓" if impact.p_value < 0.1 else "~"
                print(f"  {marker} {cond}: {impact.avg_lift:+.1%} (p={impact.p_value:.2f})")


if __name__ == "__main__":
    demo()
