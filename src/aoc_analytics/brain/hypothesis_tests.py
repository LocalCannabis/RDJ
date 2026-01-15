"""
Hypothesis Test Implementations

Actual statistical tests for each hypothesis category.
"""

import sqlite3
from datetime import datetime, timedelta, date
from collections import defaultdict
from typing import Dict, List, Tuple, Optional
import math

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class HypothesisTests:
    """Collection of hypothesis test implementations."""
    
    SIGNIFICANCE = 0.05
    MIN_EFFECT = 0.05
    MIN_SAMPLES = 20
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    # ==================== TEMPORAL TESTS ====================
    
    def test_payday_effect(self) -> dict:
        """Test if 1st and 15th (payday) have higher sales."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            GROUP BY date
        """)
        
        payday = []
        other = []
        
        for date_str, units in cur.fetchall():
            day = int(date_str.split("-")[2])
            if day in [1, 2, 15, 16]:  # Payday and day after
                payday.append(units)
            elif day not in [6, 7, 20, 21]:  # Exclude weekends near payday
                other.append(units)
        
        conn.close()
        
        return self._compare_groups(payday, other, "payday", "other days")
    
    def test_day_of_week(self, target_day: int = 1) -> dict:
        """Test if a specific day differs. 0=Sun, 1=Mon, etc."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        day_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        cur.execute("""
            SELECT strftime('%w', date) as dow, SUM(quantity) as units
            FROM sales
            GROUP BY date
        """)
        
        target = []
        other = []
        
        for dow, units in cur.fetchall():
            if int(dow) == target_day:
                target.append(units)
            elif int(dow) not in [0, 6]:  # Compare to other weekdays
                other.append(units)
        
        conn.close()
        
        return self._compare_groups(target, other, day_names[target_day], "other weekdays")
    
    def test_hour_pattern(self, target_hour: int) -> dict:
        """Test if a specific hour has different sales."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(quantity) as units,
                date
            FROM sales
            WHERE time IS NOT NULL
            GROUP BY date, hour
        """)
        
        by_date = defaultdict(lambda: defaultdict(int))
        for hour, units, date_str in cur.fetchall():
            by_date[date_str][hour] = units
        
        conn.close()
        
        target = []
        other = []
        
        for date_str, hours in by_date.items():
            if target_hour in hours:
                target.append(hours[target_hour])
            # Average of nearby hours
            nearby = [hours[h] for h in range(target_hour - 2, target_hour + 3) if h in hours and h != target_hour]
            if nearby:
                other.append(sum(nearby) / len(nearby))
        
        return self._compare_groups(target, other, f"hour {target_hour}", "nearby hours")
    
    def test_last_hour_rush(self) -> dict:
        """Test if last hour (10pm) has higher sales than average."""
        return self.test_hour_pattern(22)
    
    def test_friday_afternoon(self) -> dict:
        """Test if Friday afternoon (3-6pm) is busier."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                date,
                CAST(strftime('%H', time) AS INTEGER) as hour,
                strftime('%w', date) as dow,
                SUM(quantity) as units
            FROM sales
            WHERE time IS NOT NULL
            GROUP BY date, hour
        """)
        
        friday_afternoon = []
        other_afternoon = []
        
        for date_str, hour, dow, units in cur.fetchall():
            if hour in [15, 16, 17, 18]:  # 3-6pm
                if int(dow) == 5:  # Friday
                    friday_afternoon.append(units)
                elif int(dow) in [1, 2, 3, 4]:  # Other weekdays
                    other_afternoon.append(units)
        
        conn.close()
        
        return self._compare_groups(friday_afternoon, other_afternoon, 
                                   "Friday afternoon", "other weekday afternoons")
    
    def test_month_end_budget(self) -> dict:
        """Test if last 3 days of month have lower sales."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            GROUP BY date
        """)
        
        month_end = []
        other = []
        
        for date_str, units in cur.fetchall():
            day = int(date_str.split("-")[2])
            if day >= 28:  # Last few days
                month_end.append(units)
            elif 5 <= day <= 25:  # Middle of month
                other.append(units)
        
        conn.close()
        
        return self._compare_groups(month_end, other, "month end (28-31)", "mid-month")
    
    def test_week_of_month(self) -> dict:
        """Test if first week differs from last week."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            GROUP BY date
        """)
        
        first_week = []
        last_week = []
        
        for date_str, units in cur.fetchall():
            day = int(date_str.split("-")[2])
            if 1 <= day <= 7:
                first_week.append(units)
            elif day >= 22:
                last_week.append(units)
        
        conn.close()
        
        return self._compare_groups(first_week, last_week, "first week", "last week")
    
    # ==================== COSMIC TESTS ====================
    
    def test_moon_phase(self, phase: str = "full") -> dict:
        """Test if moon phase affects sales."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            GROUP BY date
        """)
        
        # Reference full moon: Jan 13, 2025
        reference = date(2025, 1, 13)
        synodic = 29.53
        
        phase_sales = []
        other_sales = []
        
        for date_str, units in cur.fetchall():
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_since = (d - reference).days
            moon_cycle = (days_since % synodic) / synodic
            
            # Full moon: 0 or 1, New moon: 0.5
            if phase == "full":
                is_phase = moon_cycle < 0.07 or moon_cycle > 0.93
            elif phase == "new":
                is_phase = 0.43 < moon_cycle < 0.57
            else:
                is_phase = False
            
            if is_phase:
                phase_sales.append(units)
            else:
                other_sales.append(units)
        
        conn.close()
        
        return self._compare_groups(phase_sales, other_sales, 
                                   f"{phase} moon", "other days")
    
    def test_friday_13th(self) -> dict:
        """Test if Friday the 13th affects sales."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, strftime('%w', date) as dow, SUM(quantity) as units
            FROM sales
            GROUP BY date
        """)
        
        friday_13 = []
        other_friday = []
        
        for date_str, dow, units in cur.fetchall():
            if int(dow) == 5:  # Friday
                day = int(date_str.split("-")[2])
                if day == 13:
                    friday_13.append(units)
                else:
                    other_friday.append(units)
        
        conn.close()
        
        return self._compare_groups(friday_13, other_friday, 
                                   "Friday 13th", "other Fridays")
    
    # ==================== PRODUCT TESTS ====================
    
    def test_category_by_day(self, category: str, target_dow: int) -> dict:
        """Test if category sells more on specific day."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                strftime('%w', date) as dow,
                SUM(quantity) as units
            FROM sales
            WHERE category LIKE ?
            GROUP BY date
        """, (f"%{category}%",))
        
        target = []
        other = []
        
        for dow, units in cur.fetchall():
            if int(dow) == target_dow:
                target.append(units)
            else:
                other.append(units)
        
        conn.close()
        
        day_names = ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat']
        return self._compare_groups(target, other, 
                                   f"{category} on {day_names[target_dow]}", 
                                   f"{category} other days")
    
    def test_flower_friday(self) -> dict:
        """Test if flower sells more on Friday."""
        return self.test_category_by_day("Flower", 5)
    
    def test_preroll_rush_hour(self) -> dict:
        """Test if pre-rolls sell more during rush hours (5-7pm)."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(quantity) as units,
                date
            FROM sales
            WHERE category LIKE '%Pre-Roll%' AND time IS NOT NULL
            GROUP BY date, hour
        """)
        
        rush = []
        other = []
        
        for hour, units, date_str in cur.fetchall():
            if 17 <= hour <= 19:  # 5-7pm
                rush.append(units)
            elif 12 <= hour <= 16:  # Midday for comparison
                other.append(units)
        
        conn.close()
        
        return self._compare_groups(rush, other, "rush hour pre-rolls", "midday pre-rolls")
    
    def test_concentrate_weekend(self) -> dict:
        """Test if concentrates sell more on weekends."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                strftime('%w', date) as dow,
                SUM(quantity) as units
            FROM sales
            WHERE category LIKE '%Concentrate%' 
               OR category LIKE '%Rosin%'
               OR category LIKE '%Resin%'
               OR category LIKE '%Hash%'
               OR category LIKE '%Shatter%'
            GROUP BY date
        """)
        
        weekend = []
        weekday = []
        
        for dow, units in cur.fetchall():
            if int(dow) in [0, 6]:  # Sun, Sat
                weekend.append(units)
            else:
                weekday.append(units)
        
        conn.close()
        
        return self._compare_groups(weekend, weekday, "weekend concentrates", "weekday concentrates")
    
    def test_edibles_rain(self) -> dict:
        """Test if edibles sell more on rainy days (needs weather data)."""
        # Would need weather integration
        return {
            "tested": False,
            "notes": "Requires weather data integration"
        }
    
    def test_beverage_temperature(self) -> dict:
        """Test if beverages correlate with temperature."""
        return {
            "tested": False,
            "notes": "Requires weather data integration"
        }
    
    # ==================== BEHAVIORAL TESTS ====================
    
    def test_basket_size_friday(self) -> dict:
        """Test if Friday has larger basket sizes."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT 
                strftime('%w', date) as dow,
                invoice_id,
                SUM(quantity) as basket_items
            FROM sales
            WHERE invoice_id IS NOT NULL
            GROUP BY invoice_id
        """)
        
        friday_baskets = []
        other_baskets = []
        
        for dow, invoice, items in cur.fetchall():
            if dow and int(dow) == 5:
                friday_baskets.append(items)
            elif dow and int(dow) in [1, 2, 3, 4]:
                other_baskets.append(items)
        
        conn.close()
        
        return self._compare_groups(friday_baskets, other_baskets,
                                   "Friday basket size", "weekday basket size")
    
    def test_premium_payday(self) -> dict:
        """Test if premium products sell more around payday."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Premium = higher price products
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            WHERE unit_price >= 50  -- Premium threshold
            GROUP BY date
        """)
        
        payday = []
        other = []
        
        for date_str, units in cur.fetchall():
            day = int(date_str.split("-")[2])
            if day in [1, 2, 15, 16]:
                payday.append(units)
            elif 5 <= day <= 12 or 18 <= day <= 25:
                other.append(units)
        
        conn.close()
        
        return self._compare_groups(payday, other, "payday premium", "mid-period premium")
    
    def test_budget_month_end(self) -> dict:
        """Test if budget products sell more at month end."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            WHERE unit_price < 25  -- Budget threshold
            GROUP BY date
        """)
        
        month_end = []
        mid_month = []
        
        for date_str, units in cur.fetchall():
            day = int(date_str.split("-")[2])
            if day >= 25:
                month_end.append(units)
            elif 5 <= day <= 20:
                mid_month.append(units)
        
        conn.close()
        
        return self._compare_groups(month_end, mid_month, 
                                   "month-end budget", "mid-month budget")
    
    # ==================== CATEGORY CORRELATIONS ====================
    
    def test_category_correlation(self, cat1: str, cat2: str) -> dict:
        """Test if two categories' daily sales correlate."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, category, SUM(quantity) as units
            FROM sales
            GROUP BY date, category
        """)
        
        by_date = defaultdict(dict)
        for date_str, category, units in cur.fetchall():
            by_date[date_str][category] = units
        
        conn.close()
        
        # Find matching categories
        all_cats = set()
        for cats in by_date.values():
            all_cats.update(cats.keys())
        
        match1 = next((c for c in all_cats if cat1.lower() in c.lower()), None)
        match2 = next((c for c in all_cats if cat2.lower() in c.lower()), None)
        
        if not match1 or not match2:
            return {"tested": False, "notes": f"Categories not found: {cat1}, {cat2}"}
        
        # Build paired data
        series1 = []
        series2 = []
        
        for date_str in by_date:
            if match1 in by_date[date_str] and match2 in by_date[date_str]:
                series1.append(by_date[date_str][match1])
                series2.append(by_date[date_str][match2])
        
        if len(series1) < self.MIN_SAMPLES:
            return {"tested": True, "notes": f"Insufficient samples: {len(series1)}"}
        
        if HAS_SCIPY:
            corr, p_value = stats.pearsonr(series1, series2)
        else:
            # Manual correlation
            n = len(series1)
            mean1, mean2 = sum(series1)/n, sum(series2)/n
            num = sum((x - mean1) * (y - mean2) for x, y in zip(series1, series2))
            den1 = sum((x - mean1)**2 for x in series1) ** 0.5
            den2 = sum((y - mean2)**2 for y in series2) ** 0.5
            corr = num / (den1 * den2) if den1 * den2 > 0 else 0
            p_value = 0.5  # Placeholder
        
        proven = p_value < self.SIGNIFICANCE and abs(corr) > 0.3
        
        return {
            "tested": True,
            "proven": proven,
            "effect_size": corr,
            "p_value": p_value,
            "sample_size": len(series1),
            "notes": f"{match1} vs {match2}: r={corr:.3f}, p={p_value:.3f}"
        }
    
    def test_accessory_flower_correlation(self) -> dict:
        """Test if accessory sales correlate with flower spikes."""
        return self.test_category_correlation("Accessories", "Flower")
    
    # ==================== HELPER METHODS ====================
    
    def _compare_groups(self, group1: List[float], group2: List[float],
                       name1: str, name2: str) -> dict:
        """Compare two groups statistically."""
        if len(group1) < self.MIN_SAMPLES or len(group2) < self.MIN_SAMPLES:
            return {
                "tested": True,
                "proven": False,
                "notes": f"Insufficient samples: {len(group1)} vs {len(group2)}"
            }
        
        mean1 = sum(group1) / len(group1)
        mean2 = sum(group2) / len(group2)
        effect = (mean1 - mean2) / mean2 if mean2 > 0 else 0
        
        if HAS_SCIPY:
            t_stat, p_value = stats.ttest_ind(group1, group2)
        else:
            # Simple approximation
            p_value = 0.5
        
        proven = p_value < self.SIGNIFICANCE and abs(effect) > self.MIN_EFFECT
        
        return {
            "tested": True,
            "proven": proven,
            "effect_size": effect,
            "p_value": p_value,
            "sample_size": len(group1) + len(group2),
            "mean_group1": mean1,
            "mean_group2": mean2,
            "notes": f"{name1}: {mean1:.1f} vs {name2}: {mean2:.1f} ({effect:+.1%})"
        }
    
    def run_all_tests(self) -> Dict[str, dict]:
        """Run all implemented tests."""
        tests = {
            # Temporal
            "payday_effect": self.test_payday_effect,
            "monday_effect": lambda: self.test_day_of_week(1),
            "tuesday_effect": lambda: self.test_day_of_week(2),
            "wednesday_effect": lambda: self.test_day_of_week(3),
            "thursday_effect": lambda: self.test_day_of_week(4),
            "last_hour_rush": self.test_last_hour_rush,
            "friday_afternoon": self.test_friday_afternoon,
            "month_end_effect": self.test_month_end_budget,
            "week_of_month": self.test_week_of_month,
            
            # Cosmic
            "full_moon": lambda: self.test_moon_phase("full"),
            "new_moon": lambda: self.test_moon_phase("new"),
            "friday_13th": self.test_friday_13th,
            
            # Product
            "flower_friday": self.test_flower_friday,
            "preroll_rush": self.test_preroll_rush_hour,
            "concentrate_weekend": self.test_concentrate_weekend,
            
            # Behavioral
            "basket_friday": self.test_basket_size_friday,
            "premium_payday": self.test_premium_payday,
            "budget_month_end": self.test_budget_month_end,
            
            # Correlations
            "accessory_flower": self.test_accessory_flower_correlation,
        }
        
        results = {}
        for name, test_fn in tests.items():
            try:
                results[name] = test_fn()
            except Exception as e:
                results[name] = {"tested": False, "error": str(e)}
        
        return results


if __name__ == "__main__":
    from pathlib import Path
    
    db_path = str(Path.cwd() / "aoc_sales.db")
    tests = HypothesisTests(db_path)
    
    print("=" * 70)
    print("🔬 HYPOTHESIS TEST SUITE")
    print("=" * 70)
    
    results = tests.run_all_tests()
    
    proven = []
    disproven = []
    
    for name, result in results.items():
        if result.get("proven"):
            proven.append((name, result))
        elif result.get("tested"):
            disproven.append((name, result))
    
    print(f"\n✅ PROVEN HYPOTHESES ({len(proven)}):")
    print("-" * 70)
    for name, r in sorted(proven, key=lambda x: abs(x[1].get("effect_size", 0)), reverse=True):
        print(f"  {name}")
        print(f"    Effect: {r['effect_size']:+.1%}, p={r['p_value']:.4f}")
        print(f"    {r.get('notes', '')}")
        print()
    
    print(f"\n❌ DISPROVEN/INCONCLUSIVE ({len(disproven)}):")
    print("-" * 70)
    for name, r in disproven[:10]:  # Show first 10
        print(f"  {name}: {r.get('notes', 'No details')}")
