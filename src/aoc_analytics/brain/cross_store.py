"""
Cross-Store Learning

Analyzes patterns across all stores to determine:
1. Which patterns are universal (apply everywhere)
2. Which are location-specific (e.g., cruise ships only affect Kingsway)
3. Where knowledge can be transferred between stores

This is crucial for:
- New store openings (transfer knowledge from similar stores)
- Inventory allocation (different category mix by location)
- Staffing optimization (location-specific busy times)
"""

import sqlite3
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
import numpy as np

try:
    from scipy import stats
except ImportError:
    stats = None


@dataclass
class StoreProfile:
    """Profile of a single store's characteristics."""
    location: str
    avg_daily_units: float
    avg_daily_revenue: float
    top_categories: List[Tuple[str, float]]  # (category, pct_of_sales)
    peak_hours: List[int]
    busiest_day: str
    slowest_day: str
    
    def __str__(self) -> str:
        cats = ", ".join([f"{c[0]} ({c[1]:.0%})" for c in self.top_categories[:3]])
        return (
            f"📍 {self.location}\n"
            f"   Daily: {self.avg_daily_units:.0f} units, ${self.avg_daily_revenue:.0f} revenue\n"
            f"   Top categories: {cats}\n"
            f"   Peak hours: {self.peak_hours[:3]}\n"
            f"   Busiest day: {self.busiest_day}, Slowest: {self.slowest_day}"
        )


@dataclass
class PatternComparison:
    """Comparison of a pattern across stores."""
    pattern_name: str
    store_impacts: Dict[str, float]  # {location: impact}
    is_universal: bool  # True if consistent across stores
    variance: float  # How much stores differ
    strongest_store: str
    weakest_store: str


class CrossStoreLearner:
    """
    Learn and compare patterns across all store locations.
    """
    
    DAY_NAMES = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    
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
        
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def get_all_locations(self) -> List[str]:
        """Get all store locations in the database."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("SELECT DISTINCT location FROM sales WHERE location IS NOT NULL")
        locations = [row[0] for row in cursor.fetchall()]
        conn.close()
        
        return sorted(locations)
    
    def build_store_profile(self, location: str, days: int = 365) -> StoreProfile:
        """Build a profile for a single store."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        # Daily averages
        cursor.execute("""
            SELECT 
                AVG(daily_qty) as avg_qty,
                AVG(daily_rev) as avg_rev
            FROM (
                SELECT 
                    date,
                    SUM(quantity) as daily_qty,
                    SUM(subtotal) as daily_rev
                FROM sales
                WHERE location = ? AND date >= ?
                GROUP BY date
            )
        """, (location, cutoff_str))
        
        row = cursor.fetchone()
        avg_units = row[0] or 0
        avg_revenue = row[1] or 0
        
        # Top categories
        cursor.execute("""
            SELECT 
                category,
                SUM(quantity) as cat_qty
            FROM sales
            WHERE location = ? AND date >= ?
            GROUP BY category
            ORDER BY cat_qty DESC
        """, (location, cutoff_str))
        
        cat_rows = cursor.fetchall()
        total_qty = sum(r[1] for r in cat_rows)
        top_categories = [
            (r[0], r[1] / total_qty if total_qty > 0 else 0)
            for r in cat_rows[:5]
        ]
        
        # Peak hours
        cursor.execute("""
            SELECT 
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(quantity) as hour_qty
            FROM sales
            WHERE location = ? AND date >= ? AND time IS NOT NULL
            GROUP BY hour
            ORDER BY hour_qty DESC
            LIMIT 5
        """, (location, cutoff_str))
        
        peak_hours = [row[0] for row in cursor.fetchall()]
        
        # Busiest/slowest day
        cursor.execute("""
            SELECT 
                CAST(strftime('%w', date) AS INTEGER) as dow,
                AVG(day_qty) as avg_day_qty
            FROM (
                SELECT date, SUM(quantity) as day_qty
                FROM sales
                WHERE location = ? AND date >= ?
                GROUP BY date
            )
            GROUP BY dow
            ORDER BY avg_day_qty DESC
        """, (location, cutoff_str))
        
        dow_rows = cursor.fetchall()
        busiest_day = self.DAY_NAMES[dow_rows[0][0]] if dow_rows else "Unknown"
        slowest_day = self.DAY_NAMES[dow_rows[-1][0]] if dow_rows else "Unknown"
        
        conn.close()
        
        return StoreProfile(
            location=location,
            avg_daily_units=float(avg_units),
            avg_daily_revenue=float(avg_revenue),
            top_categories=top_categories,
            peak_hours=peak_hours,
            busiest_day=busiest_day,
            slowest_day=slowest_day,
        )
    
    def compare_hourly_patterns(self) -> Dict[str, List[Tuple[int, float]]]:
        """Compare hourly patterns across stores."""
        
        locations = self.get_all_locations()
        result = {}
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=365)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        for location in locations:
            cursor.execute("""
                SELECT 
                    CAST(strftime('%H', time) AS INTEGER) as hour,
                    SUM(quantity) as hour_qty
                FROM sales
                WHERE location = ? AND date >= ? AND time IS NOT NULL
                GROUP BY hour
            """, (location, cutoff_str))
            
            rows = cursor.fetchall()
            total = sum(r[1] for r in rows)
            
            result[location] = [
                (r[0], r[1] / total if total > 0 else 0)
                for r in sorted(rows, key=lambda x: x[0])
            ]
        
        conn.close()
        return result
    
    def compare_category_mix(self) -> Dict[str, Dict[str, float]]:
        """Compare category mix across stores."""
        
        locations = self.get_all_locations()
        result = {}
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=365)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        for location in locations:
            cursor.execute("""
                SELECT 
                    category,
                    SUM(quantity) as cat_qty
                FROM sales
                WHERE location = ? AND date >= ?
                GROUP BY category
            """, (location, cutoff_str))
            
            rows = cursor.fetchall()
            total = sum(r[1] for r in rows)
            
            result[location] = {
                r[0]: r[1] / total if total > 0 else 0
                for r in rows
            }
        
        conn.close()
        return result
    
    def compare_day_of_week_patterns(self) -> Dict[str, Dict[str, float]]:
        """Compare day-of-week patterns across stores."""
        
        locations = self.get_all_locations()
        result = {}
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=365)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        for location in locations:
            cursor.execute("""
                SELECT 
                    CAST(strftime('%w', date) AS INTEGER) as dow,
                    SUM(quantity) as dow_qty
                FROM sales
                WHERE location = ? AND date >= ?
                GROUP BY dow
            """, (location, cutoff_str))
            
            rows = cursor.fetchall()
            total = sum(r[1] for r in rows)
            
            result[location] = {
                self.DAY_NAMES[r[0]]: r[1] / total if total > 0 else 0
                for r in rows
            }
        
        conn.close()
        return result
    
    def find_universal_vs_local_patterns(
        self,
        hourly: Dict[str, List[Tuple[int, float]]],
        dow: Dict[str, Dict[str, float]],
        categories: Dict[str, Dict[str, float]],
    ) -> Dict[str, Dict[str, Any]]:
        """Analyze which patterns are universal vs location-specific."""
        
        analysis = {
            "universal_patterns": [],
            "location_specific": [],
            "recommendations": [],
        }
        
        locations = list(hourly.keys())
        if len(locations) < 2:
            analysis["recommendations"].append("Need data from multiple locations for comparison")
            return analysis
        
        # Analyze peak hours
        all_peaks = []
        for loc, hours in hourly.items():
            sorted_hours = sorted(hours, key=lambda x: x[1], reverse=True)
            top3 = [h[0] for h in sorted_hours[:3]]
            all_peaks.append(set(top3))
        
        common_peaks = all_peaks[0]
        for peaks in all_peaks[1:]:
            common_peaks = common_peaks.intersection(peaks)
        
        if len(common_peaks) >= 2:
            analysis["universal_patterns"].append({
                "type": "peak_hours",
                "description": f"All stores peak around {sorted(common_peaks)}",
                "confidence": len(common_peaks) / 3,
            })
        
        # Analyze busiest days - only for stores with data
        all_busiest = []
        loc_to_busiest = {}
        for loc, days in dow.items():
            if days:  # Only if store has day data
                busiest = max(days.items(), key=lambda x: x[1])[0]
                all_busiest.append(busiest)
                loc_to_busiest[loc] = busiest
        
        if len(all_busiest) >= 2:
            if len(set(all_busiest)) == 1:
                analysis["universal_patterns"].append({
                    "type": "busiest_day",
                    "description": f"All stores busiest on {all_busiest[0]}",
                    "confidence": 1.0,
                })
            else:
                analysis["location_specific"].append({
                    "type": "busiest_day",
                    "description": f"Busiest day varies: {loc_to_busiest}",
                    "action": "Staff each store according to its own pattern",
                })
        
        # Analyze category differences
        all_categories = set()
        for cat_dict in categories.values():
            all_categories.update(cat_dict.keys())
        
        for cat in all_categories:
            shares = [categories[loc].get(cat, 0) for loc in locations]
            if len(shares) >= 2 and max(shares) > 0:
                cv = np.std(shares) / np.mean(shares) if np.mean(shares) > 0 else 0
                
                if cv > 0.5:  # High variation
                    best_loc = locations[np.argmax(shares)]
                    worst_loc = locations[np.argmin(shares)]
                    analysis["location_specific"].append({
                        "type": "category_mix",
                        "category": cat,
                        "description": f"{cat} varies: {best_loc} ({max(shares):.1%}) vs {worst_loc} ({min(shares):.1%})",
                        "action": f"Stock more {cat} at {best_loc}",
                    })
        
        return analysis
    
    def save_cross_store_analysis(
        self,
        profiles: Dict[str, StoreProfile],
        analysis: Dict[str, Any],
    ) -> str:
        """Save cross-store analysis to JSON."""
        import json
        
        output = {
            "generated": str(datetime.now()),
            "locations": list(profiles.keys()),
            "profiles": {
                loc: {
                    "avg_daily_units": p.avg_daily_units,
                    "avg_daily_revenue": p.avg_daily_revenue,
                    "top_categories": p.top_categories,
                    "peak_hours": p.peak_hours,
                    "busiest_day": p.busiest_day,
                    "slowest_day": p.slowest_day,
                }
                for loc, p in profiles.items()
            },
            "analysis": analysis,
        }
        
        # Save
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "cross_store_analysis.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate cross-store learning."""
    
    print("=" * 70)
    print("🏪 CROSS-STORE LEARNING")
    print("   Comparing patterns across all locations")
    print("=" * 70)
    print()
    
    learner = CrossStoreLearner()
    
    # Get all locations
    locations = learner.get_all_locations()
    print(f"Found {len(locations)} locations: {locations}")
    print()
    
    if len(locations) < 2:
        print("Need at least 2 locations for cross-store comparison.")
        return
    
    # Build profiles
    print("Building store profiles...")
    profiles = {}
    for loc in locations:
        profiles[loc] = learner.build_store_profile(loc)
    
    print("\n" + "=" * 70)
    print("📊 STORE PROFILES")
    print("=" * 70 + "\n")
    
    for profile in profiles.values():
        print(profile)
        print()
    
    # Compare patterns
    print("Comparing patterns across stores...")
    hourly = learner.compare_hourly_patterns()
    dow = learner.compare_day_of_week_patterns()
    categories = learner.compare_category_mix()
    
    # Find universal vs local
    analysis = learner.find_universal_vs_local_patterns(hourly, dow, categories)
    
    print("\n" + "=" * 70)
    print("🌍 UNIVERSAL PATTERNS (apply to all stores)")
    print("=" * 70 + "\n")
    
    for pattern in analysis.get("universal_patterns", []):
        print(f"   ✓ {pattern['description']}")
    
    if not analysis.get("universal_patterns"):
        print("   (No strongly universal patterns found)")
    
    print("\n" + "=" * 70)
    print("📍 LOCATION-SPECIFIC PATTERNS")
    print("=" * 70 + "\n")
    
    for pattern in analysis.get("location_specific", []):
        print(f"   • {pattern['description']}")
        if pattern.get("action"):
            print(f"     → {pattern['action']}")
    
    if not analysis.get("location_specific"):
        print("   (Stores are remarkably similar)")
    
    # Day-of-week comparison
    print("\n" + "=" * 70)
    print("📅 DAY-OF-WEEK BY STORE")
    print("=" * 70 + "\n")
    
    print(f"{'Day':<12}", end="")
    for loc in locations:
        print(f"{loc:<15}", end="")
    print()
    print("-" * (12 + 15 * len(locations)))
    
    for day in learner.DAY_NAMES:
        print(f"{day:<12}", end="")
        for loc in locations:
            pct = dow.get(loc, {}).get(day, 0)
            bar = "█" * int(pct * 30)
            print(f"{pct:>5.1%} {bar:<8}", end="")
        print()
    
    # Save results
    output_file = learner.save_cross_store_analysis(profiles, analysis)
    print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    demo()
