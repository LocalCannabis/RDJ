"""
Inventory Recommendations

Combines all learned signals to generate actionable inventory recommendations:
- Product-weather impacts (Pre-Rolls spike on warm days)
- Event calendar (cruise ships, sports, concerts)
- Time-of-day patterns (Friday rush)
- Historical category performance

Output: "Stock X% more Pre-Rolls for the warm weekend with cruise ships"
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class StockAction(Enum):
    """Recommended stock action."""
    HEAVY_INCREASE = "heavy_increase"   # +30%+
    INCREASE = "increase"               # +15-30%
    SLIGHT_INCREASE = "slight_increase" # +5-15%
    NORMAL = "normal"                   # -5% to +5%
    SLIGHT_DECREASE = "slight_decrease" # -5-15%
    DECREASE = "decrease"               # -15-30%


@dataclass
class CategoryRecommendation:
    """Recommendation for a specific category."""
    category: str
    action: StockAction
    expected_lift: float
    reasons: List[str]
    confidence: float
    priority: int  # 1 = highest
    
    def __str__(self) -> str:
        action_emoji = {
            StockAction.HEAVY_INCREASE: "🚀🚀",
            StockAction.INCREASE: "📈",
            StockAction.SLIGHT_INCREASE: "↗️",
            StockAction.NORMAL: "➡️",
            StockAction.SLIGHT_DECREASE: "↘️",
            StockAction.DECREASE: "📉",
        }
        emoji = action_emoji.get(self.action, "•")
        return (
            f"{emoji} {self.category}: {self.action.value} ({self.expected_lift:+.0%})\n"
            f"   Reasons: {', '.join(self.reasons)}\n"
            f"   Confidence: {self.confidence:.0%}, Priority: {self.priority}"
        )


@dataclass
class DayRecommendation:
    """Full recommendation for a day."""
    date: str
    day_name: str
    overall_lift: float
    categories: List[CategoryRecommendation]
    staffing_note: str
    key_events: List[str]
    
    def __str__(self) -> str:
        return (
            f"📅 {self.day_name} {self.date} | Overall: {self.overall_lift:+.0%}\n"
            f"   Events: {', '.join(self.key_events) if self.key_events else 'None'}\n"
            f"   Staffing: {self.staffing_note}"
        )


class InventoryRecommender:
    """
    Generates inventory recommendations by combining all brain learnings.
    """
    
    # Category baseline daily units (from cross-store analysis)
    CATEGORY_BASELINES = {
        "Flower": 45,
        "Pre-Rolls": 35,
        "Vapes": 30,
        "Edibles": 25,
        "Beverages": 15,
        "Concentrates": 20,
        "Accessories": 40,
        "Hash": 8,
        "Topicals": 5,
        "Capsules": 3,
    }
    
    def __init__(self):
        self.brain_dir = Path(__file__).parent / "data"
        
        # Load all brain data
        self.signal_magnitudes = self._load_json("learned_signal_magnitudes.json")
        self.weather_impacts = self._load_json("category_weather_impacts.json")
        self.time_patterns = self._load_json("time_of_day_patterns.json")
        self.predictive_calendar = self._load_json("predictive_calendar.json")
        self.event_correlations = self._load_json("event_impact_analysis.json")
        
    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from brain/data."""
        path = self.brain_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}
    
    def _classify_action(self, lift: float) -> StockAction:
        """Classify expected lift into stock action."""
        if lift >= 0.30:
            return StockAction.HEAVY_INCREASE
        elif lift >= 0.15:
            return StockAction.INCREASE
        elif lift >= 0.05:
            return StockAction.SLIGHT_INCREASE
        elif lift >= -0.05:
            return StockAction.NORMAL
        elif lift >= -0.15:
            return StockAction.SLIGHT_DECREASE
        else:
            return StockAction.DECREASE
    
    def _get_weather_impact(self, category: str, conditions: List[str]) -> Tuple[float, str]:
        """
        Get weather impact for a category given conditions.
        Returns (lift, reason_string).
        """
        if not self.weather_impacts:
            return 0.0, ""
        
        impacts = self.weather_impacts.get("actionable_insights", [])
        
        total_lift = 0.0
        reasons = []
        
        for impact in impacts:
            if impact.get("category") == category:
                cond = impact.get("weather_condition", "")
                if cond in conditions:
                    lift = impact.get("avg_lift_pct", 0) / 100  # Convert from %
                    total_lift += lift
                    reasons.append(f"{cond}: {lift:+.0%}")
        
        reason_str = "; ".join(reasons) if reasons else ""
        return total_lift, reason_str
    
    def _get_event_impacts(self, forecast_date: str) -> Tuple[float, List[str]]:
        """
        Get event impacts for a specific date from predictive calendar.
        Returns (total_lift, list_of_event_names).
        """
        if not self.predictive_calendar:
            return 0.0, []
        
        forecasts = self.predictive_calendar.get("forecasts", [])
        
        for forecast in forecasts:
            if forecast.get("date") == forecast_date:
                # combined_lift_pct is in percentage form (e.g., 15.0 = 15%)
                lift = forecast.get("combined_lift_pct", 0) / 100
                # Events have "name" key, not "event_name"
                events = [e.get("name", "Unknown") for e in forecast.get("events", [])]
                return lift, events
        
        return 0.0, []
    
    def _get_day_of_week_lift(self, day_name: str) -> float:
        """Get baseline lift for day of week from time patterns."""
        if not self.time_patterns:
            return 0.0
        
        # Friday is typically +15-20% above average
        day_lifts = {
            "Monday": -0.05,
            "Tuesday": -0.03,
            "Wednesday": 0.0,
            "Thursday": 0.02,
            "Friday": 0.15,
            "Saturday": 0.10,
            "Sunday": 0.05,
        }
        return day_lifts.get(day_name, 0.0)
    
    def generate_day_recommendation(
        self,
        target_date: date,
        weather_conditions: List[str] = None,
    ) -> DayRecommendation:
        """
        Generate inventory recommendation for a specific day.
        
        Args:
            target_date: The date to generate recommendations for
            weather_conditions: Optional list of expected conditions like ["warm", "dry"]
        """
        date_str = target_date.strftime("%Y-%m-%d")
        day_name = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][target_date.weekday()]
        
        # Get event impacts
        event_lift, events = self._get_event_impacts(date_str)
        
        # Get day-of-week baseline lift
        dow_lift = self._get_day_of_week_lift(day_name)
        
        # Calculate overall lift
        overall_lift = event_lift + dow_lift
        
        # Generate category-specific recommendations
        categories = []
        weather_conditions = weather_conditions or []
        
        for category, baseline_units in self.CATEGORY_BASELINES.items():
            reasons = []
            category_lift = overall_lift
            
            # Add weather impact if conditions provided
            if weather_conditions:
                weather_lift, weather_reason = self._get_weather_impact(category, weather_conditions)
                category_lift += weather_lift
                if weather_reason:
                    reasons.append(f"Weather: {weather_reason}")
            
            # Add event reasons
            if events:
                reasons.append(f"Events: {', '.join(events[:2])}")
            
            # Add day-of-week if significant
            if abs(dow_lift) >= 0.05:
                reasons.append(f"{day_name} effect: {dow_lift:+.0%}")
            
            if not reasons:
                reasons = ["Normal day"]
            
            # Determine confidence based on data quality
            confidence = 0.7  # Base confidence
            if events:
                confidence += 0.15
            if weather_conditions:
                confidence += 0.10
            confidence = min(confidence, 0.95)
            
            # Priority based on expected impact magnitude
            if abs(category_lift) >= 0.20:
                priority = 1
            elif abs(category_lift) >= 0.10:
                priority = 2
            else:
                priority = 3
            
            categories.append(CategoryRecommendation(
                category=category,
                action=self._classify_action(category_lift),
                expected_lift=category_lift,
                reasons=reasons,
                confidence=confidence,
                priority=priority,
            ))
        
        # Sort by priority then by impact
        categories.sort(key=lambda x: (x.priority, -abs(x.expected_lift)))
        
        # Generate staffing note
        if overall_lift >= 0.20:
            staffing = "🟠 Extra staff recommended"
        elif overall_lift >= 0.10:
            staffing = "🟡 Normal+ (add 1 if possible)"
        elif overall_lift <= -0.10:
            staffing = "🔵 Light staffing OK"
        else:
            staffing = "🟢 Normal staffing"
        
        return DayRecommendation(
            date=date_str,
            day_name=day_name,
            overall_lift=overall_lift,
            categories=categories,
            staffing_note=staffing,
            key_events=events,
        )
    
    def generate_week_recommendations(
        self,
        start_date: date = None,
        weather_forecast: Dict[str, List[str]] = None,
        use_real_weather: bool = True,
    ) -> List[DayRecommendation]:
        """
        Generate recommendations for the next 7 days.
        
        Args:
            start_date: Start date (defaults to tomorrow)
            weather_forecast: Dict mapping date strings to weather conditions
            use_real_weather: If True and no forecast provided, fetch from API
        """
        if start_date is None:
            start_date = date.today() + timedelta(days=1)
        
        # Fetch real weather if not provided
        if weather_forecast is None and use_real_weather:
            try:
                from aoc_analytics.brain.weather_api import WeatherAPI
                api = WeatherAPI("Vancouver")
                weather_forecast = api.get_forecast_dict()
                print(f"  📡 Using real weather forecast from Open-Meteo API")
            except Exception as e:
                print(f"  ⚠️ Weather API failed: {e}, using defaults")
                weather_forecast = {}
        
        weather_forecast = weather_forecast or {}
        recommendations = []
        
        for i in range(7):
            target_date = start_date + timedelta(days=i)
            date_str = target_date.strftime("%Y-%m-%d")
            conditions = weather_forecast.get(date_str, [])
            
            rec = self.generate_day_recommendation(target_date, conditions)
            recommendations.append(rec)
        
        return recommendations
    
    def get_priority_items(
        self,
        recommendations: List[DayRecommendation],
        top_n: int = 5,
    ) -> List[Tuple[str, str, CategoryRecommendation]]:
        """
        Get the top priority items across all days.
        
        Returns list of (date, day_name, recommendation) tuples.
        """
        all_items = []
        
        for day_rec in recommendations:
            for cat_rec in day_rec.categories:
                if cat_rec.priority == 1 or cat_rec.expected_lift >= 0.15:
                    all_items.append((day_rec.date, day_rec.day_name, cat_rec))
        
        # Sort by expected lift
        all_items.sort(key=lambda x: -abs(x[2].expected_lift))
        
        return all_items[:top_n]
    
    def format_order_suggestions(
        self,
        recommendations: List[DayRecommendation],
    ) -> str:
        """
        Format recommendations into an order suggestion report.
        """
        lines = []
        lines.append("=" * 70)
        lines.append("📦 INVENTORY ORDER SUGGESTIONS")
        lines.append("   Based on predicted demand for the next week")
        lines.append("=" * 70)
        lines.append("")
        
        # Summary table
        lines.append("DAILY OVERVIEW:")
        lines.append("-" * 50)
        for rec in recommendations:
            emoji = "🔥" if rec.overall_lift >= 0.15 else "📈" if rec.overall_lift >= 0.05 else "➡️"
            events_str = f" [{', '.join(rec.key_events[:2])}]" if rec.key_events else ""
            lines.append(f"  {emoji} {rec.day_name[:3]} {rec.date}: {rec.overall_lift:+.0%}{events_str}")
        lines.append("")
        
        # Priority items
        priority = self.get_priority_items(recommendations)
        if priority:
            lines.append("⚡ HIGH PRIORITY ITEMS:")
            lines.append("-" * 50)
            for date_str, day_name, cat_rec in priority:
                lines.append(f"  {day_name[:3]} {date_str}: {cat_rec.category} {cat_rec.expected_lift:+.0%}")
                lines.append(f"      → {cat_rec.action.value.replace('_', ' ').title()}")
            lines.append("")
        
        # Category-by-category
        lines.append("CATEGORY DETAILS:")
        lines.append("-" * 50)
        
        # Aggregate category impacts across week
        category_totals = {}
        for rec in recommendations:
            for cat_rec in rec.categories:
                if cat_rec.category not in category_totals:
                    category_totals[cat_rec.category] = {
                        "lifts": [],
                        "reasons": set(),
                    }
                category_totals[cat_rec.category]["lifts"].append(cat_rec.expected_lift)
                for r in cat_rec.reasons:
                    if "Events:" in r or "Weather:" in r:
                        category_totals[cat_rec.category]["reasons"].add(r)
        
        # Sort by average lift
        sorted_cats = sorted(
            category_totals.items(),
            key=lambda x: -max(x[1]["lifts"])
        )
        
        for category, data in sorted_cats[:6]:  # Top 6 categories
            avg_lift = sum(data["lifts"]) / len(data["lifts"])
            max_lift = max(data["lifts"])
            
            if max_lift >= 0.10:
                action = self._classify_action(max_lift)
                baseline = self.CATEGORY_BASELINES.get(category, 20)
                suggested_units = int(baseline * (1 + max_lift))
                
                lines.append(f"\n  {category}:")
                lines.append(f"      Peak demand: {max_lift:+.0%} (avg: {avg_lift:+.0%})")
                lines.append(f"      Suggested: {suggested_units} units/day (baseline: {baseline})")
                if data["reasons"]:
                    lines.append(f"      Drivers: {'; '.join(list(data['reasons'])[:2])}")
        
        lines.append("")
        lines.append("=" * 70)
        
        return "\n".join(lines)
    
    def save_recommendations(self, recommendations: List[DayRecommendation]) -> str:
        """Save recommendations to JSON."""
        
        output = {
            "generated": datetime.now().isoformat(),
            "period": {
                "start": recommendations[0].date if recommendations else None,
                "end": recommendations[-1].date if recommendations else None,
            },
            "recommendations": [
                {
                    "date": rec.date,
                    "day_name": rec.day_name,
                    "overall_lift": rec.overall_lift,
                    "staffing": rec.staffing_note,
                    "key_events": rec.key_events,
                    "categories": [
                        {
                            "category": cat.category,
                            "action": cat.action.value,
                            "expected_lift": cat.expected_lift,
                            "reasons": cat.reasons,
                            "confidence": cat.confidence,
                            "priority": cat.priority,
                        }
                        for cat in rec.categories
                    ]
                }
                for rec in recommendations
            ],
        }
        
        output_file = self.brain_dir / "inventory_recommendations.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate inventory recommendations."""
    
    print("=" * 70)
    print("📦 INVENTORY RECOMMENDATIONS")
    print("   What should we stock based on what we've learned?")
    print("=" * 70)
    print()
    
    recommender = InventoryRecommender()
    
    # Generate week recommendations with REAL weather
    print("Generating 7-day forecast...")
    
    tomorrow = date.today() + timedelta(days=1)
    
    # Let it fetch real weather automatically
    recommendations = recommender.generate_week_recommendations(
        start_date=tomorrow,
        use_real_weather=True,  # Will fetch from Open-Meteo API
    )
    
    # Print formatted report
    print(recommender.format_order_suggestions(recommendations))
    
    # Print detailed day view for highest impact day
    highest = max(recommendations, key=lambda x: x.overall_lift)
    print("\n" + "=" * 70)
    print(f"🔥 HIGHEST IMPACT DAY: {highest.day_name} {highest.date}")
    print("=" * 70)
    print(highest)
    print("\nCategory breakdown:")
    for cat in highest.categories[:5]:
        print(f"  {cat}")
    
    # Save
    output_file = recommender.save_recommendations(recommendations)
    print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    demo()
