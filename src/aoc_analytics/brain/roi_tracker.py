"""ROI Tracker - Calculate the dollar value of the brain's predictions.

This module answers: "How much money did following these predictions save/make?"

Key metrics:
1. Waste Avoided: If we predicted a slow day and reduced inventory
2. Stockout Prevention: If we predicted a rush and had adequate stock
3. Total Value Generated: Sum of all prediction-driven savings

"If I had followed these recommendations, I would have saved/made $X"
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

from aoc_analytics.core.db_adapter import get_connection


@dataclass
class DayROI:
    """ROI calculation for a single day."""
    date: str
    predicted_lift: float  # What we predicted
    actual_lift: float     # What actually happened
    prediction_followed: bool  # Did user follow the recommendation?
    
    # Cost/revenue assumptions
    avg_transaction_value: float = 35.0
    daily_baseline_transactions: int = 100
    inventory_cost_per_unit: float = 12.0
    waste_rate_without_prediction: float = 0.08  # 8% waste normally
    stockout_loss_rate: float = 0.15  # 15% lost sales from stockouts
    
    # Calculated values
    waste_avoided: float = 0.0
    stockout_prevented: float = 0.0
    total_value: float = 0.0
    
    def calculate(self):
        """Calculate ROI for this day."""
        prediction_error = abs(self.predicted_lift - self.actual_lift)
        
        if self.actual_lift > 0.1:  # High-demand day (>10%)
            if self.predicted_lift > 0.05:  # We correctly predicted elevated demand
                # Stockout prevention value
                extra_transactions = self.daily_baseline_transactions * self.actual_lift
                potential_loss = extra_transactions * self.avg_transaction_value * self.stockout_loss_rate
                self.stockout_prevented = potential_loss * (1 - min(prediction_error, 0.5))
            # If we missed a high day, no value (but also no penalty in this calc)
        
        elif self.actual_lift < -0.05:  # Slow day (<-5%)
            if self.predicted_lift < 0:  # We correctly predicted slow
                # Waste reduction value
                reduced_inventory = abs(self.predicted_lift) * self.daily_baseline_transactions
                self.waste_avoided = reduced_inventory * self.inventory_cost_per_unit * self.waste_rate_without_prediction
            elif abs(self.predicted_lift) < 0.03:  # Neutral prediction on slow day still OK
                # Small value for not over-ordering
                self.waste_avoided = 5.0  # $5 for being conservative
        
        else:  # Normal day (-5% to +10%)
            # Value from accurate normal-day prediction (not over/under preparing)
            if abs(self.predicted_lift) < 0.05:  # Correctly predicted normal
                # $10 value for correctly identifying a normal day
                self.total_value = 10.0
                return
        
        # Bonus for accurate predictions (within 10% error)
        if prediction_error < 0.1:
            accuracy_bonus = self.avg_transaction_value * 2  # $70 bonus for accurate prediction
            self.total_value = self.waste_avoided + self.stockout_prevented + accuracy_bonus
        elif prediction_error < 0.15:
            accuracy_bonus = self.avg_transaction_value  # $35 for close prediction
            self.total_value = self.waste_avoided + self.stockout_prevented + accuracy_bonus
        else:
            self.total_value = self.waste_avoided + self.stockout_prevented


@dataclass
class ROIReport:
    """Full ROI report over a time period."""
    period_start: str
    period_end: str
    days_analyzed: int
    
    # High-level metrics
    total_value_generated: float = 0.0
    waste_avoided: float = 0.0
    stockout_prevented: float = 0.0
    
    # Accuracy metrics
    high_demand_days_predicted: int = 0
    high_demand_days_actual: int = 0
    correct_high_demand_predictions: int = 0
    
    slow_days_predicted: int = 0
    slow_days_actual: int = 0
    correct_slow_predictions: int = 0
    
    # Per-category breakdown
    category_roi: dict = field(default_factory=dict)
    
    # Daily details
    daily_details: list = field(default_factory=list)
    
    # Grade
    grade: str = "N/A"
    grade_explanation: str = ""
    
    def calculate_grade(self):
        """Calculate overall ROI grade."""
        if self.days_analyzed == 0:
            self.grade = "N/A"
            self.grade_explanation = "No data to analyze"
            return
            
        # Value per day
        value_per_day = self.total_value_generated / self.days_analyzed
        
        # High-demand accuracy (caught / actual high days)
        if self.high_demand_days_actual > 0:
            high_recall = self.correct_high_demand_predictions / self.high_demand_days_actual
        else:
            high_recall = 0
            
        # Slow day accuracy
        if self.slow_days_actual > 0:
            slow_recall = self.correct_slow_predictions / self.slow_days_actual
        else:
            slow_recall = 0
            
        # Combined accuracy (weighted: rush days matter more)
        combined_accuracy = (high_recall * 0.7 + slow_recall * 0.3)
        
        # Grade based on value AND accuracy
        # Realistic targets given data limitations:
        # - A: Excellent actionable predictions
        # - B: Good enough to act on
        # - C: Some value, needs work
        # - D: Minimal value
        
        if value_per_day > 75 and combined_accuracy > 0.4:
            self.grade = "A"
            self.grade_explanation = f"Excellent! ${value_per_day:.0f}/day, {combined_accuracy:.0%} combined accuracy"
        elif value_per_day > 40 and combined_accuracy > 0.25:
            self.grade = "B"
            self.grade_explanation = f"Good performance. ${value_per_day:.0f}/day, {combined_accuracy:.0%} accuracy"
        elif value_per_day > 20:
            self.grade = "C"
            self.grade_explanation = f"Moderate value. ${value_per_day:.0f}/day, predictions need tuning"
        elif value_per_day > 0:
            self.grade = "D"
            self.grade_explanation = f"Minimal value. ${value_per_day:.0f}/day, significant improvement needed"
        else:
            self.grade = "F"
            self.grade_explanation = "Predictions not generating measurable value"


class ROITracker:
    """
    Tracks ROI of the brain's predictions.
    
    Compares predicted impact vs actual sales to calculate:
    - How much waste we avoided on slow days
    - How much revenue we captured on rush days
    - Total dollar value of following the brain's recommendations
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "aoc_sales.db"
        self.db_path = Path(db_path)
        self.output_dir = Path(__file__).parent / "data"
        self.output_dir.mkdir(exist_ok=True)
        
        # Load predictions from brain output
        self.predictions_file = self.output_dir / "predictive_calendar.json"
        self.backtest_file = self.output_dir / "backtest_results.json"
        
    def load_predictions(self) -> dict:
        """Load the brain's predictions."""
        if self.predictions_file.exists():
            with open(self.predictions_file) as f:
                return json.load(f)
        return {}
        
    def load_backtest(self) -> dict:
        """Load backtest results for accuracy metrics."""
        if self.backtest_file.exists():
            with open(self.backtest_file) as f:
                return json.load(f)
        return {}
        
    def get_actual_sales(self, date: str) -> Optional[float]:
        """Get actual sales for a date from database."""
        if not self.db_path.exists():
            return None
            
        conn = get_connection(str(self.db_path))
        cursor = conn.cursor()
        
        # Get sales for the date
        cursor.execute("""
            SELECT SUM(total_price) 
            FROM sales 
            WHERE DATE(sold_at) = ?
        """, (date,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else None
        
    def get_baseline_for_day(self, day_of_week: int) -> float:
        """Get baseline sales for a day of week."""
        if not self.db_path.exists():
            return 3500.0  # Default baseline
            
        conn = get_connection(str(self.db_path))
        cursor = conn.cursor()
        
        # Get average sales for this day of week over last 90 days
        cursor.execute("""
            SELECT AVG(daily_total) FROM (
                SELECT DATE(sold_at) as sale_date, SUM(total_price) as daily_total
                FROM sales
                WHERE sold_at >= DATE('now', '-90 days')
                GROUP BY sale_date
                HAVING CAST(strftime('%w', sale_date) AS INTEGER) = ?
            )
        """, (day_of_week,))
        
        result = cursor.fetchone()
        conn.close()
        
        return result[0] if result and result[0] else 3500.0
        
    def calculate_historical_roi(self, days_back: int = 90) -> ROIReport:
        """
        Calculate ROI over the past N days.
        
        Compares what we would have predicted vs what actually happened.
        """
        backtest = self.load_backtest()
        
        report = ROIReport(
            period_start=(datetime.now() - timedelta(days=days_back)).strftime("%Y-%m-%d"),
            period_end=datetime.now().strftime("%Y-%m-%d"),
            days_analyzed=0
        )
        
        if not backtest or "daily_comparison" not in backtest:
            # Generate ROI from scratch using database
            return self._calculate_roi_from_database(days_back, report)
            
        # Use backtest results
        for day_data in backtest.get("daily_comparison", []):
            day_roi = DayROI(
                date=day_data["date"],
                predicted_lift=day_data["predicted_lift"],
                actual_lift=day_data["actual_lift"],
                prediction_followed=True  # Assume for calculation
            )
            day_roi.calculate()
            
            report.days_analyzed += 1
            report.total_value_generated += day_roi.total_value
            report.waste_avoided += day_roi.waste_avoided
            report.stockout_prevented += day_roi.stockout_prevented
            
            # Track accuracy
            if day_data["predicted_lift"] > 0.1:
                report.high_demand_days_predicted += 1
            if day_data["actual_lift"] > 0.1:
                report.high_demand_days_actual += 1
                if day_data["predicted_lift"] > 0.05:
                    report.correct_high_demand_predictions += 1
                    
            if day_data["predicted_lift"] < -0.05:
                report.slow_days_predicted += 1
            if day_data["actual_lift"] < -0.05:
                report.slow_days_actual += 1
                if day_data["predicted_lift"] < 0:
                    report.correct_slow_predictions += 1
                    
            report.daily_details.append({
                "date": day_roi.date,
                "predicted_lift": f"{day_roi.predicted_lift:+.1%}",
                "actual_lift": f"{day_roi.actual_lift:+.1%}",
                "waste_avoided": f"${day_roi.waste_avoided:.2f}",
                "stockout_prevented": f"${day_roi.stockout_prevented:.2f}",
                "total_value": f"${day_roi.total_value:.2f}"
            })
            
        report.calculate_grade()
        return report
        
    def _calculate_roi_from_database(self, days_back: int, report: ROIReport) -> ROIReport:
        """Calculate ROI directly from database if no backtest available."""
        if not self.db_path.exists():
            report.grade = "N/A"
            report.grade_explanation = "No database available"
            return report
            
        # This would need actual prediction logs
        # For now, return empty report
        report.grade = "N/A"
        report.grade_explanation = "Run backtest first to calculate ROI"
        return report
        
    def project_future_roi(self, days_ahead: int = 30) -> dict:
        """
        Project ROI for following future predictions.
        
        "If you follow the brain's recommendations for the next month,
        you could save/make approximately $X"
        """
        predictions = self.load_predictions()
        
        if not predictions:
            return {"error": "No predictions available"}
            
        # Count high-impact days
        high_impact_days = 0
        total_predicted_lift = 0
        
        upcoming = predictions.get("upcoming_7_days", [])
        for day in upcoming:
            lift = day.get("predicted_lift", 0)
            total_predicted_lift += lift
            if lift > 0.15:
                high_impact_days += 1
                
        # Project based on what we've seen
        avg_value_per_high_day = 150  # Based on stockout prevention
        avg_value_per_normal_day = 50  # Baseline optimization
        
        # Scale to 30 days
        scale_factor = days_ahead / max(len(upcoming), 1)
        
        projected_high_days = high_impact_days * scale_factor
        projected_normal_days = days_ahead - projected_high_days
        
        projected_value = (
            projected_high_days * avg_value_per_high_day +
            projected_normal_days * avg_value_per_normal_day
        )
        
        return {
            "period_days": days_ahead,
            "projected_high_impact_days": int(projected_high_days),
            "projected_total_value": f"${projected_value:.0f}",
            "value_per_day": f"${projected_value / days_ahead:.0f}",
            "assumptions": [
                "$150/day value from correctly predicting rush days (stockout prevention)",
                "$50/day value from baseline inventory optimization",
                f"Based on {len(upcoming)} days of predictions"
            ]
        }
        
    def generate_roi_summary(self) -> dict:
        """Generate complete ROI summary."""
        historical = self.calculate_historical_roi(90)
        future = self.project_future_roi(30)
        
        summary = {
            "generated_at": datetime.now().isoformat(),
            "headline": self._generate_headline(historical, future),
            
            "historical_performance": {
                "period": f"{historical.period_start} to {historical.period_end}",
                "days_analyzed": historical.days_analyzed,
                "grade": historical.grade,
                "explanation": historical.grade_explanation,
                
                "total_value_generated": f"${historical.total_value_generated:.0f}",
                "waste_avoided": f"${historical.waste_avoided:.0f}",
                "stockout_prevented": f"${historical.stockout_prevented:.0f}",
                
                "value_per_day": f"${historical.total_value_generated / max(historical.days_analyzed, 1):.0f}",
                "value_per_month": f"${historical.total_value_generated / max(historical.days_analyzed, 1) * 30:.0f}",
                
                "accuracy": {
                    "high_demand_days_caught": f"{historical.correct_high_demand_predictions}/{historical.high_demand_days_actual}",
                    "slow_days_caught": f"{historical.correct_slow_predictions}/{historical.slow_days_actual}"
                }
            },
            
            "future_projection": future,
            
            "key_insights": self._generate_insights(historical),
            
            "daily_breakdown": historical.daily_details[:10]  # Top 10 days
        }
        
        # Save to file
        output_file = self.output_dir / "roi_report.json"
        with open(output_file, "w") as f:
            json.dump(summary, f, indent=2, default=str)
            
        return summary
        
    def _generate_headline(self, historical: ROIReport, future: dict) -> str:
        """Generate attention-grabbing headline."""
        if historical.days_analyzed == 0:
            return "Start tracking to see your ROI!"
            
        monthly_value = historical.total_value_generated / max(historical.days_analyzed, 1) * 30
        
        if monthly_value > 3000:
            return f"🚀 Brain predictions generating ${monthly_value:.0f}/month in value!"
        elif monthly_value > 1000:
            return f"📈 ${monthly_value:.0f}/month in waste avoided and sales captured"
        elif monthly_value > 0:
            return f"💡 ${monthly_value:.0f}/month - room to grow with better data"
        else:
            return "📊 Gathering data to calculate ROI..."
            
    def _generate_insights(self, historical: ROIReport) -> list:
        """Generate actionable insights from ROI data."""
        insights = []
        
        if historical.days_analyzed == 0:
            return ["Run the brain daemon for a few weeks to generate ROI insights"]
            
        # Stockout prevention insight
        if historical.stockout_prevented > 0:
            insights.append(
                f"💰 Prevented ${historical.stockout_prevented:.0f} in lost sales "
                f"by predicting {historical.correct_high_demand_predictions} rush days"
            )
            
        # Waste reduction insight
        if historical.waste_avoided > 0:
            insights.append(
                f"🗑️ Avoided ${historical.waste_avoided:.0f} in waste "
                f"by predicting {historical.correct_slow_predictions} slow days"
            )
            
        # Accuracy insight
        if historical.high_demand_days_actual > 0:
            accuracy = historical.correct_high_demand_predictions / historical.high_demand_days_actual
            if accuracy > 0.7:
                insights.append(f"🎯 {accuracy:.0%} accuracy on rush days - excellent!")
            elif accuracy > 0.5:
                insights.append(f"🎯 {accuracy:.0%} accuracy on rush days - good, improving")
            else:
                insights.append(f"⚠️ {accuracy:.0%} accuracy on rush days - needs calibration")
                
        # Grade insight
        if historical.grade in ["A", "B"]:
            insights.append("✅ Keep following the brain's recommendations")
        elif historical.grade in ["C", "D"]:
            insights.append("🔧 Brain is learning - accuracy will improve with more data")
        else:
            insights.append("📊 Need more historical data for accurate ROI calculation")
            
        return insights
        
    def print_roi_report(self, summary: dict = None):
        """Print formatted ROI report."""
        if summary is None:
            summary = self.generate_roi_summary()
            
        print("=" * 70)
        print("💰 ROI REPORT")
        print("   How much value is the brain generating?")
        print("=" * 70)
        print()
        print(f"  {summary['headline']}")
        print()
        
        hist = summary["historical_performance"]
        print("-" * 70)
        print(f"  HISTORICAL PERFORMANCE ({hist['period']})")
        print("-" * 70)
        print(f"  Grade: {hist['grade']} - {hist['explanation']}")
        print()
        print(f"  📊 Days Analyzed: {hist['days_analyzed']}")
        print(f"  💵 Total Value: {hist['total_value_generated']}")
        print(f"     └── Value/Day: {hist['value_per_day']}")
        print(f"     └── Value/Month: {hist['value_per_month']}")
        print()
        print(f"  🗑️ Waste Avoided: {hist['waste_avoided']}")
        print(f"  📈 Stockout Prevented: {hist['stockout_prevented']}")
        print()
        print(f"  🎯 Rush Days Caught: {hist['accuracy']['high_demand_days_caught']}")
        print(f"  🐢 Slow Days Caught: {hist['accuracy']['slow_days_caught']}")
        print()
        
        future = summary["future_projection"]
        if "error" not in future:
            print("-" * 70)
            print(f"  30-DAY PROJECTION")
            print("-" * 70)
            print(f"  📅 High-Impact Days Expected: {future['projected_high_impact_days']}")
            print(f"  💰 Projected Value: {future['projected_total_value']}")
            print(f"     └── {future['value_per_day']}/day")
            print()
            
        print("-" * 70)
        print("  KEY INSIGHTS")
        print("-" * 70)
        for insight in summary.get("key_insights", []):
            print(f"  {insight}")
        print()
        
        if summary.get("daily_breakdown"):
            print("-" * 70)
            print("  TOP DAYS (Recent)")
            print("-" * 70)
            for day in summary["daily_breakdown"][:5]:
                print(f"  {day['date']}: Predicted {day['predicted_lift']}, "
                      f"Actual {day['actual_lift']} → {day['total_value']}")
        print()


def demo():
    """Run ROI tracker demo."""
    tracker = ROITracker()
    summary = tracker.generate_roi_summary()
    tracker.print_roi_report(summary)
    
    print("=" * 70)
    print(f"  Report saved to: brain/data/roi_report.json")
    print("=" * 70)


if __name__ == "__main__":
    demo()
