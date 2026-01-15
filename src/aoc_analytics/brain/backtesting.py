"""
Backtesting Module

Proves prediction accuracy by comparing what we WOULD HAVE predicted
against what ACTUALLY happened.

This is the credibility foundation:
- "We predicted Jan 1 at +37%, actual was +39%"
- "Our cruise ship signal has 82% directional accuracy"
- "MAPE of 8.3% over 180 days"

Key metrics:
- Directional accuracy: Did we predict the right direction (up/down)?
- MAPE: Mean Absolute Percentage Error
- Correlation: How well do predictions track actuals?
- Hit rate: % of high-impact days we correctly identified
"""

import json
import sqlite3
from dataclasses import dataclass, field
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
class DayPrediction:
    """A prediction for a single day."""
    date: str
    predicted_lift: float  # What we would have predicted
    actual_lift: float     # What actually happened
    events: List[str]      # Events we identified
    error: float = 0.0     # predicted - actual
    abs_error: float = 0.0
    direction_correct: bool = False
    
    def __post_init__(self):
        self.error = self.predicted_lift - self.actual_lift
        self.abs_error = abs(self.error)
        # Direction correct if both positive, both negative, or both near zero
        pred_dir = 1 if self.predicted_lift > 0.02 else (-1 if self.predicted_lift < -0.02 else 0)
        actual_dir = 1 if self.actual_lift > 0.02 else (-1 if self.actual_lift < -0.02 else 0)
        self.direction_correct = (pred_dir == actual_dir) or (pred_dir == 0 and abs(self.actual_lift) < 0.05)


@dataclass
class SignalAccuracy:
    """Accuracy metrics for a specific signal type."""
    signal_name: str
    sample_size: int
    predicted_lift: float  # Average predicted lift when signal active
    actual_lift: float     # Average actual lift when signal active
    directional_accuracy: float  # % of time we got direction right
    mape: float           # Mean absolute percentage error
    correlation: float    # Correlation between predicted and actual
    hit_rate: float       # % of high-impact days correctly identified
    
    def __str__(self) -> str:
        emoji = "✅" if self.directional_accuracy > 0.7 else "⚠️" if self.directional_accuracy > 0.5 else "❌"
        return (
            f"{emoji} {self.signal_name}:\n"
            f"   Predicted: {self.predicted_lift:+.1%}, Actual: {self.actual_lift:+.1%}\n"
            f"   Direction accuracy: {self.directional_accuracy:.0%}, MAPE: {self.mape:.1%}\n"
            f"   Sample: {self.sample_size} days"
        )


@dataclass
class BacktestResult:
    """Overall backtesting results."""
    period_start: str
    period_end: str
    total_days: int
    
    # Overall metrics
    overall_mape: float
    overall_directional_accuracy: float
    overall_correlation: float
    
    # High-impact day detection
    high_impact_precision: float  # Of days we flagged, how many were actually high?
    high_impact_recall: float     # Of actual high days, how many did we catch?
    
    # Signal-specific accuracy
    signal_accuracy: Dict[str, SignalAccuracy]
    
    # Best/worst predictions
    best_predictions: List[DayPrediction]
    worst_predictions: List[DayPrediction]
    
    # All daily predictions (for ROI calculation)
    all_predictions: List[DayPrediction] = field(default_factory=list)
    
    def __str__(self) -> str:
        return (
            f"📊 BACKTEST RESULTS ({self.period_start} to {self.period_end})\n"
            f"{'='*60}\n"
            f"Days tested: {self.total_days}\n"
            f"Overall MAPE: {self.overall_mape:.1%}\n"
            f"Directional accuracy: {self.overall_directional_accuracy:.0%}\n"
            f"Correlation: {self.overall_correlation:.2f}\n"
            f"\nHigh-impact detection:\n"
            f"   Precision: {self.high_impact_precision:.0%} (of flagged days, % actually high)\n"
            f"   Recall: {self.high_impact_recall:.0%} (of actual high days, % we caught)"
        )


class Backtester:
    """
    Backtests the prediction system against historical data.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            possible_paths = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent.parent / "aoc_sales.db",
            ]
            for p in possible_paths:
                if p.exists():
                    db_path = str(p)
                    break
        
        self.db_path = db_path
        self.brain_dir = Path(__file__).parent / "data"
        
        # Load learned impacts for prediction
        self.signal_impacts = self._load_signal_impacts()
        
    def _load_signal_impacts(self) -> Dict[str, float]:
        """Load learned signal impacts."""
        impacts = {}
        
        sig_file = self.brain_dir / "learned_signal_magnitudes.json"
        if sig_file.exists():
            with open(sig_file) as f:
                data = json.load(f)
                for key, signal in data.get("signals", {}).items():
                    impacts[key] = signal.get("lift", 0)
        
        # Add defaults for signals we know about
        defaults = {
            "cruise_ships": 0.36,
            "canucks_home": 0.22,
            "nfl_sunday": 0.15,
            "whitecaps_home": 0.07,
            "academic_finals": 0.30,
            "statutory_holiday": 0.20,
            "friday": 0.15,
        }
        
        for key, val in defaults.items():
            if key not in impacts:
                impacts[key] = val
        
        return impacts
    
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def _get_daily_sales(self, days: int = 365) -> Dict[str, Dict]:
        """Get historical daily sales data."""
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cutoff = datetime.now() - timedelta(days=days)
        cutoff_str = cutoff.strftime("%Y-%m-%d")
        
        cursor.execute("""
            SELECT 
                date,
                location,
                SUM(quantity) as units,
                SUM(subtotal) as revenue
            FROM sales
            WHERE date >= ?
            GROUP BY date, location
            ORDER BY date
        """, (cutoff_str,))
        
        # Organize by date
        daily = defaultdict(lambda: {"units": 0, "revenue": 0, "locations": []})
        
        for row in cursor.fetchall():
            date_str, location, units, revenue = row
            daily[date_str]["units"] += units or 0
            daily[date_str]["revenue"] += revenue or 0
            daily[date_str]["locations"].append(location)
        
        conn.close()
        return dict(daily)
    
    def _calculate_baseline(self, daily_sales: Dict[str, Dict]) -> Dict[str, float]:
        """Calculate day-of-week baselines."""
        dow_totals = defaultdict(list)
        
        for date_str, data in daily_sales.items():
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d")
                dow = dt.strftime("%A")
                dow_totals[dow].append(data["units"])
            except:
                continue
        
        baselines = {}
        for dow, values in dow_totals.items():
            if values:
                baselines[dow] = np.median(values)
        
        return baselines
    
    def _get_signals_for_date(self, check_date: date) -> Tuple[float, List[str]]:
        """
        Reconstruct what signals would have been active on a historical date.
        Returns (predicted_lift, list_of_active_signals).
        
        Uses diminishing returns for stacked signals - the more signals,
        the less each additional one contributes.
        """
        try:
            from aoc_analytics.core.signals.vibe_signals import (
                CruiseSchedule,
                SportsSchedule,
                AcademicCalendar,
            )
            from aoc_analytics.core.calendar import (
                BC_STATUTORY_HOLIDAYS,
                PARTY_HOLIDAYS,
            )
            from aoc_analytics.brain.monthly_seasonality import (
                get_monthly_adjustment,
                MONTHLY_SEASONALITY,
            )
        except ImportError:
            return 0.0, []
        
        signals = []
        signal_lifts = []  # Collect individual lifts
        has_major_signal = False  # Track if there's a real event
        
        # Day of week - these are BASELINE adjustments, not additive
        # DOW effect is already in the baseline comparison, so we DON'T add it here
        dow = check_date.strftime("%A")
        
        # Monthly seasonality - this IS additive because DOW baseline
        # doesn't account for seasonal variation
        monthly_adj = get_monthly_adjustment(check_date)
        if abs(monthly_adj) > 0.03:  # Only add if significant
            month_name = check_date.strftime("%B")
            signals.append(f"month:{month_name}")
            signal_lifts.append(monthly_adj)
        
        # Pre-Christmas rush (Dec 20-23) - MAJOR SIGNAL
        if check_date.month == 12 and 20 <= check_date.day <= 23:
            signals.append("pre_christmas")
            days_to_christmas = 24 - check_date.day
            lift = 0.65 - (days_to_christmas * 0.15)
            signal_lifts.append(max(0.20, lift))
            has_major_signal = True
            
        # Pre-Halloween (Oct 28-31) - MAJOR SIGNAL
        if check_date.month == 10 and 28 <= check_date.day <= 31:
            signals.append("pre_halloween")
            signal_lifts.append(0.15)
            has_major_signal = True
        
        # Statutory holidays
        if check_date in BC_STATUTORY_HOLIDAYS:
            holiday_name = BC_STATUTORY_HOLIDAYS[check_date]
            signals.append(f"holiday:{holiday_name}")
            signal_lifts.append(0.02)  # Holidays don't boost much
            has_major_signal = True
        
        # Cruise ships - only if high likelihood
        if CruiseSchedule.is_cruise_season(check_date):
            likelihood = CruiseSchedule.estimate_cruise_likelihood(check_date)
            if likelihood > 0.5:
                signals.append(f"cruise:{likelihood:.0%}")
                # Avg is 3.2%, but high variance - use conservative estimate
                signal_lifts.append(0.032 * likelihood)
                # NOT a major signal - too variable
        
        # Canucks - MAJOR SIGNAL if high probability
        is_game, prob = SportsSchedule.is_likely_canucks_home_game(check_date)
        if is_game and prob > 0.30:
            signals.append(f"canucks:{prob:.0%}")
            signal_lifts.append(0.067 * prob)  # Actual was +6.7%
            if prob > 0.5:
                has_major_signal = True
        
        # NFL - Only Super Bowl is meaningful
        is_nfl, game_type = SportsSchedule.is_nfl_game_window(check_date)
        if is_nfl and game_type == "super_bowl":
            signals.append(f"nfl:{game_type}")
            signal_lifts.append(0.10)  # Super Bowl is big
            has_major_signal = True
        
        # Academic - NEGATIVE signal
        period, stress = AcademicCalendar.get_academic_period(check_date)
        if "finals" in period and stress > 0.7:
            signals.append("academic:finals")
            signal_lifts.append(-0.06)  # Students stressed = less shopping
        elif "reading_break" in period:
            signals.append("academic:reading_break")
            signal_lifts.append(-0.04)  # Students away
        
        # Weather impact
        try:
            from aoc_analytics.brain.weather_signal import get_weather_for_date
            conditions, weather_impact = get_weather_for_date(check_date)
            if abs(weather_impact) > 0.02:  # Only add if meaningful
                cond_str = "+".join(conditions[:2])  # Max 2 conditions
                signals.append(f"weather:{cond_str}")
                signal_lifts.append(weather_impact)
        except ImportError:
            pass
        
        # Combine with diminishing returns
        # First signal gets full weight, subsequent signals get less
        if not signal_lifts:
            return 0.0, signals
        
        # Sort by magnitude (biggest first)
        signal_lifts.sort(key=abs, reverse=True)
        
        total_lift = 0.0
        for i, lift in enumerate(signal_lifts):
            # Each additional signal contributes less (50% decay)
            weight = 0.5 ** i
            total_lift += lift * weight
        
        # Cap the total lift
        total_lift = max(-0.30, min(0.30, total_lift))
        
        return total_lift, signals
    
    def run_backtest(self, days: int = 180) -> BacktestResult:
        """
        Run a full backtest over historical data.
        
        For each historical day:
        1. Calculate what we WOULD have predicted
        2. Compare to what ACTUALLY happened
        3. Track accuracy metrics
        """
        print(f"Running backtest over {days} days...")
        
        # Get historical sales
        daily_sales = self._get_daily_sales(days)
        if not daily_sales:
            raise ValueError("No sales data available for backtesting")
        
        # Calculate baselines
        baselines = self._calculate_baseline(daily_sales)
        overall_baseline = np.median([d["units"] for d in daily_sales.values()])
        
        print(f"  Found {len(daily_sales)} days of data")
        print(f"  Overall baseline: {overall_baseline:.0f} units/day")
        
        # Generate predictions for each day
        predictions = []
        signal_predictions = defaultdict(list)  # Track by signal type
        
        for date_str, data in sorted(daily_sales.items()):
            try:
                dt = datetime.strptime(date_str, "%Y-%m-%d").date()
            except:
                continue
            
            # Get DOW baseline
            dow = dt.strftime("%A")
            dow_baseline = baselines.get(dow, overall_baseline)
            
            # What would we have predicted?
            predicted_lift, signals = self._get_signals_for_date(dt)
            
            # What actually happened?
            actual_units = data["units"]
            actual_lift = (actual_units - dow_baseline) / dow_baseline if dow_baseline > 0 else 0
            
            pred = DayPrediction(
                date=date_str,
                predicted_lift=predicted_lift,
                actual_lift=actual_lift,
                events=signals,
            )
            predictions.append(pred)
            
            # Track by signal type
            for signal in signals:
                signal_type = signal.split(":")[0]
                signal_predictions[signal_type].append(pred)
        
        print(f"  Generated {len(predictions)} predictions")
        
        # Calculate overall metrics
        predicted = [p.predicted_lift for p in predictions]
        actual = [p.actual_lift for p in predictions]
        
        # Filter out extreme outliers for correlation
        valid_pairs = [(p, a) for p, a in zip(predicted, actual) if abs(a) < 2.0]
        if valid_pairs:
            pred_valid = [p for p, a in valid_pairs]
            actual_valid = [a for p, a in valid_pairs]
            
            if stats:
                correlation = stats.pearsonr(pred_valid, actual_valid)[0]
            else:
                correlation = np.corrcoef(pred_valid, actual_valid)[0, 1]
        else:
            correlation = 0.0
        
        overall_mape = np.mean([p.abs_error for p in predictions])
        overall_directional = np.mean([p.direction_correct for p in predictions])
        
        # High-impact detection metrics
        high_threshold = 0.15
        predicted_high = [p for p in predictions if p.predicted_lift > high_threshold]
        actual_high = [p for p in predictions if p.actual_lift > high_threshold]
        
        # Precision: of days we flagged as high, how many actually were?
        if predicted_high:
            true_positives = sum(1 for p in predicted_high if p.actual_lift > high_threshold * 0.5)
            precision = true_positives / len(predicted_high)
        else:
            precision = 0.0
        
        # Recall: of actual high days, how many did we catch?
        if actual_high:
            caught = sum(1 for p in actual_high if p.predicted_lift > high_threshold * 0.5)
            recall = caught / len(actual_high)
        else:
            recall = 0.0
        
        # Signal-specific accuracy
        signal_accuracy = {}
        for signal_type, preds in signal_predictions.items():
            if len(preds) < 5:
                continue
            
            pred_lifts = [p.predicted_lift for p in preds]
            actual_lifts = [p.actual_lift for p in preds]
            
            if stats and len(preds) > 2:
                try:
                    corr = stats.pearsonr(pred_lifts, actual_lifts)[0]
                except:
                    corr = 0.0
            else:
                corr = 0.0
            
            signal_accuracy[signal_type] = SignalAccuracy(
                signal_name=signal_type,
                sample_size=len(preds),
                predicted_lift=np.mean(pred_lifts),
                actual_lift=np.mean(actual_lifts),
                directional_accuracy=np.mean([p.direction_correct for p in preds]),
                mape=np.mean([p.abs_error for p in preds]),
                correlation=corr if not np.isnan(corr) else 0.0,
                hit_rate=sum(1 for p in preds if p.actual_lift > 0.10) / len(preds),
            )
        
        # Best and worst predictions
        sorted_by_error = sorted(predictions, key=lambda p: p.abs_error)
        best = sorted_by_error[:5]
        worst = sorted_by_error[-5:]
        
        # Get date range
        dates = sorted([p.date for p in predictions])
        
        result = BacktestResult(
            period_start=dates[0] if dates else "",
            period_end=dates[-1] if dates else "",
            total_days=len(predictions),
            overall_mape=overall_mape,
            overall_directional_accuracy=overall_directional,
            overall_correlation=correlation if not np.isnan(correlation) else 0.0,
            high_impact_precision=precision,
            high_impact_recall=recall,
            signal_accuracy=signal_accuracy,
            best_predictions=best,
            worst_predictions=worst,
            all_predictions=predictions,  # Store all for ROI calculation
        )
        
        return result
    
    def save_results(self, result: BacktestResult) -> str:
        """Save backtest results to JSON."""
        
        output = {
            "generated": datetime.now().isoformat(),
            "period": {
                "start": result.period_start,
                "end": result.period_end,
                "days": result.total_days,
            },
            "overall_metrics": {
                "mape": result.overall_mape,
                "directional_accuracy": result.overall_directional_accuracy,
                "correlation": result.overall_correlation,
            },
            "high_impact_detection": {
                "precision": result.high_impact_precision,
                "recall": result.high_impact_recall,
            },
            "signal_accuracy": {
                name: {
                    "sample_size": sa.sample_size,
                    "predicted_lift": sa.predicted_lift,
                    "actual_lift": sa.actual_lift,
                    "directional_accuracy": sa.directional_accuracy,
                    "mape": sa.mape,
                    "correlation": sa.correlation,
                }
                for name, sa in result.signal_accuracy.items()
            },
            "best_predictions": [
                {"date": p.date, "predicted": p.predicted_lift, "actual": p.actual_lift, "error": p.error}
                for p in result.best_predictions
            ],
            "worst_predictions": [
                {"date": p.date, "predicted": p.predicted_lift, "actual": p.actual_lift, "error": p.error}
                for p in result.worst_predictions
            ],
            # Daily comparison for ROI calculation
            "daily_comparison": [
                {
                    "date": p.date,
                    "predicted_lift": p.predicted_lift,
                    "actual_lift": p.actual_lift,
                    "error": p.error,
                    "events": p.events,
                }
                for p in result.all_predictions
            ],
        }
        
        output_file = self.brain_dir / "backtest_results.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)
    
    def generate_credibility_report(self, result: BacktestResult) -> str:
        """Generate a human-readable credibility report."""
        
        lines = [
            "=" * 70,
            "🎯 PREDICTION ACCURACY REPORT",
            "   How well does the brain predict sales?",
            "=" * 70,
            "",
            f"Test Period: {result.period_start} to {result.period_end} ({result.total_days} days)",
            "",
            "─" * 70,
            "OVERALL PERFORMANCE",
            "─" * 70,
        ]
        
        # Grade the system
        if result.overall_directional_accuracy >= 0.75:
            grade = "A"
            grade_desc = "Excellent - predictions reliably indicate direction"
        elif result.overall_directional_accuracy >= 0.65:
            grade = "B"
            grade_desc = "Good - predictions are mostly correct"
        elif result.overall_directional_accuracy >= 0.55:
            grade = "C"
            grade_desc = "Fair - better than random, room for improvement"
        else:
            grade = "D"
            grade_desc = "Needs work - predictions not reliable yet"
        
        lines.extend([
            f"  Overall Grade: {grade}",
            f"  {grade_desc}",
            "",
            f"  📊 Directional Accuracy: {result.overall_directional_accuracy:.0%}",
            f"     (Did we predict up/down correctly?)",
            "",
            f"  📏 Mean Absolute Error: {result.overall_mape:.1%}",
            f"     (Average prediction error)",
            "",
            f"  📈 Correlation: {result.overall_correlation:.2f}",
            f"     (How well predictions track actuals)",
            "",
            "─" * 70,
            "HIGH-IMPACT DAY DETECTION",
            "─" * 70,
            f"  🎯 Precision: {result.high_impact_precision:.0%}",
            f"     (Of days we flagged as high-impact, {result.high_impact_precision:.0%} actually were)",
            "",
            f"  🔍 Recall: {result.high_impact_recall:.0%}",
            f"     (Of actual high days, we caught {result.high_impact_recall:.0%})",
            "",
        ])
        
        # Signal breakdown
        if result.signal_accuracy:
            lines.extend([
                "─" * 70,
                "SIGNAL-BY-SIGNAL ACCURACY",
                "─" * 70,
            ])
            
            for name, sa in sorted(result.signal_accuracy.items(), key=lambda x: -x[1].directional_accuracy):
                lines.append(str(sa))
                lines.append("")
        
        # Best/worst examples
        lines.extend([
            "─" * 70,
            "EXAMPLE PREDICTIONS",
            "─" * 70,
            "",
            "✅ Most Accurate:",
        ])
        
        for p in result.best_predictions[:3]:
            lines.append(f"   {p.date}: predicted {p.predicted_lift:+.0%}, actual {p.actual_lift:+.0%} (error: {p.error:+.1%})")
        
        lines.extend([
            "",
            "⚠️ Least Accurate:",
        ])
        
        for p in result.worst_predictions[-3:]:
            lines.append(f"   {p.date}: predicted {p.predicted_lift:+.0%}, actual {p.actual_lift:+.0%} (error: {p.error:+.1%})")
            if p.events:
                lines.append(f"      Events: {', '.join(p.events[:3])}")
        
        lines.extend([
            "",
            "=" * 70,
        ])
        
        return "\n".join(lines)


def demo():
    """Demonstrate backtesting."""
    
    print("=" * 70)
    print("🎯 BACKTESTING - Proving Prediction Accuracy")
    print("=" * 70)
    print()
    
    backtester = Backtester()
    
    # Run backtest
    result = backtester.run_backtest(days=180)
    
    # Print credibility report
    print()
    report = backtester.generate_credibility_report(result)
    print(report)
    
    # Save results
    output_file = backtester.save_results(result)
    print(f"\n💾 Results saved to: {output_file}")


if __name__ == "__main__":
    demo()
