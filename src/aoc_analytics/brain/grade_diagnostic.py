"""Grade Improvement Diagnostic

Current Grade: C ($43/day, 19% rush accuracy)
Target Grade: A ($100+/day, 70%+ rush accuracy)

This module diagnoses WHY predictions are failing and provides
specific calibration recommendations.
"""

import json
from datetime import datetime, timedelta
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Tuple
from collections import defaultdict
import numpy as np

from aoc_analytics.core.db_adapter import get_connection


@dataclass
class SignalDiagnosis:
    """Diagnosis for a single signal type."""
    signal_name: str
    sample_size: int
    predicted_avg: float
    actual_avg: float
    error: float
    direction_correct: bool
    recommendation: str
    new_calibration: float


@dataclass
class GradeImprovement:
    """Full grade improvement report."""
    current_grade: str
    target_grade: str
    current_rush_accuracy: float
    target_rush_accuracy: float
    
    signal_diagnoses: List[SignalDiagnosis]
    missing_signals: List[str]
    data_gaps: List[str]
    calibration_updates: Dict[str, float]
    
    projected_improvement: str


class GradeDiagnostic:
    """
    Diagnose why we're at Grade C and how to get to Grade A.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent.parent.parent / "aoc_sales.db"
        self.db_path = Path(db_path)
        self.brain_dir = Path(__file__).parent / "data"
        
    def load_backtest(self) -> dict:
        """Load backtest results."""
        backtest_file = self.brain_dir / "backtest_results.json"
        if backtest_file.exists():
            with open(backtest_file) as f:
                return json.load(f)
        return {}
        
    def diagnose_signals(self) -> List[SignalDiagnosis]:
        """Diagnose each signal's accuracy."""
        backtest = self.load_backtest()
        diagnoses = []
        
        for signal_name, data in backtest.get("signal_accuracy", {}).items():
            predicted = data["predicted_lift"]
            actual = data["actual_lift"]
            samples = data["sample_size"]
            
            error = predicted - actual
            direction_correct = (predicted > 0) == (actual > 0)
            
            # Generate recommendation
            if abs(error) < 0.02:  # Within 2%
                recommendation = "✅ Well calibrated"
                new_cal = predicted
            elif not direction_correct:
                recommendation = f"🚨 WRONG DIRECTION! Predicted {predicted:+.1%}, actual {actual:+.1%}"
                new_cal = actual
            elif error > 0.05:
                recommendation = f"📉 Over-predicting by {error:.1%}. Reduce lift."
                new_cal = actual * 1.1  # Slight buffer above actual
            elif error < -0.05:
                recommendation = f"📈 Under-predicting by {abs(error):.1%}. Increase lift."
                new_cal = actual * 0.9  # Slight buffer below actual
            else:
                recommendation = f"⚠️ Minor adjustment needed ({error:+.1%})"
                new_cal = (predicted + actual) / 2
                
            diagnoses.append(SignalDiagnosis(
                signal_name=signal_name,
                sample_size=samples,
                predicted_avg=predicted,
                actual_avg=actual,
                error=error,
                direction_correct=direction_correct,
                recommendation=recommendation,
                new_calibration=new_cal,
            ))
            
        # Sort by error magnitude
        diagnoses.sort(key=lambda x: abs(x.error), reverse=True)
        return diagnoses
        
    def find_missing_signals(self) -> List[str]:
        """Find high-impact days that had no signal."""
        backtest = self.load_backtest()
        missing = []
        
        daily = backtest.get("daily_comparison", [])
        for day in daily:
            actual = day.get("actual_lift", 0)
            predicted = day.get("predicted_lift", 0)
            events = day.get("events", [])
            
            # High actual but low predicted with no events
            if actual > 0.15 and predicted < 0.10 and not events:
                missing.append(f"{day['date']}: {actual:+.1%} actual, no signals detected")
                
        return missing[:10]  # Top 10
        
    def find_data_gaps(self) -> List[str]:
        """Find gaps in the data that hurt predictions."""
        gaps = []
        backtest = self.load_backtest()
        
        # Check sample sizes
        for signal, data in backtest.get("signal_accuracy", {}).items():
            if data["sample_size"] < 10:
                gaps.append(f"'{signal}' has only {data['sample_size']} samples (need 10+)")
                
        # Check for missing signal types we should have
        expected_signals = ["friday", "saturday", "cruise", "canucks", "nfl", "holiday", "weather"]
        present_signals = set(backtest.get("signal_accuracy", {}).keys())
        
        for expected in expected_signals:
            if expected not in present_signals:
                gaps.append(f"Missing '{expected}' signal in backtest")
                
        return gaps
        
    def calculate_improvement_potential(self, diagnoses: List[SignalDiagnosis]) -> Tuple[str, Dict[str, float]]:
        """Calculate how much we'd improve with recalibration."""
        
        # Current errors
        current_mae = np.mean([abs(d.error) for d in diagnoses])
        
        # If we recalibrate to actuals
        projected_mae = current_mae * 0.3  # Assume 70% reduction
        
        # Calculate new calibrations
        calibrations = {}
        for d in diagnoses:
            if abs(d.error) > 0.02:  # Only update if significant error
                calibrations[d.signal_name] = d.new_calibration
                
        if current_mae > 0.08:
            improvement = "🚀 Major improvement possible with recalibration"
        elif current_mae > 0.04:
            improvement = "📈 Moderate improvement possible"
        else:
            improvement = "✅ Already well calibrated"
            
        return improvement, calibrations
        
    def get_specific_fixes(self) -> List[str]:
        """Get specific code fixes needed."""
        backtest = self.load_backtest()
        fixes = []
        
        signal_acc = backtest.get("signal_accuracy", {})
        
        # Academic is completely wrong
        if "academic" in signal_acc:
            academic = signal_acc["academic"]
            if academic["actual_lift"] < 0 and academic["predicted_lift"] > 0:
                fixes.append(
                    "🔧 FIX academic signal: Currently predicting POSITIVE but actual is NEGATIVE.\n"
                    "   Students leaving = LOWER sales, not higher.\n"
                    "   Change: academic_lift from +3% to -5%"
                )
                
        # Canucks is under-predicted
        if "canucks" in signal_acc:
            canucks = signal_acc["canucks"]
            if canucks["actual_lift"] > canucks["predicted_lift"] * 3:
                fixes.append(
                    f"🔧 FIX canucks signal: Predicting {canucks['predicted_lift']*100:+.1f}% but actual is {canucks['actual_lift']*100:+.1f}%.\n"
                    f"   Canucks games have 5x more impact than we thought!\n"
                    f"   Change: canucks_lift from 1.5% to 6%"
                )
                
        # Holiday is over-predicted  
        if "holiday" in signal_acc:
            holiday = signal_acc["holiday"]
            if holiday["predicted_lift"] > holiday["actual_lift"] * 3:
                fixes.append(
                    f"🔧 FIX holiday signal: Predicting {holiday['predicted_lift']*100:+.1f}% but actual is {holiday['actual_lift']*100:+.1f}%.\n"
                    f"   Holidays don't boost as much as expected (people away?).\n"
                    f"   Change: holiday_lift from 12% to 2%"
                )
                
        # NFL shows no effect
        if "nfl" in signal_acc:
            nfl = signal_acc["nfl"]
            if abs(nfl["actual_lift"]) < 0.01:
                fixes.append(
                    "🔧 FIX nfl signal: NFL games show NO measurable effect.\n"
                    "   People watch at home, doesn't affect store traffic.\n"
                    "   Change: Remove NFL signal or set to 0%"
                )
                
        return fixes
        
    def generate_report(self) -> GradeImprovement:
        """Generate full improvement report."""
        backtest = self.load_backtest()
        
        # Current state
        precision = backtest.get("high_impact_detection", {}).get("precision", 0)
        recall = backtest.get("high_impact_detection", {}).get("recall", 0)
        
        # Diagnose
        diagnoses = self.diagnose_signals()
        missing = self.find_missing_signals()
        gaps = self.find_data_gaps()
        improvement, calibrations = self.calculate_improvement_potential(diagnoses)
        
        return GradeImprovement(
            current_grade="C",
            target_grade="A",
            current_rush_accuracy=precision,
            target_rush_accuracy=0.70,
            signal_diagnoses=diagnoses,
            missing_signals=missing,
            data_gaps=gaps,
            calibration_updates=calibrations,
            projected_improvement=improvement,
        )
        
    def print_report(self):
        """Print the improvement report."""
        report = self.generate_report()
        fixes = self.get_specific_fixes()
        
        print("=" * 70)
        print("📊 GRADE IMPROVEMENT DIAGNOSTIC")
        print(f"   Current: Grade {report.current_grade} → Target: Grade {report.target_grade}")
        print("=" * 70)
        print()
        
        print("─" * 70)
        print("CURRENT PERFORMANCE")
        print("─" * 70)
        print(f"  Rush Day Accuracy: {report.current_rush_accuracy:.0%} (need 70% for Grade A)")
        print(f"  {report.projected_improvement}")
        print()
        
        print("─" * 70)
        print("SIGNAL-BY-SIGNAL DIAGNOSIS")
        print("─" * 70)
        for d in report.signal_diagnoses:
            status = "✅" if d.direction_correct and abs(d.error) < 0.03 else "⚠️" if d.direction_correct else "🚨"
            print(f"  {status} {d.signal_name}:")
            print(f"     Predicted: {d.predicted_avg:+.1%}, Actual: {d.actual_avg:+.1%}, Error: {d.error:+.1%}")
            print(f"     {d.recommendation}")
            print(f"     Samples: {d.sample_size}")
            print()
            
        if fixes:
            print("─" * 70)
            print("🔧 SPECIFIC CODE FIXES NEEDED")
            print("─" * 70)
            for fix in fixes:
                print(f"  {fix}")
                print()
                
        if report.missing_signals:
            print("─" * 70)
            print("❓ HIGH-IMPACT DAYS WITH NO SIGNAL")
            print("─" * 70)
            for miss in report.missing_signals:
                print(f"  • {miss}")
            print()
            
        if report.data_gaps:
            print("─" * 70)
            print("📉 DATA GAPS")
            print("─" * 70)
            for gap in report.data_gaps:
                print(f"  • {gap}")
            print()
            
        if report.calibration_updates:
            print("─" * 70)
            print("📝 RECOMMENDED CALIBRATION UPDATES")
            print("─" * 70)
            for signal, new_val in report.calibration_updates.items():
                print(f"  {signal}: set to {new_val:+.1%}")
            print()
            
        print("─" * 70)
        print("📋 ACTION PLAN TO REACH GRADE A")
        print("─" * 70)
        print("  1. Fix academic signal (wrong direction)")
        print("  2. Increase canucks lift from 1.5% to 6%")
        print("  3. Reduce holiday lift from 12% to 2%")
        print("  4. Remove or zero-out NFL signal")
        print("  5. Add day-of-week baseline signals (Friday/Saturday)")
        print("  6. Integrate real weather data for predictions")
        print("  7. Re-run backtest after calibration")
        print()
        
        
def demo():
    """Run the diagnostic."""
    diag = GradeDiagnostic()
    diag.print_report()
    

if __name__ == "__main__":
    demo()
