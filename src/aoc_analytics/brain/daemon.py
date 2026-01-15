"""
Continuous Learning Daemon

A background process that runs nightly to:
1. Update signal magnitude learning
2. Update product-weather impacts
3. Update time-of-day patterns
4. Update customer profiles
5. Generate alerts for new patterns

Can be run via cron:
    0 3 * * * cd /path/to/aoc-analytics && python -m aoc_analytics.brain.daemon

Or as a systemd service for continuous monitoring.
"""

import json
import logging
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(),
    ]
)
logger = logging.getLogger(__name__)


class LearningDaemon:
    """
    Orchestrates all learning modules and runs them in sequence.
    """
    
    def __init__(self, db_path: str = None, output_dir: str = None):
        self.db_path = db_path
        
        if output_dir is None:
            output_dir = Path(__file__).parent / "data"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        self.run_log = {
            "started": str(datetime.now()),
            "completed": None,
            "modules": {},
            "errors": [],
            "alerts": [],
        }
    
    def run_signal_magnitude(self) -> Dict[str, Any]:
        """Run signal magnitude learning."""
        logger.info("📊 Running signal magnitude learning...")
        
        try:
            from aoc_analytics.brain.signal_magnitude import SignalMagnitudeLearner
            
            learner = SignalMagnitudeLearner(self.db_path)
            output = learner.save_learned_magnitudes()
            
            # Extract actionable signals
            actionable = {k: v for k, v in output.items() if v.get("actionable")}
            
            result = {
                "status": "success",
                "signals_learned": len(output),
                "actionable_signals": len(actionable),
                "top_signals": [
                    {"name": v["signal_name"], "lift": v["lift_pct"]}
                    for k, v in sorted(actionable.items(), key=lambda x: -abs(x[1]["lift"]))[:3]
                ]
            }
            
            logger.info(f"   ✓ Learned {len(output)} signals, {len(actionable)} actionable")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Signal magnitude failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_product_weather(self) -> Dict[str, Any]:
        """Run product-weather impact learning."""
        logger.info("🌦️  Running product-weather learning...")
        
        try:
            from aoc_analytics.brain.product_weather import ProductWeatherLearner
            
            learner = ProductWeatherLearner(self.db_path)
            results = learner.learn_all_category_weather_impacts()
            learner.save_results(results)
            
            # Count actionable
            actionable = learner.get_actionable_insights(results)
            
            result = {
                "status": "success",
                "categories_analyzed": len(results),
                "actionable_insights": len(actionable),
                "top_insights": [
                    {"category": imp.category, "condition": imp.weather_condition, "lift": f"{imp.avg_lift:+.1%}"}
                    for name, imp in actionable[:3]
                ]
            }
            
            logger.info(f"   ✓ {len(results)} categories, {len(actionable)} actionable insights")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Product-weather failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_time_of_day(self) -> Dict[str, Any]:
        """Run time-of-day pattern learning."""
        logger.info("🕐 Running time-of-day learning...")
        
        try:
            from aoc_analytics.brain.time_of_day import TimeOfDayLearner
            
            learner = TimeOfDayLearner(self.db_path)
            hourly = learner.learn_hourly_patterns()
            day_hour = learner.learn_day_hour_interactions()
            learner.save_patterns(hourly, day_hour)
            
            # Find peaks
            peaks = learner.find_peak_hours(hourly, top_n=3)
            
            result = {
                "status": "success",
                "hours_analyzed": len(hourly),
                "peak_hours": [f"{p.hour}:00 ({p.pct_of_daily:.1%})" for p in peaks]
            }
            
            logger.info(f"   ✓ {len(hourly)} hours, peaks at {[p.hour for p in peaks]}")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Time-of-day failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_event_correlation(self) -> Dict[str, Any]:
        """Run event-sales correlation analysis."""
        logger.info("🔗 Running event-sales correlation...")
        
        try:
            from aoc_analytics.brain.event_correlation import EventImpactAnalyzer
            
            analyzer = EventImpactAnalyzer(self.db_path)
            
            # Detect anomalies and load events
            anomalies = analyzer.detect_anomalies()
            signal_events = analyzer.load_signal_events()
            
            # Correlate events with anomalies (returns Dict[str, List[float]])
            event_impacts = analyzer.correlate_events_with_anomalies(anomalies, signal_events)
            
            # Run statistical analysis
            correlations = analyzer.analyze_event_correlations(event_impacts)
            
            # Get significant correlations
            sig_events = [e for e in correlations if e.p_value < 0.05]
            
            # Save results
            reddit_events = analyzer.load_reddit_events()
            explanations = analyzer.get_anomaly_explanations(anomalies, signal_events, reddit_events)
            output = analyzer.save_analysis(anomalies, correlations, explanations)
            
            result = {
                "status": "success",
                "events_analyzed": len(correlations),
                "significant_correlations": len(sig_events),
                "top_impacts": [
                    {"event": e.event_type, "lift": f"{e.avg_impact:+.1%}", "p": f"{e.p_value:.3f}"}
                    for e in sorted(sig_events, key=lambda x: -abs(x.avg_impact))[:3]
                ]
            }
            
            logger.info(f"   ✓ {len(correlations)} events, {len(sig_events)} significant")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Event correlation failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_predictive_calendar(self) -> Dict[str, Any]:
        """Generate forward-looking predictive calendar."""
        logger.info("📅 Generating predictive calendar...")
        
        try:
            from aoc_analytics.brain.predictive_calendar import PredictiveCalendar
            
            calendar = PredictiveCalendar()
            forecast = calendar.build_calendar(days_ahead=14)
            output = calendar.save_calendar(forecast)
            
            # Count high-impact days (using combined_lift)
            high_impact = [f for f in forecast if f.combined_lift > 0.15]
            
            result = {
                "status": "success",
                "days_forecasted": len(forecast),
                "high_impact_days": len(high_impact),
                "top_days": [
                    {"date": str(f.date), "lift": f"{f.combined_lift:+.0%}", "events": len(f.events)}
                    for f in sorted(forecast, key=lambda x: -x.combined_lift)[:3]
                ]
            }
            
            logger.info(f"   ✓ {len(forecast)} days, {len(high_impact)} high-impact")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Predictive calendar failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_inventory_recommendations(self) -> Dict[str, Any]:
        """Generate inventory recommendations."""
        logger.info("📦 Generating inventory recommendations...")
        
        try:
            from aoc_analytics.brain.inventory_recommendations import InventoryRecommender
            
            recommender = InventoryRecommender()
            recs = recommender.generate_week_recommendations()
            output = recommender.save_recommendations(recs)
            
            # Get priority items
            priority = recommender.get_priority_items(recs)
            high_impact_days = [r for r in recs if r.overall_lift > 0.15]
            
            result = {
                "status": "success",
                "days_covered": len(recs),
                "high_impact_days": len(high_impact_days),
                "priority_items": len(priority),
                "top_priorities": [
                    {"date": d, "category": c.category, "lift": f"{c.expected_lift:+.0%}"}
                    for d, _, c in priority[:3]
                ]
            }
            
            logger.info(f"   ✓ {len(recs)} days, {len(priority)} priority items")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Inventory recommendations failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_llm_synthesis(self) -> Dict[str, Any]:
        """Generate LLM-powered summaries."""
        logger.info("🧠 Running LLM synthesis...")
        
        try:
            from aoc_analytics.brain.llm_synthesis import LLMSynthesizer
            
            synth = LLMSynthesizer()
            
            # Generate executive summary
            summary = synth.synthesize_executive_summary()
            synth.save_synthesis(summary, "executive_summary")
            
            # Generate daily briefing
            briefing = synth.synthesize_daily_briefing()
            synth.save_synthesis(briefing, "daily_briefing")
            
            result = {
                "status": "success",
                "ollama_available": synth.ollama_available,
                "model": synth.model if synth.ollama_available else "template",
                "summaries_generated": 2,
            }
            
            logger.info(f"   ✓ 2 summaries generated ({'LLM' if synth.ollama_available else 'template'})")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ LLM synthesis failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_pattern_alerts(self) -> Dict[str, Any]:
        """Check for new pattern alerts."""
        logger.info("🚨 Checking pattern alerts...")
        
        try:
            # Try to import pattern monitor
            from aoc_analytics.scripts.pattern_alerts import PatternMonitor
            
            monitor = PatternMonitor()
            alerts = monitor.scan_for_new_patterns(lookback_days=30)
            
            # Filter to important alerts
            important = [a for a in alerts if a.significance > 0.7]
            
            result = {
                "status": "success",
                "total_alerts": len(alerts),
                "important_alerts": len(important),
                "alerts": [
                    {"type": a.alert_type.value, "message": a.message}
                    for a in important[:5]
                ]
            }
            
            # Store alerts for later
            self.run_log["alerts"].extend(result["alerts"])
            
            logger.info(f"   ✓ {len(alerts)} alerts, {len(important)} important")
            return result
            
        except ImportError:
            logger.info("   ~ Pattern alerts module not available")
            return {"status": "skipped", "reason": "module not available"}
        except Exception as e:
            logger.error(f"   ✗ Pattern alerts failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_roi_tracker(self) -> Dict[str, Any]:
        """Calculate ROI from predictions."""
        logger.info("💰 Running ROI tracker...")
        
        try:
            from aoc_analytics.brain.roi_tracker import ROITracker
            
            tracker = ROITracker(self.db_path)
            summary = tracker.generate_roi_summary()
            
            result = {
                "status": "success",
                "grade": summary["historical_performance"]["grade"],
                "value_per_month": summary["historical_performance"]["value_per_month"],
                "days_analyzed": summary["historical_performance"]["days_analyzed"],
            }
            
            logger.info(f"   ✓ Grade {result['grade']}, {result['value_per_month']}/month")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ ROI tracker failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_notifications(self) -> Dict[str, Any]:
        """Send alerts for high-impact days."""
        logger.info("🔔 Running notification service...")
        
        try:
            from aoc_analytics.brain.notifications import NotificationService
            
            service = NotificationService()
            alerts = service.generate_alerts()
            
            # Only send if there are high-priority alerts
            high_priority = [a for a in alerts if a.priority == "high"]
            
            result = {
                "status": "success",
                "alerts_generated": len(alerts),
                "high_priority_alerts": len(high_priority),
                "notifications_sent": False,
            }
            
            # Only actually send if configured and there are high-priority alerts
            if high_priority and (service.slack_webhook or service.email_user):
                send_results = service.send_all(alerts)
                result["slack_sent"] = send_results.get("slack_sent", False)
                result["email_sent"] = send_results.get("email_sent", False)
                result["notifications_sent"] = result["slack_sent"] or result["email_sent"]
            
            logger.info(f"   ✓ {len(alerts)} alerts ({len(high_priority)} high priority)")
            return result
            
        except Exception as e:
            logger.error(f"   ✗ Notifications failed: {e}")
            return {"status": "error", "error": str(e)}
    
    def run_all(self) -> Dict[str, Any]:
        """Run all learning modules."""
        
        logger.info("=" * 60)
        logger.info("🧠 AOC BRAIN - CONTINUOUS LEARNING RUN")
        logger.info(f"   Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 60)
        
        self.run_log["started"] = str(datetime.now())
        
        # Run each module
        self.run_log["modules"]["signal_magnitude"] = self.run_signal_magnitude()
        self.run_log["modules"]["product_weather"] = self.run_product_weather()
        self.run_log["modules"]["time_of_day"] = self.run_time_of_day()
        self.run_log["modules"]["event_correlation"] = self.run_event_correlation()
        self.run_log["modules"]["predictive_calendar"] = self.run_predictive_calendar()
        self.run_log["modules"]["inventory_recommendations"] = self.run_inventory_recommendations()
        self.run_log["modules"]["llm_synthesis"] = self.run_llm_synthesis()
        self.run_log["modules"]["pattern_alerts"] = self.run_pattern_alerts()
        self.run_log["modules"]["roi_tracker"] = self.run_roi_tracker()
        self.run_log["modules"]["notifications"] = self.run_notifications()
        
        # Finalize
        self.run_log["completed"] = str(datetime.now())
        
        # Count successes/failures
        statuses = [m.get("status") for m in self.run_log["modules"].values()]
        successes = statuses.count("success")
        errors = statuses.count("error")
        
        logger.info("")
        logger.info("=" * 60)
        logger.info(f"✅ LEARNING RUN COMPLETE")
        logger.info(f"   Modules: {successes} succeeded, {errors} failed, {len(statuses) - successes - errors} skipped")
        logger.info(f"   Alerts: {len(self.run_log['alerts'])} important alerts")
        logger.info("=" * 60)
        
        # Save run log
        self._save_run_log()
        
        return self.run_log
    
    def _save_run_log(self):
        """Save the run log to JSON."""
        
        log_file = self.output_dir / "last_run.json"
        with open(log_file, "w") as f:
            json.dump(self.run_log, f, indent=2)
        
        # Also append to history
        history_file = self.output_dir / "run_history.jsonl"
        with open(history_file, "a") as f:
            f.write(json.dumps({
                "timestamp": self.run_log["started"],
                "modules": {k: v.get("status") for k, v in self.run_log["modules"].items()},
                "alerts": len(self.run_log["alerts"]),
            }) + "\n")
        
        logger.info(f"💾 Log saved to: {log_file}")
    
    @staticmethod
    def get_last_run() -> Optional[Dict[str, Any]]:
        """Get the last run log."""
        
        log_file = Path(__file__).parent / "data" / "last_run.json"
        if not log_file.exists():
            return None
        
        with open(log_file) as f:
            return json.load(f)
    
    @staticmethod
    def get_run_history(limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent run history."""
        
        history_file = Path(__file__).parent / "data" / "run_history.jsonl"
        if not history_file.exists():
            return []
        
        runs = []
        with open(history_file) as f:
            for line in f:
                if line.strip():
                    runs.append(json.loads(line))
        
        return runs[-limit:]


def main():
    """Run the learning daemon."""
    
    import argparse
    
    parser = argparse.ArgumentParser(description="AOC Brain Continuous Learning Daemon")
    parser.add_argument("--db", type=str, help="Path to aoc_sales.db")
    parser.add_argument("--output", type=str, help="Output directory for learned data")
    parser.add_argument("--status", action="store_true", help="Show last run status")
    parser.add_argument("--history", action="store_true", help="Show run history")
    
    args = parser.parse_args()
    
    if args.status:
        last_run = LearningDaemon.get_last_run()
        if last_run:
            print(f"Last run: {last_run['started']}")
            print(f"Completed: {last_run['completed']}")
            for module, result in last_run.get("modules", {}).items():
                status = result.get("status", "unknown")
                print(f"  {module}: {status}")
            if last_run.get("alerts"):
                print(f"\nAlerts ({len(last_run['alerts'])}):")
                for alert in last_run["alerts"][:5]:
                    print(f"  - [{alert['type']}] {alert['message']}")
        else:
            print("No previous run found.")
        return
    
    if args.history:
        history = LearningDaemon.get_run_history()
        print(f"Last {len(history)} runs:")
        for run in reversed(history):
            statuses = run.get("modules", {})
            ok = sum(1 for s in statuses.values() if s == "success")
            print(f"  {run['timestamp']}: {ok}/{len(statuses)} modules OK, {run.get('alerts', 0)} alerts")
        return
    
    # Run the daemon
    daemon = LearningDaemon(db_path=args.db, output_dir=args.output)
    daemon.run_all()


if __name__ == "__main__":
    main()
