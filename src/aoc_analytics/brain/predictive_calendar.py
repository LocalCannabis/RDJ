"""
Predictive Event Calendar

Builds a forward-looking calendar of upcoming events with
predicted sales impact based on historical correlations.

This is the "actionable output" of the event system:
- What events are coming up?
- How much should we expect sales to change?
- What should we stock/staff for?
"""

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from enum import Enum


class ImpactLevel(Enum):
    """Predicted impact level."""
    HIGH_POSITIVE = "high_positive"      # > +20%
    MODERATE_POSITIVE = "mod_positive"   # +10% to +20%
    SLIGHT_POSITIVE = "slight_positive"  # +5% to +10%
    NEUTRAL = "neutral"                  # -5% to +5%
    SLIGHT_NEGATIVE = "slight_negative"  # -5% to -10%
    MODERATE_NEGATIVE = "mod_negative"   # -10% to -20%
    HIGH_NEGATIVE = "high_negative"      # < -20%


@dataclass
class CalendarEvent:
    """An event on the predictive calendar."""
    date: str
    event_name: str
    event_type: str
    source: str  # "vibe_signals", "reddit", "manual"
    predicted_lift: float  # as decimal (0.15 = +15%)
    confidence: float  # 0-1
    impact_level: ImpactLevel
    notes: str = ""
    
    def __str__(self) -> str:
        emoji = {
            ImpactLevel.HIGH_POSITIVE: "🚀",
            ImpactLevel.MODERATE_POSITIVE: "📈",
            ImpactLevel.SLIGHT_POSITIVE: "↗️",
            ImpactLevel.NEUTRAL: "➡️",
            ImpactLevel.SLIGHT_NEGATIVE: "↘️",
            ImpactLevel.MODERATE_NEGATIVE: "📉",
            ImpactLevel.HIGH_NEGATIVE: "⚠️",
        }
        return (
            f"{emoji.get(self.impact_level, '•')} {self.date} | {self.event_name}\n"
            f"   Predicted: {self.predicted_lift:+.0%} ({self.impact_level.value})\n"
            f"   Source: {self.source}, Confidence: {self.confidence:.0%}"
        )


@dataclass 
class DayForecast:
    """Forecast for a single day."""
    date: str
    day_name: str
    events: List[CalendarEvent]
    combined_lift: float  # Combined effect of all events
    baseline_multiplier: float  # 1.0 = normal day
    staffing_recommendation: str
    stocking_notes: List[str]
    
    def __str__(self) -> str:
        events_str = f" ({len(self.events)} events)" if self.events else ""
        return (
            f"{self.day_name} {self.date}{events_str}\n"
            f"   Expected: {self.combined_lift:+.0%} vs baseline\n"
            f"   Staffing: {self.staffing_recommendation}"
        )


class PredictiveCalendar:
    """
    Builds a forward-looking calendar with predicted sales impact.
    """
    
    DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    # Learned impact magnitudes (from event_correlation + signal_magnitude)
    # These are the validated impacts with statistical significance
    KNOWN_IMPACTS = {
        "cruise_ships": {"lift": 0.36, "confidence": 0.99},
        "academic_finals": {"lift": 0.30, "confidence": 0.95},
        "academic_midterms": {"lift": 0.25, "confidence": 0.90},
        "canucks_home": {"lift": 0.22, "confidence": 0.85},
        "whitecaps_home": {"lift": 0.07, "confidence": 0.96},
        "nfl_sunday": {"lift": 0.15, "confidence": 0.70},
        "nfl_super_bowl": {"lift": 0.25, "confidence": 0.80},
        "friday": {"lift": 0.15, "confidence": 0.99},  # Friday baseline is higher
        "academic_movein": {"lift": -0.06, "confidence": 0.93},
        "cold_weather": {"lift": -0.12, "confidence": 0.95},  # For Pre-Rolls
    }
    
    def __init__(self):
        # Try to load learned impacts
        self._load_learned_impacts()
    
    def _load_learned_impacts(self):
        """Load impacts from brain data files."""
        
        brain_dir = Path(__file__).parent / "data"
        
        # Load signal magnitudes
        sig_file = brain_dir / "learned_signal_magnitudes.json"
        if sig_file.exists():
            with open(sig_file) as f:
                data = json.load(f)
                for key, signal in data.get("signals", {}).items():
                    if signal.get("actionable"):
                        self.KNOWN_IMPACTS[key] = {
                            "lift": signal["lift"],
                            "confidence": signal["confidence"],
                        }
        
        # Load event correlations
        corr_file = brain_dir / "event_impact_analysis.json"
        if corr_file.exists():
            with open(corr_file) as f:
                data = json.load(f)
                for corr in data.get("correlations", []):
                    if corr.get("significant"):
                        self.KNOWN_IMPACTS[corr["event_type"]] = {
                            "lift": corr["avg_impact_pct"] / 100,
                            "confidence": 1 - corr["p_value"],
                        }
    
    def _classify_impact(self, lift: float) -> ImpactLevel:
        """Classify lift into impact level."""
        if lift > 0.20:
            return ImpactLevel.HIGH_POSITIVE
        elif lift > 0.10:
            return ImpactLevel.MODERATE_POSITIVE
        elif lift > 0.05:
            return ImpactLevel.SLIGHT_POSITIVE
        elif lift >= -0.05:
            return ImpactLevel.NEUTRAL
        elif lift >= -0.10:
            return ImpactLevel.SLIGHT_NEGATIVE
        elif lift >= -0.20:
            return ImpactLevel.MODERATE_NEGATIVE
        else:
            return ImpactLevel.HIGH_NEGATIVE
    
    def _get_staffing_recommendation(self, lift: float, events: List[CalendarEvent]) -> str:
        """Generate staffing recommendation."""
        if lift > 0.30:
            return "🔴 FULL STAFF - Major event day"
        elif lift > 0.15:
            return "🟠 Extra staff recommended"
        elif lift > 0.05:
            return "🟡 Normal+ (add 1 if possible)"
        elif lift >= -0.05:
            return "🟢 Normal staffing"
        elif lift >= -0.15:
            return "🔵 Can reduce by 1"
        else:
            return "⚪ Minimal staffing OK"
    
    def _get_stocking_notes(self, events: List[CalendarEvent]) -> List[str]:
        """Generate stocking recommendations based on events."""
        notes = []
        
        for event in events:
            if "cruise" in event.event_type:
                notes.append("Stock up on tourist favorites (pre-rolls, gummies)")
            if "academic" in event.event_type and "finals" in event.event_name.lower():
                notes.append("Students: edibles, vapes, budget options")
            if "canucks" in event.event_type or "whitecaps" in event.event_type:
                notes.append("Game day: pre-rolls, beverages, quick purchases")
            if "nfl" in event.event_type:
                notes.append("NFL Sunday: expect evening lull during games")
            if event.event_type == "statutory_holiday":
                if "New Year" in event.event_name:
                    notes.append("🎉 NYE/NYD: party packs, edibles, beverages, pre-rolls")
                elif "Canada Day" in event.event_name:
                    notes.append("🍁 Canada Day: outdoor-friendly (pre-rolls, beverages)")
                elif "Christmas" in event.event_name or "Boxing" in event.event_name:
                    notes.append("🎄 Holiday gifting: accessories, premium products")
                else:
                    notes.append("📅 Stat holiday: expect extended browsing")
            if event.event_type == "pre_holiday":
                notes.append("📦 Pre-holiday stock-up rush expected")
        
        return list(set(notes))  # Dedupe
    
    def get_upcoming_signal_events(self, days_ahead: int = 14) -> List[CalendarEvent]:
        """Get events from vibe_signals for upcoming days."""
        
        try:
            from aoc_analytics.core.signals.vibe_signals import (
                CruiseSchedule,
                SportsSchedule,
                AcademicCalendar,
            )
        except ImportError:
            CruiseSchedule = None
            SportsSchedule = None
            AcademicCalendar = None
        
        # Import BC holidays
        try:
            from aoc_analytics.core.calendar import (
                BC_STATUTORY_HOLIDAYS,
                PARTY_HOLIDAYS,
                GIFTING_HOLIDAYS,
            )
        except ImportError:
            BC_STATUTORY_HOLIDAYS = {}
            PARTY_HOLIDAYS = set()
            GIFTING_HOLIDAYS = set()
        
        events = []
        today = date.today()
        
        for i in range(days_ahead):
            check_date = today + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            
            # Check BC statutory holidays
            if check_date in BC_STATUTORY_HOLIDAYS:
                holiday_name = BC_STATUTORY_HOLIDAYS[check_date]
                
                # Different impacts for different holiday types
                if holiday_name in PARTY_HOLIDAYS:
                    # Party holidays (New Year's, Canada Day, etc.) - big boost
                    lift = 0.25
                    confidence = 0.90
                    notes = "Party holiday: edibles, pre-rolls, beverages"
                elif holiday_name in GIFTING_HOLIDAYS:
                    # Gifting holidays (Christmas, Mother's Day, etc.)
                    lift = 0.20
                    confidence = 0.85
                    notes = "Gifting holiday: accessories, premium products"
                else:
                    # Other stat holidays - day off = more shopping time
                    lift = 0.15
                    confidence = 0.80
                    notes = "Stat holiday: extended browse time"
                
                events.append(CalendarEvent(
                    date=date_str,
                    event_name=f"🎉 {holiday_name}",
                    event_type="statutory_holiday",
                    source="bc_calendar",
                    predicted_lift=lift,
                    confidence=confidence,
                    impact_level=self._classify_impact(lift),
                    notes=notes,
                ))
            
            # Check for pre-holiday (day before a stat holiday)
            next_day = check_date + timedelta(days=1)
            if next_day in BC_STATUTORY_HOLIDAYS:
                holiday_name = BC_STATUTORY_HOLIDAYS[next_day]
                # Day before holiday often sees stock-up behavior
                events.append(CalendarEvent(
                    date=date_str,
                    event_name=f"📅 Day Before {holiday_name}",
                    event_type="pre_holiday",
                    source="bc_calendar",
                    predicted_lift=0.18,
                    confidence=0.85,
                    impact_level=self._classify_impact(0.18),
                    notes="Pre-holiday stock-up behavior",
                ))
            
            # Cruise ships
            if CruiseSchedule and CruiseSchedule.is_cruise_season(check_date):
                likelihood = CruiseSchedule.estimate_cruise_likelihood(check_date)
                if likelihood > 0.3:
                    impact = self.KNOWN_IMPACTS.get("cruise_ships", {"lift": 0.36, "confidence": 0.95})
                    adjusted_lift = impact["lift"] * likelihood
                    events.append(CalendarEvent(
                        date=date_str,
                        event_name=f"Cruise Ships (likelihood: {likelihood:.0%})",
                        event_type="cruise_ships",
                        source="vibe_signals",
                        predicted_lift=adjusted_lift,
                        confidence=impact["confidence"],
                        impact_level=self._classify_impact(adjusted_lift),
                    ))
            
            # Canucks
            if SportsSchedule:
                is_game, prob = SportsSchedule.is_likely_canucks_home_game(check_date)
                if is_game and prob > 0.20:
                    impact = self.KNOWN_IMPACTS.get("canucks_home", {"lift": 0.22, "confidence": 0.85})
                    adjusted_lift = impact["lift"] * prob
                    events.append(CalendarEvent(
                        date=date_str,
                        event_name=f"Canucks Home Game (prob: {prob:.0%})",
                        event_type="canucks_home",
                        source="vibe_signals",
                        predicted_lift=adjusted_lift,
                        confidence=impact["confidence"] * prob,
                        impact_level=self._classify_impact(adjusted_lift),
                    ))
            
            # NFL
            if SportsSchedule:
                is_nfl, game_type = SportsSchedule.is_nfl_game_window(check_date)
                if is_nfl and game_type:
                    impact_key = "nfl_super_bowl" if game_type == "super_bowl" else "nfl_sunday"
                    impact = self.KNOWN_IMPACTS.get(impact_key, {"lift": 0.15, "confidence": 0.70})
                    events.append(CalendarEvent(
                        date=date_str,
                        event_name=f"NFL: {game_type.replace('_', ' ').title()}",
                        event_type=impact_key,
                        source="vibe_signals",
                        predicted_lift=impact["lift"],
                        confidence=impact["confidence"],
                        impact_level=self._classify_impact(impact["lift"]),
                    ))
            
            # Academic calendar
            if AcademicCalendar:
                period, stress = AcademicCalendar.get_academic_period(check_date)
                if stress > 0.5:
                    if "finals" in period:
                        impact = self.KNOWN_IMPACTS.get("academic_finals", {"lift": 0.30, "confidence": 0.95})
                        event_name = "Academic: Finals Week"
                    elif "midterms" in period:
                        impact = self.KNOWN_IMPACTS.get("academic_midterms", {"lift": 0.25, "confidence": 0.90})
                        event_name = "Academic: Midterms"
                    elif "start" in period:
                        impact = self.KNOWN_IMPACTS.get("academic_movein", {"lift": -0.06, "confidence": 0.93})
                        event_name = "Academic: Move-in Week"
                    else:
                        continue
                    
                    events.append(CalendarEvent(
                        date=date_str,
                        event_name=event_name,
                        event_type=f"academic_{period}",
                        source="vibe_signals",
                        predicted_lift=impact["lift"],
                        confidence=impact["confidence"],
                        impact_level=self._classify_impact(impact["lift"]),
                    ))
        
        return events
    
    def get_reddit_events(self, days_ahead: int = 7) -> List[CalendarEvent]:
        """Get events from Reddit scan."""
        
        events_file = Path(__file__).parent / "data" / "reddit_events.json"
        if not events_file.exists():
            return []
        
        with open(events_file) as f:
            data = json.load(f)
        
        events = []
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        
        for event in data.get("events", []):
            if not event.get("event_date"):
                continue
            
            try:
                event_dt = datetime.strptime(event["event_date"], "%Y-%m-%d").date()
            except ValueError:
                continue
            
            if not (today <= event_dt <= cutoff):
                continue
            
            # Estimate impact based on event type and score
            base_lift = 0.05  # Default small impact
            if event.get("type") == "concert":
                base_lift = 0.10
            elif event.get("type") == "festival":
                base_lift = 0.15
            elif event.get("type") == "road_closure":
                base_lift = -0.05
            
            # Adjust by Reddit score (higher engagement = more impact)
            score = event.get("score", 0)
            if score > 500:
                base_lift *= 1.5
            elif score > 100:
                base_lift *= 1.2
            
            events.append(CalendarEvent(
                date=event["event_date"],
                event_name=event.get("title", "")[:50],
                event_type=f"reddit_{event.get('type', 'other')}",
                source="reddit",
                predicted_lift=base_lift,
                confidence=min(0.5, score / 500),  # Low confidence for Reddit
                impact_level=self._classify_impact(base_lift),
                notes=event.get("summary", "")[:100],
            ))
        
        return events
    
    def build_calendar(self, days_ahead: int = 14) -> List[DayForecast]:
        """Build the full predictive calendar."""
        
        # Gather all events
        signal_events = self.get_upcoming_signal_events(days_ahead)
        reddit_events = self.get_reddit_events(min(days_ahead, 7))
        
        all_events = signal_events + reddit_events
        
        # Group by date
        events_by_date = {}
        for event in all_events:
            if event.date not in events_by_date:
                events_by_date[event.date] = []
            events_by_date[event.date].append(event)
        
        # Build forecast for each day
        forecasts = []
        today = date.today()
        
        for i in range(days_ahead):
            check_date = today + timedelta(days=i)
            date_str = check_date.strftime("%Y-%m-%d")
            day_name = self.DAY_NAMES[check_date.weekday()]
            
            day_events = events_by_date.get(date_str, [])
            
            # Calculate combined lift
            # For multiple events, use a dampened combination (not purely additive)
            if day_events:
                lifts = [e.predicted_lift for e in day_events]
                # Primary event gets full weight, others are dampened
                lifts_sorted = sorted(lifts, key=abs, reverse=True)
                combined = lifts_sorted[0]
                for lift in lifts_sorted[1:]:
                    combined += lift * 0.5  # 50% of additional events
            else:
                combined = 0.0
            
            # Add Friday bonus if applicable
            if day_name == "Friday":
                combined += 0.10  # Fridays are typically +10%
            
            forecasts.append(DayForecast(
                date=date_str,
                day_name=day_name,
                events=day_events,
                combined_lift=combined,
                baseline_multiplier=1.0 + combined,
                staffing_recommendation=self._get_staffing_recommendation(combined, day_events),
                stocking_notes=self._get_stocking_notes(day_events),
            ))
        
        return forecasts
    
    def save_calendar(self, forecasts: List[DayForecast]) -> str:
        """Save calendar to JSON."""
        
        output = {
            "generated": str(datetime.now()),
            "days_ahead": len(forecasts),
            "forecasts": [
                {
                    "date": f.date,
                    "day_name": f.day_name,
                    "combined_lift_pct": round(f.combined_lift * 100, 1),
                    "baseline_multiplier": round(f.baseline_multiplier, 2),
                    "staffing": f.staffing_recommendation,
                    "stocking_notes": f.stocking_notes,
                    "events": [
                        {
                            "name": e.event_name,
                            "type": e.event_type,
                            "lift_pct": round(e.predicted_lift * 100, 1),
                            "confidence": round(e.confidence, 2),
                        }
                        for e in f.events
                    ],
                }
                for f in forecasts
            ],
        }
        
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "predictive_calendar.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate predictive calendar."""
    
    print("=" * 70)
    print("📅 PREDICTIVE EVENT CALENDAR")
    print("   What's coming up and how will it affect sales?")
    print("=" * 70)
    print()
    
    calendar = PredictiveCalendar()
    
    print("Building 14-day forecast...\n")
    forecasts = calendar.build_calendar(days_ahead=14)
    
    print("=" * 70)
    print("🗓️  NEXT 14 DAYS FORECAST")
    print("=" * 70 + "\n")
    
    for forecast in forecasts:
        # Highlight significant days
        if abs(forecast.combined_lift) > 0.10:
            print("─" * 50)
        
        print(forecast)
        
        if forecast.events:
            for event in forecast.events:
                print(f"      • {event.event_name}: {event.predicted_lift:+.0%}")
        
        if forecast.stocking_notes:
            for note in forecast.stocking_notes:
                print(f"      📦 {note}")
        
        print()
    
    # Summary
    print("=" * 70)
    print("📊 FORECAST SUMMARY")
    print("=" * 70 + "\n")
    
    high_days = [f for f in forecasts if f.combined_lift > 0.15]
    low_days = [f for f in forecasts if f.combined_lift < -0.05]
    
    if high_days:
        print("🔥 HIGH IMPACT DAYS (>+15%):")
        for f in high_days:
            print(f"   {f.day_name} {f.date}: {f.combined_lift:+.0%}")
    
    if low_days:
        print("\n⚠️ LOWER THAN NORMAL (<-5%):")
        for f in low_days:
            print(f"   {f.day_name} {f.date}: {f.combined_lift:+.0%}")
    
    avg_lift = sum(f.combined_lift for f in forecasts) / len(forecasts)
    print(f"\n📈 Average expected lift: {avg_lift:+.1%}")
    
    # Save
    output_file = calendar.save_calendar(forecasts)
    print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    demo()
