"""
LLM Synthesis for Brain Insights

Uses Ollama (local LLM) to generate natural language summaries
of what the brain has learned, turning statistical findings into
actionable human-readable insights.

Examples:
- "Cruise ship days boost sales by 36% - this is your #1 external driver"
- "Pre-Rolls are your most weather-sensitive category: down 12% on cold days"
- "Friday is your busiest day, especially 1pm-2pm (+107% above normal)"
"""

import json
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class BrainInsight:
    """A synthesized insight from brain data."""
    category: str  # "sales_drivers", "weather", "timing", "events"
    headline: str
    details: str
    confidence: float
    actionable: bool
    source_data: Dict


class LLMSynthesizer:
    """
    Generates natural language summaries using Ollama.
    Falls back to template-based generation if Ollama unavailable.
    """
    
    # Default Ollama model - use smaller models for speed
    DEFAULT_MODEL = "llama3.2:3b"  # Fast, good for summaries
    FALLBACK_MODELS = ["llama3.2:1b", "phi3", "mistral", "llama2"]
    
    def __init__(self, model: str = None):
        self.model = model or self.DEFAULT_MODEL
        self.brain_dir = Path(__file__).parent / "data"
        self.ollama_available = self._check_ollama()
        
        # Load all brain data
        self.signal_magnitudes = self._load_json("learned_signal_magnitudes.json")
        self.weather_impacts = self._load_json("category_weather_impacts.json")
        self.time_patterns = self._load_json("time_of_day_patterns.json")
        self.predictive_calendar = self._load_json("predictive_calendar.json")
        self.event_correlations = self._load_json("event_impact_analysis.json")
        self.inventory_recs = self._load_json("inventory_recommendations.json")
        self.cross_store = self._load_json("cross_store_analysis.json")
    
    def _load_json(self, filename: str) -> Dict:
        """Load a JSON file from brain/data."""
        path = self.brain_dir / filename
        if path.exists():
            with open(path) as f:
                return json.load(f)
        return {}
    
    def _check_ollama(self) -> bool:
        """Check if Ollama is available and running."""
        try:
            result = subprocess.run(
                ["ollama", "list"],
                capture_output=True,
                text=True,
                timeout=5,
            )
            return result.returncode == 0
        except (subprocess.SubprocessError, FileNotFoundError):
            return False
    
    def _call_ollama(self, prompt: str, system: str = None) -> Optional[str]:
        """Call Ollama with a prompt."""
        if not self.ollama_available:
            return None
        
        try:
            cmd = ["ollama", "run", self.model]
            
            full_prompt = prompt
            if system:
                full_prompt = f"System: {system}\n\nUser: {prompt}"
            
            result = subprocess.run(
                cmd,
                input=full_prompt,
                capture_output=True,
                text=True,
                timeout=60,
            )
            
            if result.returncode == 0:
                return result.stdout.strip()
            return None
            
        except subprocess.SubprocessError:
            return None
    
    def _format_signal_data(self) -> str:
        """Format signal magnitude data for the LLM."""
        if not self.signal_magnitudes:
            return "No signal data available."
        
        signals = self.signal_magnitudes.get("signals", {})
        lines = []
        
        for key, sig in sorted(signals.items(), key=lambda x: -abs(x[1].get("lift", 0))):
            lift = sig.get("lift", 0)
            lift_pct = sig.get("lift_pct", f"{lift*100:.1f}%")
            conf = sig.get("confidence", 0)
            actionable = "✓" if sig.get("actionable") else ""
            lines.append(f"- {sig.get('signal_name', key)}: {lift_pct} lift (confidence: {conf:.0%}) {actionable}")
        
        return "\n".join(lines)
    
    def _format_weather_data(self) -> str:
        """Format weather impact data for the LLM."""
        if not self.weather_impacts:
            return "No weather impact data available."
        
        insights = self.weather_impacts.get("actionable_insights", [])
        lines = []
        
        for imp in insights[:10]:
            cat = imp.get("category", "Unknown")
            cond = imp.get("weather_condition", "unknown")
            lift = imp.get("avg_lift_pct", 0)
            lines.append(f"- {cat} on {cond} days: {lift:+.1f}%")
        
        return "\n".join(lines)
    
    def _format_time_data(self) -> str:
        """Format time-of-day data for the LLM."""
        if not self.time_patterns:
            return "No time pattern data available."
        
        lines = []
        
        # Peak hours
        hourly = self.time_patterns.get("hourly_patterns", [])
        sorted_hours = sorted(hourly, key=lambda x: -x.get("pct_of_daily", 0))
        
        lines.append("Peak hours:")
        for h in sorted_hours[:3]:
            hour = h.get("hour", 0)
            pct = h.get("pct_of_daily", 0)
            lines.append(f"  - {hour}:00: {pct:.1%} of daily sales")
        
        # Day-hour interactions
        day_hour = self.time_patterns.get("day_hour_patterns", [])
        sorted_dh = sorted(day_hour, key=lambda x: -x.get("relative_strength", 0))
        
        lines.append("\nHot spots (day + hour combinations):")
        for dh in sorted_dh[:3]:
            day = dh.get("day_name", "?")
            hour = dh.get("hour", 0)
            strength = dh.get("relative_strength", 0)
            lines.append(f"  - {day} {hour}:00: {strength:+.0%} above baseline")
        
        return "\n".join(lines)
    
    def _format_event_data(self) -> str:
        """Format event correlation data for the LLM."""
        if not self.event_correlations:
            return "No event correlation data available."
        
        correlations = self.event_correlations.get("correlations", [])
        lines = []
        
        for corr in correlations:
            event = corr.get("event_type", "Unknown")
            impact = corr.get("avg_impact_pct", 0)
            p = corr.get("p_value", 1)
            sig = "significant" if p < 0.05 else "not significant"
            lines.append(f"- {event}: {impact:+.1f}% ({sig}, p={p:.3f})")
        
        return "\n".join(lines)
    
    def synthesize_executive_summary(self) -> str:
        """Generate an executive summary of all brain learnings."""
        
        system_prompt = """You are a data analyst for a cannabis retail store in Vancouver. 
Generate a brief, actionable executive summary based on the sales pattern data provided.
Keep it under 200 words. Focus on the top 3-4 insights that would help the store manager.
Use specific numbers. Be direct and practical."""

        data_prompt = f"""Here's what our analysis has learned about sales patterns:

## EXTERNAL SIGNALS (events that affect sales)
{self._format_signal_data()}

## WEATHER IMPACTS (by product category)
{self._format_weather_data()}

## TIME PATTERNS
{self._format_time_data()}

## EVENT CORRELATIONS
{self._format_event_data()}

Generate an executive summary with the top actionable insights."""

        # Try Ollama first
        if self.ollama_available:
            response = self._call_ollama(data_prompt, system_prompt)
            if response:
                return response
        
        # Fallback to template-based summary
        return self._generate_template_summary()
    
    def _generate_template_summary(self) -> str:
        """Generate a template-based summary when LLM is unavailable."""
        
        lines = [
            "📊 BRAIN INSIGHTS SUMMARY",
            "=" * 50,
            "",
        ]
        
        # Top signal
        if self.signal_magnitudes:
            signals = self.signal_magnitudes.get("signals", {})
            actionable = [(k, v) for k, v in signals.items() if v.get("actionable")]
            if actionable:
                top = max(actionable, key=lambda x: abs(x[1].get("lift", 0)))
                lines.append(f"🎯 TOP DRIVER: {top[1].get('signal_name', top[0])}")
                lines.append(f"   Impact: {top[1].get('lift_pct', '?')} lift on sales")
                lines.append("")
        
        # Weather insight
        if self.weather_impacts:
            insights = self.weather_impacts.get("actionable_insights", [])
            if insights:
                top_weather = insights[0]
                lines.append(f"🌦️ WEATHER SENSITIVE: {top_weather.get('category', '?')}")
                lines.append(f"   {top_weather.get('avg_lift_pct', 0):+.1f}% on {top_weather.get('weather_condition', '?')} days")
                lines.append("")
        
        # Time insight
        if self.time_patterns:
            hourly = self.time_patterns.get("hourly_patterns", [])
            if hourly:
                peak = max(hourly, key=lambda x: x.get("pct_of_daily", 0))
                lines.append(f"⏰ PEAK HOUR: {peak.get('hour', '?')}:00")
                lines.append(f"   {peak.get('pct_of_daily', 0):.1%} of daily sales")
                lines.append("")
        
        # Upcoming events
        if self.predictive_calendar:
            forecasts = self.predictive_calendar.get("forecasts", [])
            high_impact = [f for f in forecasts if f.get("combined_lift_pct", 0) > 15]
            if high_impact:
                lines.append(f"📅 HIGH IMPACT DAYS AHEAD: {len(high_impact)}")
                for f in high_impact[:3]:
                    events = [e.get("name", "?") for e in f.get("events", [])]
                    lines.append(f"   • {f.get('date')}: {f.get('combined_lift_pct', 0):+.0f}% ({', '.join(events[:2])})")
                lines.append("")
        
        lines.append("=" * 50)
        
        return "\n".join(lines)
    
    def synthesize_daily_briefing(self, target_date: str = None) -> str:
        """Generate a daily briefing for a specific date."""
        
        if target_date is None:
            target_date = datetime.now().strftime("%Y-%m-%d")
        
        # Find this date in predictive calendar
        forecast = None
        if self.predictive_calendar:
            for f in self.predictive_calendar.get("forecasts", []):
                if f.get("date") == target_date:
                    forecast = f
                    break
        
        if not forecast:
            return f"No forecast data available for {target_date}"
        
        system_prompt = """You are a shift manager at a cannabis retail store.
Generate a brief daily briefing (under 100 words) based on today's forecast.
Be practical and specific about what staff should expect and prepare for."""

        data_prompt = f"""Today's forecast ({target_date}):

Day: {forecast.get('day_name', '?')}
Expected sales change: {forecast.get('combined_lift_pct', 0):+.1f}% vs normal
Staffing recommendation: {forecast.get('staffing', 'Normal')}

Events today:
{chr(10).join(f"- {e.get('name', '?')}: {e.get('lift_pct', 0):+.1f}%" for e in forecast.get('events', []))}

Stock focus: {', '.join(forecast.get('stocking_notes', ['Normal mix']))}

Generate a brief daily briefing for staff."""

        if self.ollama_available:
            response = self._call_ollama(data_prompt, system_prompt)
            if response:
                return response
        
        # Fallback
        lines = [
            f"📋 DAILY BRIEFING: {forecast.get('day_name', '?')} {target_date}",
            "-" * 40,
            f"Expected: {forecast.get('combined_lift_pct', 0):+.0f}% vs normal",
            f"Staffing: {forecast.get('staffing', 'Normal')}",
            "",
        ]
        
        events = forecast.get("events", [])
        if events:
            lines.append("Events:")
            for e in events:
                lines.append(f"  • {e.get('name', '?')}")
        
        notes = forecast.get("stocking_notes", [])
        if notes:
            lines.append("")
            lines.append("Stock focus:")
            for note in notes:
                lines.append(f"  • {note}")
        
        return "\n".join(lines)
    
    def synthesize_weekly_newsletter(self) -> str:
        """Generate a weekly newsletter summarizing learnings and upcoming events."""
        
        system_prompt = """You are the owner of a cannabis retail store writing a weekly 
newsletter for your management team. Summarize what we learned this week about sales 
patterns and what to expect next week. Keep it under 300 words, conversational but 
professional. Include specific numbers."""

        # Gather all data
        data_sections = [
            "## SALES PATTERNS WE'VE LEARNED",
            self._format_signal_data(),
            "",
            "## WEATHER IMPACTS BY CATEGORY",
            self._format_weather_data(),
            "",
            "## TIMING PATTERNS",
            self._format_time_data(),
            "",
            "## NEXT WEEK'S FORECAST",
        ]
        
        if self.predictive_calendar:
            forecasts = self.predictive_calendar.get("forecasts", [])[:7]
            for f in forecasts:
                lift = f.get("combined_lift_pct", 0)
                events = [e.get("name", "?") for e in f.get("events", [])]
                events_str = f" [{', '.join(events[:2])}]" if events else ""
                data_sections.append(f"- {f.get('day_name', '?')} {f.get('date', '?')}: {lift:+.0f}%{events_str}")
        
        data_prompt = "\n".join(data_sections) + "\n\nGenerate a weekly newsletter summary."

        if self.ollama_available:
            response = self._call_ollama(data_prompt, system_prompt)
            if response:
                return response
        
        # Fallback - structured summary
        return self._generate_template_summary()
    
    def save_synthesis(self, content: str, synthesis_type: str) -> str:
        """Save synthesized content to file."""
        
        output = {
            "generated": datetime.now().isoformat(),
            "type": synthesis_type,
            "ollama_used": self.ollama_available,
            "model": self.model if self.ollama_available else "template",
            "content": content,
        }
        
        output_file = self.brain_dir / f"synthesis_{synthesis_type}.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)


def demo():
    """Demonstrate LLM synthesis."""
    
    print("=" * 70)
    print("🧠 LLM SYNTHESIS - Natural Language Brain Insights")
    print("=" * 70)
    print()
    
    synth = LLMSynthesizer()
    
    if synth.ollama_available:
        print(f"✓ Ollama available, using model: {synth.model}")
    else:
        print("⚠️ Ollama not available, using template-based synthesis")
        print("   Install Ollama for richer AI-generated summaries:")
        print("   curl -fsSL https://ollama.com/install.sh | sh")
        print("   ollama pull llama3.2:3b")
    print()
    
    # Executive summary
    print("=" * 70)
    print("📊 EXECUTIVE SUMMARY")
    print("=" * 70)
    summary = synth.synthesize_executive_summary()
    print(summary)
    synth.save_synthesis(summary, "executive_summary")
    
    # Daily briefing
    print("\n" + "=" * 70)
    print("📋 TODAY'S BRIEFING")
    print("=" * 70)
    briefing = synth.synthesize_daily_briefing()
    print(briefing)
    synth.save_synthesis(briefing, "daily_briefing")
    
    # Check for upcoming high-impact day
    if synth.predictive_calendar:
        forecasts = synth.predictive_calendar.get("forecasts", [])
        high_impact = [f for f in forecasts if f.get("combined_lift_pct", 0) > 15]
        if high_impact:
            next_big = high_impact[0]
            print("\n" + "=" * 70)
            print(f"🔥 UPCOMING HIGH IMPACT: {next_big.get('date')}")
            print("=" * 70)
            briefing = synth.synthesize_daily_briefing(next_big.get("date"))
            print(briefing)
    
    print("\n💾 Synthesis saved to brain/data/")


if __name__ == "__main__":
    demo()
