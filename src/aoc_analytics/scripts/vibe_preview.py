#!/usr/bin/env python3
"""
AOC Vibe Preview CLI

Preview vibe signals for any date to understand what factors
affect demand forecasting.

Usage:
    python -m aoc_analytics.scripts.vibe_preview 2024-04-20
    python -m aoc_analytics.scripts.vibe_preview today
    python -m aoc_analytics.scripts.vibe_preview tomorrow
    python -m aoc_analytics.scripts.vibe_preview --range 2024-12-20 2024-12-31
"""

import argparse
import sys
from datetime import date, datetime, timedelta
from typing import Optional

from aoc_analytics.core.signals.vibe_signals import (
    VibeEngine,
    DayVibe,
    VibeType,
    get_vibe_for_date,
)
from aoc_analytics.core.signals.external_calendar import CalendarService


def parse_date(date_str: str) -> date:
    """Parse a date string, supporting 'today', 'tomorrow', etc."""
    if date_str.lower() == "today":
        return date.today()
    elif date_str.lower() == "tomorrow":
        return date.today() + timedelta(days=1)
    elif date_str.lower() == "yesterday":
        return date.today() - timedelta(days=1)
    else:
        try:
            return date.fromisoformat(date_str)
        except ValueError:
            # Try other formats
            for fmt in ["%Y/%m/%d", "%m/%d/%Y", "%d-%m-%Y"]:
                try:
                    return datetime.strptime(date_str, fmt).date()
                except ValueError:
                    continue
            raise ValueError(f"Could not parse date: {date_str}")


def format_vibe_type(vibe_type: VibeType) -> str:
    """Format vibe type with emoji."""
    emoji_map = {
        VibeType.COUCH_MODE: "🛋️",
        VibeType.PARTY_MODE: "🎉",
        VibeType.COZY_INDOOR: "🌧️",
        VibeType.OUTDOOR_ACTIVE: "☀️",
        VibeType.STRESS_MODE: "😰",
        VibeType.CELEBRATION: "🎊",
        VibeType.EXODUS: "✈️",
        VibeType.TOURIST_INFLUX: "🚢",
    }
    return f"{emoji_map.get(vibe_type, '❓')} {vibe_type.value.replace('_', ' ').title()}"


def print_vibe(day_vibe: DayVibe, verbose: bool = False):
    """Pretty-print a DayVibe."""
    dt = day_vibe.date
    day_name = dt.strftime("%A")
    
    print(f"\n{'='*60}")
    print(f"📅 {dt.isoformat()} ({day_name})")
    print(f"{'='*60}")
    
    # Dominant vibe
    print(f"\n🎯 Dominant Vibe: {format_vibe_type(day_vibe.dominant_vibe)}")
    
    # Indices as bar graphs
    print(f"\n📊 Indices:")
    print(f"  Couch Index:    {_bar(day_vibe.couch_index)} {day_vibe.couch_index:.2f}")
    print(f"  Party Index:    {_bar(day_vibe.party_index)} {day_vibe.party_index:.2f}")
    print(f"  Stress Index:   {_bar(day_vibe.stress_index)} {day_vibe.stress_index:.2f}")
    print(f"  Weather Cozy:   {_bar(day_vibe.weather_cozy)} {day_vibe.weather_cozy:.2f}")
    print(f"  Foot Traffic:   {_bar(day_vibe.foot_traffic)} {day_vibe.foot_traffic:.2f}")
    
    # Major event flag
    if day_vibe.has_major_event:
        print(f"\n🚨 HAS MAJOR EVENT")
    
    # Signals
    if day_vibe.signals:
        print(f"\n📡 Active Signals ({len(day_vibe.signals)}):")
        for i, signal in enumerate(day_vibe.signals, 1):
            print(f"  {i}. {signal.name}")
            print(f"     Type: {format_vibe_type(signal.vibe_type)}")
            print(f"     Intensity: {_bar(signal.intensity)} {signal.intensity:.2f}")
            if verbose and signal.description:
                print(f"     Note: {signal.description}")
            if signal.category_boosts:
                boosts = ", ".join(f"{k}+{v:.0%}" for k, v in signal.category_boosts.items())
                print(f"     Category Boosts: {boosts}")
            if signal.category_demotes:
                demotes = ", ".join(f"{k}-{v:.0%}" for k, v in signal.category_demotes.items())
                print(f"     Category Demotes: {demotes}")
    else:
        print(f"\n📡 No special signals (normal day)")
    
    print()


def _bar(value: float, width: int = 20) -> str:
    """Create a simple bar visualization."""
    filled = int(value * width)
    empty = width - filled
    return f"[{'█' * filled}{'░' * empty}]"


def print_calendar_info(dt: date):
    """Print calendar/holiday info for a date."""
    calendar = CalendarService()
    data = calendar.get_calendar_data(dt)
    
    if data.get("is_holiday"):
        print(f"🎄 Holiday: {data['holiday_name']}")
        if data.get("is_federal_holiday"):
            print(f"   (Federal holiday)")
    
    if data.get("has_events"):
        print(f"🎪 Events:")
        for event in data["events"]:
            print(f"   - {event['name']} @ {event['venue']}")
            if event.get("attendance"):
                print(f"     Expected attendance: {event['attendance']:,}")


def preview_date(dt: date, verbose: bool = False):
    """Preview vibe for a single date."""
    # Get calendar info first
    print_calendar_info(dt)
    
    # Get vibe
    vibe = get_vibe_for_date(dt)
    print_vibe(vibe, verbose)


def preview_range(start: date, end: date, verbose: bool = False):
    """Preview vibes for a date range."""
    print(f"\n📆 Vibe Preview: {start} to {end}")
    print(f"{'='*60}")
    
    engine = VibeEngine()
    current = start
    
    while current <= end:
        vibe = engine.get_day_vibe(current)
        
        # Compact format for ranges
        signals_str = ", ".join(s.name for s in vibe.signals) if vibe.signals else "Normal"
        major = "⚡" if vibe.has_major_event else "  "
        
        print(
            f"{current.isoformat()} {current.strftime('%a'):>3} "
            f"{major} C:{vibe.couch_index:.1f} P:{vibe.party_index:.1f} "
            f"S:{vibe.stress_index:.1f} | {signals_str[:40]}"
        )
        
        current += timedelta(days=1)
    
    print()


def main():
    parser = argparse.ArgumentParser(
        description="Preview AOC vibe signals for demand forecasting",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s today                    # Today's vibe
  %(prog)s 2024-04-20               # 4/20 vibes 🌿
  %(prog)s --range 2024-12-20 2024-12-31  # Holiday week range
  %(prog)s tomorrow -v              # Verbose output
        """
    )
    
    parser.add_argument(
        "date",
        nargs="?",
        default="today",
        help="Date to preview (YYYY-MM-DD, 'today', 'tomorrow')",
    )
    
    parser.add_argument(
        "--range", "-r",
        nargs=2,
        metavar=("START", "END"),
        help="Preview a date range",
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed signal descriptions",
    )
    
    args = parser.parse_args()
    
    try:
        if args.range:
            start = parse_date(args.range[0])
            end = parse_date(args.range[1])
            if end < start:
                start, end = end, start
            preview_range(start, end, args.verbose)
        else:
            dt = parse_date(args.date)
            preview_date(dt, args.verbose)
            
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nCancelled.")
        sys.exit(0)


if __name__ == "__main__":
    main()
