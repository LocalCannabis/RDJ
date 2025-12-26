"""
AOC Calendar System

Holiday detection, cannabis culture events, paydays, and local events
for intelligent regime selection.

BC Statutory Holidays + Cannabis Culture Events
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS
# =============================================================================

class EventType(str, Enum):
    STATUTORY_HOLIDAY = "statutory_holiday"
    CANNABIS_CULTURE = "cannabis_culture"
    PAYDAY = "payday"
    LONG_WEEKEND = "long_weekend"
    LOCAL_EVENT = "local_event"
    SEASONAL = "seasonal"


class EventImpact(str, Enum):
    """How the event affects shopping behavior."""
    GIFTING = "gifting"           # Premium/gift-sets boost
    PARTY = "party"               # Edibles/beverages/pre-rolls boost
    BULK_BUY = "bulk_buy"         # Ounces/bulk boost (payday)
    INDOOR_COZY = "indoor_cozy"   # Indica/flower/edibles boost
    CELEBRATION = "celebration"   # Social products boost
    VALUE_SEEKING = "value_seeking"  # Budget-conscious behavior


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class CalendarEvent:
    """A calendar event that affects shopping behavior."""
    name: str
    event_type: EventType
    impacts: List[EventImpact]
    category_boosts: List[str]
    category_demotes: List[str]
    boost_weights: Dict[str, float]
    description: str = ""


# =============================================================================
# BC STATUTORY HOLIDAYS (2025-2026)
# =============================================================================

BC_STATUTORY_HOLIDAYS: Dict[date, str] = {
    # 2025
    date(2025, 1, 1): "New Year's Day",
    date(2025, 2, 17): "Family Day",  # 3rd Monday of Feb
    date(2025, 4, 18): "Good Friday",
    date(2025, 5, 19): "Victoria Day",  # Monday before May 25
    date(2025, 7, 1): "Canada Day",
    date(2025, 8, 4): "BC Day",  # 1st Monday of Aug
    date(2025, 9, 1): "Labour Day",  # 1st Monday of Sep
    date(2025, 9, 30): "National Day for Truth and Reconciliation",
    date(2025, 10, 13): "Thanksgiving",  # 2nd Monday of Oct
    date(2025, 11, 11): "Remembrance Day",
    date(2025, 12, 25): "Christmas Day",
    date(2025, 12, 26): "Boxing Day",
    # 2026
    date(2026, 1, 1): "New Year's Day",
    date(2026, 2, 16): "Family Day",
    date(2026, 4, 3): "Good Friday",
    date(2026, 5, 18): "Victoria Day",
    date(2026, 7, 1): "Canada Day",
    date(2026, 8, 3): "BC Day",
    date(2026, 9, 7): "Labour Day",
    date(2026, 9, 30): "National Day for Truth and Reconciliation",
    date(2026, 10, 12): "Thanksgiving",
    date(2026, 11, 11): "Remembrance Day",
    date(2026, 12, 25): "Christmas Day",
    date(2026, 12, 26): "Boxing Day",
}

# Holidays that create gifting behavior
GIFTING_HOLIDAYS: Set[str] = {
    "Christmas Day",
    "Boxing Day",
    "Valentine's Day",  # Not statutory but relevant
    "Mother's Day",
    "Father's Day",
}

# Holidays that create party/celebration behavior
PARTY_HOLIDAYS: Set[str] = {
    "New Year's Day",
    "Canada Day",
    "BC Day",
    "Victoria Day",
    "Labour Day",
}


# =============================================================================
# CANNABIS CULTURE EVENTS
# =============================================================================

CANNABIS_CULTURE_EVENTS: Dict[Tuple[int, int], CalendarEvent] = {
    # 4/20 - The big one
    (4, 20): CalendarEvent(
        name="4/20",
        event_type=EventType.CANNABIS_CULTURE,
        impacts=[EventImpact.PARTY, EventImpact.CELEBRATION],
        category_boosts=["edibles", "pre-rolls", "beverages", "party-packs", "ounces"],
        category_demotes=["CBD", "tinctures"],
        boost_weights={"edibles": 2.0, "pre-rolls": 1.5, "beverages": 1.0, "ounces": 1.0},
        description="Cannabis culture's biggest day - expect 2-3x normal volume",
    ),
    # 7/10 - Oil Day (710 upside down = OIL)
    (7, 10): CalendarEvent(
        name="7/10 (Oil Day)",
        event_type=EventType.CANNABIS_CULTURE,
        impacts=[EventImpact.CELEBRATION],
        category_boosts=["extracts", "concentrates", "vapes", "dabs"],
        category_demotes=["flower", "pre-rolls"],
        boost_weights={"extracts": 2.0, "concentrates": 1.5, "vapes": 1.0},
        description="Concentrate/extract celebration day",
    ),
    # Green Wednesday (day before US Thanksgiving - still relevant in Canada)
    (11, 26): CalendarEvent(
        name="Green Wednesday",
        event_type=EventType.CANNABIS_CULTURE,
        impacts=[EventImpact.PARTY, EventImpact.BULK_BUY],
        category_boosts=["edibles", "pre-rolls", "ounces"],
        category_demotes=[],
        boost_weights={"edibles": 1.0, "pre-rolls": 0.5},
        description="Pre-holiday stock-up day",
    ),
}


# =============================================================================
# PAYDAY PATTERNS
# =============================================================================

def is_payday_period(dt: date) -> Tuple[bool, str]:
    """
    Detect if date is in a payday period.
    Common paydays: 1st, 15th, last Friday of month.
    
    Returns (is_payday, reason)
    """
    day = dt.day
    weekday = dt.weekday()  # 0=Monday, 4=Friday
    
    # 1st of month (or next business day)
    if day <= 3 and day >= 1:
        return True, "Beginning of month payday"
    
    # 15th of month (or adjacent days)
    if 14 <= day <= 16:
        return True, "Mid-month payday"
    
    # Last Friday of month
    # Check if it's Friday and no more Fridays this month
    if weekday == 4:  # Friday
        next_friday = dt + timedelta(days=7)
        if next_friday.month != dt.month:
            return True, "Last Friday payday"
    
    # Also check Saturday after last Friday (people cash out)
    if weekday == 5:  # Saturday
        prev_friday = dt - timedelta(days=1)
        next_friday = prev_friday + timedelta(days=7)
        if next_friday.month != prev_friday.month:
            return True, "Day after last Friday payday"
    
    return False, ""


# =============================================================================
# SPORTS SEASONS & SUNDAY CONTEXT
# =============================================================================

def is_nfl_season(dt: date) -> bool:
    """
    NFL season roughly runs early September through early February.
    This affects Sunday shopping patterns significantly in BC
    (Seahawks/49ers territory).
    """
    month = dt.month
    # Regular season: Sep 7 - Feb 13 (approximate)
    if month in (9, 10, 11, 12, 1):
        return True
    if month == 2 and dt.day <= 15:  # Super Bowl period
        return True
    return False


def is_super_bowl_sunday(dt: date) -> bool:
    """Check if it's Super Bowl Sunday (always 2nd Sunday of February)."""
    if dt.month != 2:
        return False
    if dt.weekday() != 6:  # Not Sunday
        return False
    # Second Sunday of February is between day 8-14
    return 8 <= dt.day <= 14


def get_sunday_type(dt: date) -> str:
    """
    Categorize Sundays to help with prediction variance.
    
    Returns: 'super_bowl', 'nfl_season', 'long_weekend', 'month_end', 'regular'
    """
    if dt.weekday() != 6:
        return "not_sunday"
    
    if is_super_bowl_sunday(dt):
        return "super_bowl"
    
    if is_nfl_season(dt):
        return "nfl_season"
    
    # Check if it's part of a long weekend
    next_day = dt + timedelta(days=1)
    if next_day in BC_STATUTORY_HOLIDAYS:
        return "long_weekend"
    
    # Month position
    if dt.day >= 25:
        return "month_end"
    elif dt.day <= 7:
        return "month_start"
    
    return "regular"


# =============================================================================
# SEASONAL DETECTION
# =============================================================================

def get_season(dt: date) -> str:
    """Get meteorological season for a date."""
    month = dt.month
    if month in (12, 1, 2):
        return "winter"
    elif month in (3, 4, 5):
        return "spring"
    elif month in (6, 7, 8):
        return "summer"
    else:  # 9, 10, 11
        return "fall"


# Seasonal category adjustments (from data analysis)
SEASONAL_ADJUSTMENTS: Dict[str, Dict[str, float]] = {
    "winter": {
        # Flower Indica peaks Winter (4.7%)
        "indica": 0.5,
        "flower": 0.3,
        "edibles": 0.2,
    },
    "summer": {
        # Beverages peak Summer (5.1%)
        "beverages": 0.5,
        "sativa": 0.3,
        "pre-rolls": 0.2,
    },
    "fall": {
        # Pre-Rolls peak Fall (21.4%)
        "pre-rolls": 0.5,
    },
    "spring": {
        "sativa": 0.2,
        "beverages": 0.2,
    },
}


# =============================================================================
# CALENDAR ENGINE
# =============================================================================

class AOCCalendar:
    """
    Calendar engine for AOC regime detection.
    
    Handles:
    - BC statutory holidays
    - Cannabis culture events (4/20, 7/10)
    - Payday detection
    - Seasonal adjustments
    - Long weekend detection
    
    Usage:
        calendar = AOCCalendar()
        
        # Get all events for a date
        events = calendar.get_events(date.today())
        
        # Check specific event types
        is_holiday, name = calendar.is_holiday(date.today())
        is_payday, reason = calendar.is_payday(date.today())
        
        # Get aggregated category adjustments
        boosts, demotes = calendar.get_category_adjustments(date.today())
    """
    
    def __init__(self):
        self.holidays = BC_STATUTORY_HOLIDAYS
        self.cannabis_events = CANNABIS_CULTURE_EVENTS
        self.seasonal_adjustments = SEASONAL_ADJUSTMENTS
    
    def get_events(self, dt: date) -> List[CalendarEvent]:
        """Get all calendar events for a date."""
        events = []
        
        # Check statutory holiday
        if dt in self.holidays:
            holiday_name = self.holidays[dt]
            impacts = []
            boosts = []
            boost_weights = {}
            
            if holiday_name in GIFTING_HOLIDAYS:
                impacts.append(EventImpact.GIFTING)
                boosts.extend(["premium", "gift-sets", "bundles"])
                boost_weights = {"premium": 1.0, "gift-sets": 0.5}
            
            if holiday_name in PARTY_HOLIDAYS:
                impacts.append(EventImpact.PARTY)
                boosts.extend(["edibles", "beverages", "pre-rolls"])
                boost_weights.update({"edibles": 0.5, "beverages": 0.3})
            
            events.append(CalendarEvent(
                name=holiday_name,
                event_type=EventType.STATUTORY_HOLIDAY,
                impacts=impacts,
                category_boosts=boosts,
                category_demotes=[],
                boost_weights=boost_weights,
            ))
        
        # Check cannabis culture event
        month_day = (dt.month, dt.day)
        if month_day in self.cannabis_events:
            events.append(self.cannabis_events[month_day])
        
        # Check payday
        is_payday, reason = is_payday_period(dt)
        if is_payday:
            events.append(CalendarEvent(
                name="Payday Period",
                event_type=EventType.PAYDAY,
                impacts=[EventImpact.BULK_BUY],
                category_boosts=["ounces", "bulk", "premium"],
                category_demotes=[],
                boost_weights={"ounces": 1.0, "premium": 0.5},
                description=reason,
            ))
        
        # Check long weekend (Friday before or Monday of stat holiday)
        for holiday_date, holiday_name in self.holidays.items():
            # Friday before Monday holiday
            if holiday_date.weekday() == 0:  # Monday
                if dt == holiday_date - timedelta(days=3):  # Friday
                    events.append(CalendarEvent(
                        name=f"Long Weekend ({holiday_name})",
                        event_type=EventType.LONG_WEEKEND,
                        impacts=[EventImpact.PARTY],
                        category_boosts=["edibles", "beverages", "pre-rolls"],
                        category_demotes=[],
                        boost_weights={"edibles": 0.5, "beverages": 0.3},
                    ))
        
        return events
    
    def is_holiday(self, dt: date) -> Tuple[bool, Optional[str]]:
        """Check if date is a statutory holiday."""
        if dt in self.holidays:
            return True, self.holidays[dt]
        return False, None
    
    def is_payday(self, dt: date) -> Tuple[bool, str]:
        """Check if date is a payday period."""
        return is_payday_period(dt)
    
    def is_cannabis_event(self, dt: date) -> Tuple[bool, Optional[CalendarEvent]]:
        """Check if date is a cannabis culture event."""
        month_day = (dt.month, dt.day)
        if month_day in self.cannabis_events:
            return True, self.cannabis_events[month_day]
        return False, None
    
    def get_season(self, dt: date) -> str:
        """Get the season for a date."""
        return get_season(dt)
    
    def get_seasonal_boosts(self, dt: date) -> Dict[str, float]:
        """Get seasonal category boost weights."""
        season = self.get_season(dt)
        return self.seasonal_adjustments.get(season, {})
    
    def get_category_adjustments(
        self, dt: date
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Get aggregated category adjustments for a date.
        
        Returns:
            (boost_weights, demote_weights) - aggregated from all events
        """
        boosts: Dict[str, float] = {}
        demotes: Dict[str, float] = {}
        
        # Get all events
        events = self.get_events(dt)
        
        # Aggregate boosts from events
        for event in events:
            for cat, weight in event.boost_weights.items():
                boosts[cat] = boosts.get(cat, 0) + weight
        
        # Add seasonal adjustments
        seasonal = self.get_seasonal_boosts(dt)
        for cat, weight in seasonal.items():
            boosts[cat] = boosts.get(cat, 0) + weight
        
        return boosts, demotes
    
    def get_event_summary(self, dt: date) -> str:
        """Get a human-readable summary of events for a date."""
        events = self.get_events(dt)
        season = self.get_season(dt)
        
        if not events:
            return f"Regular {season} day"
        
        event_names = [e.name for e in events]
        return f"{', '.join(event_names)} ({season})"
    
    def days_until_event(
        self, dt: date, event_type: EventType
    ) -> Tuple[Optional[int], Optional[str]]:
        """
        Find days until next event of a specific type.
        
        Returns:
            (days_until, event_name) or (None, None) if not found in next 90 days
        """
        for i in range(90):
            check_date = dt + timedelta(days=i)
            
            if event_type == EventType.STATUTORY_HOLIDAY:
                if check_date in self.holidays:
                    return i, self.holidays[check_date]
            
            elif event_type == EventType.CANNABIS_CULTURE:
                month_day = (check_date.month, check_date.day)
                if month_day in self.cannabis_events:
                    return i, self.cannabis_events[month_day].name
        
        return None, None


# =============================================================================
# INTEGRATION WITH DECISION ROUTER
# =============================================================================

def get_calendar_context(dt: Optional[datetime] = None) -> Dict[str, any]:
    """
    Get calendar context for decision routing.
    
    Returns dict compatible with TimeContext and SignalContext.
    """
    if dt is None:
        dt = datetime.now()
    
    if isinstance(dt, datetime):
        dt_date = dt.date()
    else:
        dt_date = dt
    
    calendar = AOCCalendar()
    
    is_holiday, holiday_name = calendar.is_holiday(dt_date)
    is_payday, payday_reason = calendar.is_payday(dt_date)
    is_cannabis, cannabis_event = calendar.is_cannabis_event(dt_date)
    
    events = calendar.get_events(dt_date)
    boosts, demotes = calendar.get_category_adjustments(dt_date)
    
    # Calculate holiday index (0-1)
    holiday_index = 0.0
    if is_holiday:
        holiday_index = 0.9
    elif is_cannabis:
        holiday_index = 1.0 if cannabis_event.name == "4/20" else 0.7
    
    # Calculate payday index (0-1)
    payday_index = 0.8 if is_payday else 0.3
    
    return {
        "is_holiday": is_holiday,
        "holiday_name": holiday_name,
        "is_payday": is_payday,
        "payday_reason": payday_reason,
        "is_cannabis_event": is_cannabis,
        "cannabis_event": cannabis_event.name if cannabis_event else None,
        "season": calendar.get_season(dt_date),
        "events": [e.name for e in events],
        "event_summary": calendar.get_event_summary(dt_date),
        "category_boosts": boosts,
        "category_demotes": demotes,
        "holiday_index": holiday_index,
        "payday_index": payday_index,
        # New Sunday/sports context
        "is_sunday": dt_date.weekday() == 6,
        "sunday_type": get_sunday_type(dt_date),
        "is_nfl_season": is_nfl_season(dt_date),
        "is_super_bowl": is_super_bowl_sunday(dt_date),
        # Pre-holiday detection (day before holiday)
        "is_preholiday": (dt_date + timedelta(days=1)) in BC_STATUTORY_HOLIDAYS,
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    from datetime import timedelta
    
    print("=" * 70)
    print("AOC Calendar System Test")
    print("=" * 70)
    
    calendar = AOCCalendar()
    today = date.today()
    
    print(f"\n📅 Today: {today}")
    print(f"   Season: {calendar.get_season(today)}")
    print(f"   Summary: {calendar.get_event_summary(today)}")
    
    is_holiday, name = calendar.is_holiday(today)
    print(f"   Holiday: {name if is_holiday else 'No'}")
    
    is_payday, reason = calendar.is_payday(today)
    print(f"   Payday: {reason if is_payday else 'No'}")
    
    boosts, demotes = calendar.get_category_adjustments(today)
    print(f"   Boosts: {boosts}")
    
    # Show upcoming events
    print("\n🗓️ Upcoming Events (next 30 days):")
    for i in range(30):
        check_date = today + timedelta(days=i)
        events = calendar.get_events(check_date)
        if events:
            event_names = [e.name for e in events]
            print(f"   {check_date}: {', '.join(event_names)}")
    
    # Test 4/20
    print("\n🌿 4/20 Test:")
    four_twenty = date(2025, 4, 20)
    ctx = get_calendar_context(datetime.combine(four_twenty, datetime.min.time()))
    print(f"   Events: {ctx['events']}")
    print(f"   Holiday Index: {ctx['holiday_index']}")
    print(f"   Boosts: {ctx['category_boosts']}")
