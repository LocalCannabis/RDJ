"""
AOC External Calendar Data Fetchers

Free API clients for:
- Nager.Date: Canadian/BC holidays (no API key)
- Open-Meteo: Weather history (already integrated)

Future additions (require API keys):
- Songkick/Bandsintown: Concert data
- SportsDB: Game schedules
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from functools import lru_cache
from typing import Dict, List, Optional, Any, Tuple
import json

logger = logging.getLogger(__name__)

# Check for httpx (async HTTP client) or fall back to requests
try:
    import httpx
    HTTPX_AVAILABLE = True
except ImportError:
    HTTPX_AVAILABLE = False

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False


# =============================================================================
# MODULE-LEVEL CACHES (shared across all instances)
# =============================================================================

_HOLIDAY_CACHE: Dict[int, List["Holiday"]] = {}  # year -> holidays
_HOLIDAY_CACHE_TIMES: Dict[int, datetime] = {}
_HOLIDAY_BY_DATE_CACHE: Dict[date, Optional["Holiday"]] = {}


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class Holiday:
    """Normalized holiday object."""
    date: date
    name: str
    local_name: str
    country_code: str
    is_federal: bool  # True if national, False if BC-only
    is_fixed: bool    # True if same date every year
    types: List[str]  # e.g., ["Public", "Bank"]
    
    @property
    def is_bc(self) -> bool:
        """Check if this holiday applies to BC."""
        return self.is_federal or "BC" in self.types


@dataclass
class ExternalEvent:
    """Normalized external event (concerts, sports, etc.)."""
    id: str
    name: str
    date: date
    end_date: Optional[date]
    venue: str
    category: str  # concert, sports, festival, etc.
    source: str    # nager, songkick, sportsdb, etc.
    attendance_estimate: Optional[int] = None
    url: Optional[str] = None


# =============================================================================
# NAGER.DATE API CLIENT
# =============================================================================

class NagerDateClient:
    """
    Client for Nager.Date Public Holiday API.
    
    Free, no API key required.
    https://date.nager.at/
    
    Provides Canadian federal and provincial holidays.
    Uses module-level cache shared across all instances.
    """
    
    BASE_URL = "https://date.nager.at/api/v3"
    
    def __init__(self, cache_ttl_hours: int = 24 * 30):
        """Initialize client with optional cache TTL."""
        self.cache_ttl_hours = cache_ttl_hours
    
    def _make_request(self, endpoint: str) -> Optional[List[Dict]]:
        """Make HTTP request to Nager.Date API."""
        url = f"{self.BASE_URL}/{endpoint}"
        
        try:
            if HTTPX_AVAILABLE:
                with httpx.Client(timeout=10.0) as client:
                    response = client.get(url)
                    response.raise_for_status()
                    return response.json()
            elif REQUESTS_AVAILABLE:
                response = requests.get(url, timeout=10)
                response.raise_for_status()
                return response.json()
            else:
                logger.error("No HTTP client available (install httpx or requests)")
                return None
        except Exception as e:
            logger.error(f"Nager.Date API error: {e}")
            return None
    
    def get_holidays(self, year: int) -> List[Holiday]:
        """
        Get Canadian holidays for a given year.
        
        Filters to federal + BC holidays.
        Uses module-level cache for cross-instance sharing.
        """
        # Check module-level cache
        if year in _HOLIDAY_CACHE:
            cache_time = _HOLIDAY_CACHE_TIMES.get(year)
            if cache_time and (datetime.now() - cache_time).total_seconds() < self.cache_ttl_hours * 3600:
                return _HOLIDAY_CACHE[year]
        
        # Fetch from API
        data = self._make_request(f"publicholidays/{year}/CA")
        
        if not data:
            logger.warning(f"Failed to fetch holidays for {year}, using fallback")
            return self._get_fallback_holidays(year)
        
        holidays = []
        for item in data:
            # Filter: federal OR BC
            counties = item.get("counties") or []
            is_federal = item.get("global", False)
            is_bc = "CA-BC" in counties
            
            if not (is_federal or is_bc):
                continue
            
            try:
                holiday = Holiday(
                    date=date.fromisoformat(item["date"]),
                    name=item["name"],
                    local_name=item.get("localName", item["name"]),
                    country_code=item["countryCode"],
                    is_federal=is_federal,
                    is_fixed=item.get("fixed", False),
                    types=item.get("types", []),
                )
                holidays.append(holiday)
                # Also cache by date for fast lookups
                _HOLIDAY_BY_DATE_CACHE[holiday.date] = holiday
            except (KeyError, ValueError) as e:
                logger.warning(f"Skipping malformed holiday: {e}")
        
        # Cache results at module level
        _HOLIDAY_CACHE[year] = holidays
        _HOLIDAY_CACHE_TIMES[year] = datetime.now()
        
        logger.info(f"Fetched {len(holidays)} BC holidays for {year}")
        return holidays
    
    def get_holiday_for_date(self, dt: date) -> Optional[Holiday]:
        """Check if a specific date is a holiday."""
        # Fast path: check date cache first
        if dt in _HOLIDAY_BY_DATE_CACHE:
            return _HOLIDAY_BY_DATE_CACHE[dt]
        
        # Ensure year is loaded
        holidays = self.get_holidays(dt.year)
        for h in holidays:
            if h.date == dt:
                return h
        return None
    
    def is_holiday(self, dt: date) -> tuple[bool, Optional[str]]:
        """
        Check if date is a holiday.
        
        Returns (is_holiday, holiday_name)
        """
        holiday = self.get_holiday_for_date(dt)
        if holiday:
            return True, holiday.name
        return False, None
    
    def _get_fallback_holidays(self, year: int) -> List[Holiday]:
        """
        Fallback holidays if API is unavailable.
        
        Uses static BC holiday dates.
        """
        from .calendar import BC_STATUTORY_HOLIDAYS
        
        holidays = []
        for dt, name in BC_STATUTORY_HOLIDAYS.items():
            if dt.year == year:
                holidays.append(Holiday(
                    date=dt,
                    name=name,
                    local_name=name,
                    country_code="CA",
                    is_federal=name in {"New Year's Day", "Canada Day", "Christmas Day"},
                    is_fixed=name in {"New Year's Day", "Canada Day", "Christmas Day", "Boxing Day"},
                    types=["Public"],
                ))
        return holidays


# =============================================================================
# BC EVENTS STATIC DATA
# =============================================================================

# Major annual BC events with approximate dates
# These rarely change year-to-year
BC_ANNUAL_EVENTS: Dict[str, Dict[str, Any]] = {
    "vancouver_pride": {
        "name": "Vancouver Pride Parade",
        "month": 8,  # First Sunday of August
        "week": 1,
        "day_of_week": 6,  # Sunday
        "duration": 1,
        "category": "festival",
        "venue": "Downtown Vancouver",
        "attendance": 500000,
    },
    "celebration_of_light": {
        "name": "Celebration of Light",
        "month": 7,
        "days": [20, 23, 26],  # Approximate - 3 nights over ~week
        "category": "festival",
        "venue": "English Bay",
        "attendance": 400000,
    },
    "pne_fair": {
        "name": "PNE Fair",
        "start_month": 8,
        "start_day": 17,  # Approximate - runs to Labour Day
        "duration": 15,
        "category": "festival",
        "venue": "PNE Grounds",
        "attendance": 700000,
    },
    "bard_on_beach": {
        "name": "Bard on the Beach",
        "start_month": 6,
        "end_month": 9,
        "category": "theatre",
        "venue": "Vanier Park",
        "attendance": 100000,
    },
    "jazz_festival": {
        "name": "TD Vancouver Jazz Festival",
        "month": 6,
        "start_day": 20,  # Approximate
        "duration": 11,
        "category": "concert",
        "venue": "Various Downtown",
        "attendance": 500000,
    },
    "folk_festival": {
        "name": "Vancouver Folk Music Festival",
        "month": 7,
        "week": 3,  # Third weekend of July
        "duration": 3,
        "category": "concert",
        "venue": "Jericho Beach",
        "attendance": 40000,
    },
    "fireworks_halloween": {
        "name": "Halloween Fireworks",
        "month": 10,
        "day": 31,
        "category": "festival",
        "venue": "Various",
        "attendance": 100000,
    },
    "stanley_park_christmas": {
        "name": "Bright Nights Stanley Park",
        "start_month": 11,
        "start_day": 28,  # Approximate
        "end_month": 1,
        "end_day": 1,
        "category": "festival",
        "venue": "Stanley Park",
        "attendance": 300000,
    },
    "van_dusen_lights": {
        "name": "VanDusen Festival of Lights",
        "start_month": 11,
        "start_day": 22,
        "end_month": 1,
        "end_day": 5,
        "category": "festival", 
        "venue": "VanDusen Garden",
        "attendance": 100000,
    },
}


class BCEventsClient:
    """
    Client for BC events data.
    
    Uses static data for major annual events.
    Could be extended to scrape venue calendars.
    """
    
    def __init__(self):
        self.events = BC_ANNUAL_EVENTS
    
    def get_events_for_date(self, dt: date) -> List[ExternalEvent]:
        """Get all events happening on a specific date."""
        events = []
        
        for event_id, info in self.events.items():
            if self._is_event_on_date(dt, info):
                events.append(ExternalEvent(
                    id=event_id,
                    name=info["name"],
                    date=dt,
                    end_date=None,
                    venue=info.get("venue", "Vancouver"),
                    category=info["category"],
                    source="bc_static",
                    attendance_estimate=info.get("attendance"),
                ))
        
        return events
    
    def _is_event_on_date(self, dt: date, info: Dict) -> bool:
        """Check if an event is happening on a given date."""
        # Single day event
        if "month" in info and "day" in info:
            return dt.month == info["month"] and dt.day == info["day"]
        
        # Multi-day specific dates
        if "month" in info and "days" in info:
            return dt.month == info["month"] and dt.day in info["days"]
        
        # Duration-based event
        if "start_month" in info and "start_day" in info and "duration" in info:
            start = date(dt.year, info["start_month"], info["start_day"])
            end = start + timedelta(days=info["duration"])
            return start <= dt < end
        
        # Season-based event (entire months)
        if "start_month" in info and "end_month" in info:
            if info["start_month"] <= info["end_month"]:
                return info["start_month"] <= dt.month <= info["end_month"]
            else:  # Wraps around year (Nov to Jan)
                return dt.month >= info["start_month"] or dt.month <= info["end_month"]
        
        # Week-based event (e.g., "first Sunday of August")
        if "month" in info and "week" in info and "day_of_week" in info:
            if dt.month != info["month"]:
                return False
            # Find the Nth occurrence of the day of week
            first_of_month = date(dt.year, info["month"], 1)
            first_dow = first_of_month.weekday()
            target_dow = info["day_of_week"]
            
            # Days until first occurrence
            days_until = (target_dow - first_dow) % 7
            nth_occurrence = first_of_month + timedelta(days=days_until + 7 * (info["week"] - 1))
            
            # Check if date is in range
            duration = info.get("duration", 1)
            return nth_occurrence <= dt < nth_occurrence + timedelta(days=duration)
        
        return False


# =============================================================================
# UNIFIED CALENDAR SERVICE
# =============================================================================

class CalendarService:
    """
    Unified calendar service combining all data sources.
    
    Sources:
    - Nager.Date: Holidays
    - BC Static: Major annual events
    - Vibe signals: Sports, weather, academic, etc.
    """
    
    def __init__(self):
        self.holidays = NagerDateClient()
        self.events = BCEventsClient()
    
    def get_calendar_data(self, dt: date) -> Dict[str, Any]:
        """
        Get all calendar data for a date.
        
        Returns comprehensive dict for forecasting use.
        """
        # Holiday check
        holiday = self.holidays.get_holiday_for_date(dt)
        
        # Events
        events = self.events.get_events_for_date(dt)
        
        # Build result
        result = {
            "date": dt.isoformat(),
            "day_of_week": dt.weekday(),
            "day_name": dt.strftime("%A"),
            "is_weekend": dt.weekday() >= 5,
            
            # Holiday info
            "is_holiday": holiday is not None,
            "holiday_name": holiday.name if holiday else None,
            "is_federal_holiday": holiday.is_federal if holiday else False,
            
            # Events
            "has_events": len(events) > 0,
            "events": [
                {
                    "name": e.name,
                    "category": e.category,
                    "venue": e.venue,
                    "attendance": e.attendance_estimate,
                }
                for e in events
            ],
            "total_event_attendance": sum(e.attendance_estimate or 0 for e in events),
        }
        
        return result
    
    def get_calendar_features(self, dt: date) -> Dict[str, float]:
        """
        Extract numeric features for ML/similarity matching.
        """
        data = self.get_calendar_data(dt)
        
        return {
            "is_holiday": 1.0 if data["is_holiday"] else 0.0,
            "is_federal_holiday": 1.0 if data["is_federal_holiday"] else 0.0,
            "is_weekend": 1.0 if data["is_weekend"] else 0.0,
            "has_events": 1.0 if data["has_events"] else 0.0,
            "event_attendance_scaled": min(data["total_event_attendance"] / 100000, 1.0),
        }


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def fetch_holidays(year: int) -> List[Holiday]:
    """Fetch holidays for a year."""
    client = NagerDateClient()
    return client.get_holidays(year)


def is_holiday(dt: date) -> tuple[bool, Optional[str]]:
    """Check if a date is a holiday."""
    client = NagerDateClient()
    return client.is_holiday(dt)


def get_events(dt: date) -> List[ExternalEvent]:
    """Get events for a date."""
    client = BCEventsClient()
    return client.get_events_for_date(dt)
