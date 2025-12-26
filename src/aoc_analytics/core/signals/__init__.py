"""
Signals module - THE KEYSTONE

This module contains the weather-as-a-priori transformation logic.
Weather is NOT correlated with sales - it is transformed into behavioral
propensities BEFORE any other analysis occurs.

The keystone formula:
    at_home = 0.1 + 0.45*rain + 0.3*cold + 0.1*wind + 0.15*snow

Weather conditions become behavioral weights that normalize sales expectations.

Additional signals:
- Vibe signals: Sports, weather patterns, academic, cultural, cruise ships
- External calendar: Holidays (Nager.Date API), BC events
"""

from .builder import (
    rebuild_behavioral_signals,
    _score_at_home,
    _score_out_and_about,
    _score_local_vibe,
)
from .payday_index import build_payday_index
from .vibe_signals import (
    VibeEngine,
    VibeType,
    VibeSignal,
    DayVibe,
    get_vibe_for_date,
    get_vibe_for_date_cached,
    preload_vibe_cache,
    clear_vibe_cache,
    get_vibe_features,
    SportsSchedule,
    WeatherVibe,
    AcademicCalendar,
    GovernmentPayments,
    CruiseSchedule,
    AstronomicalSignals,
)
from .external_calendar import (
    CalendarService,
    NagerDateClient,
    BCEventsClient,
    Holiday,
    ExternalEvent,
    fetch_holidays,
    is_holiday,
    get_events,
)

__all__ = [
    # Original signals
    "rebuild_behavioral_signals",
    "_score_at_home",
    "_score_out_and_about", 
    "_score_local_vibe",
    "build_payday_index",
    
    # Vibe signals
    "VibeEngine",
    "VibeType",
    "VibeSignal",
    "DayVibe",
    "get_vibe_for_date",
    "get_vibe_for_date_cached",
    "preload_vibe_cache",
    "clear_vibe_cache",
    "get_vibe_features",
    "SportsSchedule",
    "WeatherVibe",
    "AcademicCalendar",
    "GovernmentPayments",
    "CruiseSchedule",
    "AstronomicalSignals",
    
    # External calendar
    "CalendarService",
    "NagerDateClient",
    "BCEventsClient",
    "Holiday",
    "ExternalEvent",
    "fetch_holidays",
    "is_holiday",
    "get_events",
]
