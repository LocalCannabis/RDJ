"""
AOC Vibe & Traffic Signals

Creative free data sources for demand forecasting:
- Sports schedules (Canucks, Seahawks, Whitecaps, Lions)
- Weather patterns (cozy index, rain streaks, first nice day)
- Academic calendar (UBC, SFU, BCIT)
- Cultural moments (Pride, Celebration of Light, Halloween)
- Cruise ship traffic
- Astronomical (full moons, daylight hours)
- Government payment schedules

All sources are free / no API key required.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from enum import Enum
from functools import lru_cache
from typing import Dict, List, Optional, Tuple, Any
import json

logger = logging.getLogger(__name__)


# =============================================================================
# MODULE-LEVEL CACHE
# =============================================================================

# Cache for DayVibe objects by date (without weather - weather is dynamic)
_VIBE_CACHE: Dict[date, "DayVibe"] = {}


# =============================================================================
# ENUMS
# =============================================================================

class VibeType(str, Enum):
    """Categories of vibe that affect shopping behavior."""
    COUCH_MODE = "couch_mode"          # Stay home, watch sports/movies
    PARTY_MODE = "party_mode"          # Going out, social events
    COZY_INDOOR = "cozy_indoor"        # Bad weather, hunker down
    OUTDOOR_ACTIVE = "outdoor_active"  # Nice weather, patios
    STRESS_MODE = "stress_mode"        # Finals, work deadlines
    CELEBRATION = "celebration"        # Holidays, victories
    EXODUS = "exodus"                  # Locals leaving town
    TOURIST_INFLUX = "tourist_influx"  # Visitors in town


class TrafficImpact(str, Enum):
    """How traffic/attendance affects store visits."""
    HIGH_FOOT_TRAFFIC = "high_foot_traffic"
    LOW_FOOT_TRAFFIC = "low_foot_traffic"
    DOWNTOWN_BUSY = "downtown_busy"
    SUBURBAN_BUSY = "suburban_busy"
    EVERYONE_HOME = "everyone_home"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class VibeSignal:
    """A vibe signal for a specific date."""
    name: str
    vibe_type: VibeType
    traffic_impact: TrafficImpact
    intensity: float  # 0.0 to 1.0
    category_boosts: Dict[str, float] = field(default_factory=dict)
    category_demotes: Dict[str, float] = field(default_factory=dict)
    description: str = ""


@dataclass 
class DayVibe:
    """Aggregated vibe for a single day."""
    date: date
    signals: List[VibeSignal] = field(default_factory=list)
    couch_index: float = 0.5      # 0=out and about, 1=everyone home
    party_index: float = 0.0       # 0=normal, 1=peak party mode
    stress_index: float = 0.0      # 0=relaxed, 1=high stress
    weather_cozy: float = 0.5      # 0=nice out, 1=hunker down weather
    foot_traffic: float = 0.5      # 0=ghost town, 1=packed streets
    
    @property
    def has_major_event(self) -> bool:
        """True if any signal has intensity > 0.6."""
        return any(s.intensity > 0.6 for s in self.signals)
    
    @property
    def dominant_vibe(self) -> Optional[VibeType]:
        """Return the vibe type with highest total intensity."""
        if not self.signals:
            return None
        vibe_totals: Dict[VibeType, float] = {}
        for signal in self.signals:
            vibe_totals[signal.vibe_type] = vibe_totals.get(signal.vibe_type, 0) + signal.intensity
        return max(vibe_totals, key=vibe_totals.get) if vibe_totals else None
    
    @property
    def dominant_vibe(self) -> VibeType:
        """Return the dominant vibe for the day."""
        if self.couch_index > 0.7:
            return VibeType.COUCH_MODE
        if self.party_index > 0.6:
            return VibeType.PARTY_MODE
        if self.stress_index > 0.6:
            return VibeType.STRESS_MODE
        if self.weather_cozy > 0.7:
            return VibeType.COZY_INDOOR
        if self.weather_cozy < 0.3:
            return VibeType.OUTDOOR_ACTIVE
        return VibeType.COUCH_MODE  # Default


# =============================================================================
# SPORTS SCHEDULES
# =============================================================================

class SportsSchedule:
    """
    Sports schedule detection for Vancouver area.
    
    Key teams:
    - NHL: Vancouver Canucks (Rogers Arena) - Oct to Apr/Jun
    - NFL: Seattle Seahawks (proxy for Sunday couch) - Sep to Feb
    - MLS: Vancouver Whitecaps (BC Place) - Mar to Oct
    - CFL: BC Lions (BC Place) - Jun to Nov
    """
    
    # NHL Canucks approximate home game pattern
    # ~41 home games Oct-Apr, roughly 2-3 per week
    @staticmethod
    def is_nhl_season(dt: date) -> bool:
        """NHL regular season + playoffs."""
        month = dt.month
        if month in (10, 11, 12, 1, 2, 3):
            return True
        if month == 4 and dt.day <= 20:  # Regular season end
            return True
        if month in (4, 5, 6) and dt.year in _CANUCKS_PLAYOFF_YEARS:
            return True
        return False
    
    @staticmethod
    def is_likely_canucks_home_game(dt: date) -> Tuple[bool, float]:
        """
        Estimate if there's a Canucks home game.
        Returns (is_likely, confidence).
        
        Pattern: ~41 home games over ~180 days = ~23% of days
        More likely on weekends, Tue/Thu
        """
        if not SportsSchedule.is_nhl_season(dt):
            return False, 0.0
        
        dow = dt.weekday()
        # Canucks play more on Sat (0.35), Tue/Thu (0.25), less Mon/Wed/Fri
        dow_prob = {
            0: 0.15,  # Monday
            1: 0.25,  # Tuesday
            2: 0.15,  # Wednesday
            3: 0.25,  # Thursday
            4: 0.18,  # Friday
            5: 0.35,  # Saturday
            6: 0.20,  # Sunday
        }
        return True, dow_prob.get(dow, 0.2)
    
    @staticmethod
    def is_nfl_game_window(dt: date) -> Tuple[bool, str]:
        """
        NFL games affect Sunday behavior significantly.
        Seattle Seahawks are the local team of interest.
        
        Returns (is_game_window, game_type)
        """
        # NFL season: early Sep through early Feb
        month = dt.month
        if month in (3, 4, 5, 6, 7, 8):
            return False, ""
        
        dow = dt.weekday()
        
        # Sunday = main NFL day (1pm and 4pm PT games)
        if dow == 6:  # Sunday
            if month == 2 and 8 <= dt.day <= 14:
                return True, "super_bowl"
            return True, "sunday_football"
        
        # Monday Night Football
        if dow == 0:  # Monday
            return True, "monday_night"
        
        # Thursday Night Football
        if dow == 3:  # Thursday
            return True, "thursday_night"
        
        return False, ""
    
    @staticmethod
    def is_mls_season(dt: date) -> bool:
        """MLS season runs roughly March through October."""
        return dt.month in (3, 4, 5, 6, 7, 8, 9, 10)
    
    @staticmethod
    def is_cfl_season(dt: date) -> bool:
        """CFL season runs roughly June through November."""
        return dt.month in (6, 7, 8, 9, 10, 11)


# Years when Canucks made playoffs (affects Apr-Jun)
_CANUCKS_PLAYOFF_YEARS = {2020, 2024, 2025}  # Update as needed


# =============================================================================
# WEATHER VIBES
# =============================================================================

class WeatherVibe:
    """
    Weather-based vibe signals beyond raw temp/precip.
    
    Captures psychological effects:
    - Rain streak cabin fever
    - First nice day after rain
    - Heat wave outdoor mode
    - Cold snap cozy mode
    """
    
    @staticmethod
    def calculate_rain_streak(
        precip_history: List[float],  # Last 7 days of precip_mm
    ) -> Tuple[int, bool]:
        """
        Calculate consecutive rainy days and detect 'first nice day'.
        
        Returns (streak_length, is_first_nice_day)
        """
        if len(precip_history) < 2:
            return 0, False
        
        # Count consecutive rainy days (>1mm)
        streak = 0
        for precip in reversed(precip_history[:-1]):  # Exclude today
            if precip > 1.0:
                streak += 1
            else:
                break
        
        # Today is first nice day if streak was 3+ and today is dry
        today_precip = precip_history[-1] if precip_history else 0
        is_first_nice = streak >= 3 and today_precip < 1.0
        
        return streak, is_first_nice
    
    @staticmethod
    def calculate_cozy_index(
        temp_c: float,
        precip_mm: float,
        wind_kmh: float = 0,
        cloud_cover: float = 0,
    ) -> float:
        """
        Calculate 'cozy index' - how much the weather encourages staying in.
        
        0.0 = Perfect patio weather
        1.0 = Absolute hunker-down weather
        """
        cozy = 0.5  # Start neutral
        
        # Temperature factor
        if temp_c < 5:
            cozy += 0.25  # Cold = cozy up
        elif temp_c < 10:
            cozy += 0.1
        elif temp_c > 25:
            cozy -= 0.2  # Hot = patio time
        elif temp_c > 20:
            cozy -= 0.1
        
        # Precipitation factor (biggest impact)
        if precip_mm > 10:
            cozy += 0.3  # Heavy rain = stay in
        elif precip_mm > 5:
            cozy += 0.2
        elif precip_mm > 1:
            cozy += 0.1
        elif precip_mm < 0.5 and temp_c > 15:
            cozy -= 0.15  # Nice and dry = go out
        
        # Wind factor
        if wind_kmh > 30:
            cozy += 0.1
        
        # Cloud cover
        if cloud_cover > 80:
            cozy += 0.05
        elif cloud_cover < 20 and temp_c > 15:
            cozy -= 0.1
        
        return max(0.0, min(1.0, cozy))
    
    @staticmethod
    def get_weather_vibe_signal(
        temp_c: float,
        precip_mm: float,
        rain_streak: int = 0,
        is_first_nice: bool = False,
    ) -> Optional[VibeSignal]:
        """Generate a vibe signal based on weather conditions."""
        
        # First nice day after rain streak = outdoor rush
        if is_first_nice and temp_c > 12:
            return VibeSignal(
                name="First Nice Day",
                vibe_type=VibeType.OUTDOOR_ACTIVE,
                traffic_impact=TrafficImpact.HIGH_FOOT_TRAFFIC,
                intensity=min(0.3 + rain_streak * 0.1, 0.8),
                category_boosts={"pre-rolls": 0.3, "sativa": 0.2, "beverages": 0.2},
                category_demotes={"indica": 0.1},
                description=f"First nice day after {rain_streak} days of rain",
            )
        
        # Extended rain = cabin fever
        if rain_streak >= 5:
            return VibeSignal(
                name="Cabin Fever",
                vibe_type=VibeType.COZY_INDOOR,
                traffic_impact=TrafficImpact.LOW_FOOT_TRAFFIC,
                intensity=min(0.3 + (rain_streak - 5) * 0.1, 0.7),
                category_boosts={"indica": 0.2, "edibles": 0.15, "flower": 0.1},
                category_demotes={"pre-rolls": 0.15},
                description=f"{rain_streak} consecutive rainy days",
            )
        
        # Heat wave
        if temp_c > 28:
            return VibeSignal(
                name="Heat Wave",
                vibe_type=VibeType.OUTDOOR_ACTIVE,
                traffic_impact=TrafficImpact.HIGH_FOOT_TRAFFIC,
                intensity=min((temp_c - 28) * 0.15, 0.6),
                category_boosts={"beverages": 0.3, "pre-rolls": 0.2, "sativa": 0.15},
                category_demotes={"indica": 0.1, "edibles": 0.1},
                description=f"Heat wave: {temp_c}°C",
            )
        
        # Cold snap
        if temp_c < 0:
            return VibeSignal(
                name="Cold Snap", 
                vibe_type=VibeType.COZY_INDOOR,
                traffic_impact=TrafficImpact.LOW_FOOT_TRAFFIC,
                intensity=min(abs(temp_c) * 0.1, 0.5),
                category_boosts={"indica": 0.2, "flower": 0.15, "edibles": 0.1},
                category_demotes={"beverages": 0.2, "pre-rolls": 0.15},
                description=f"Cold snap: {temp_c}°C",
            )
        
        return None


# =============================================================================
# ACADEMIC CALENDAR
# =============================================================================

class AcademicCalendar:
    """
    BC post-secondary academic calendar.
    
    UBC, SFU, BCIT, Langara, Douglas, etc.
    ~200,000+ students in Metro Vancouver.
    """
    
    # Approximate academic dates (varies by year)
    @staticmethod
    def get_academic_period(dt: date) -> Tuple[str, float]:
        """
        Get academic period and stress level.
        
        Returns (period_name, stress_level)
        stress_level: 0.0 = relaxed, 1.0 = peak stress
        """
        month, day = dt.month, dt.day
        
        # Winter finals: Dec 5-20
        if month == 12 and 5 <= day <= 20:
            return "winter_finals", 0.8
        
        # Spring finals: Apr 10-25
        if month == 4 and 10 <= day <= 25:
            return "spring_finals", 0.8
        
        # Reading break: mid-February (varies)
        if month == 2 and 15 <= day <= 23:
            return "reading_break", 0.1  # Party time
        
        # Summer break: May-Aug
        if month in (5, 6, 7, 8):
            return "summer_break", 0.2
        
        # Winter break: Dec 21 - Jan 5
        if month == 12 and day >= 21:
            return "winter_break", 0.1
        if month == 1 and day <= 5:
            return "winter_break", 0.1
        
        # First week of classes (adjustment stress)
        if month == 9 and day <= 10:
            return "fall_start", 0.4
        if month == 1 and 6 <= day <= 12:
            return "winter_start", 0.4
        
        # Regular semester
        if month in (9, 10, 11) or month in (1, 2, 3):
            # Midterms mid-Oct and mid-Feb
            if (month == 10 and 15 <= day <= 25) or (month == 2 and 24 <= day <= 28):
                return "midterms", 0.6
            return "semester", 0.3
        
        return "between_terms", 0.2
    
    @staticmethod
    def get_academic_vibe_signal(dt: date) -> Optional[VibeSignal]:
        """Generate vibe signal for academic calendar."""
        period, stress = AcademicCalendar.get_academic_period(dt)
        
        if period == "winter_finals" or period == "spring_finals":
            return VibeSignal(
                name=f"Finals Week ({period.replace('_', ' ').title()})",
                vibe_type=VibeType.STRESS_MODE,
                traffic_impact=TrafficImpact.SUBURBAN_BUSY,
                intensity=stress,
                category_boosts={"indica": 0.15, "CBD": 0.2, "edibles": 0.1},
                category_demotes={"pre-rolls": 0.1},
                description="University finals period - stress purchases",
            )
        
        if period == "reading_break":
            return VibeSignal(
                name="Reading Break",
                vibe_type=VibeType.PARTY_MODE,
                traffic_impact=TrafficImpact.HIGH_FOOT_TRAFFIC,
                intensity=0.5,
                category_boosts={"pre-rolls": 0.15, "edibles": 0.15, "beverages": 0.1},
                category_demotes={},
                description="University reading break - party mode",
            )
        
        return None


# =============================================================================
# GOVERNMENT PAYMENT CALENDAR
# =============================================================================

class GovernmentPayments:
    """
    Canadian government payment schedules.
    
    - Welfare/Income Assistance: Last Wednesday of month ("Welfare Wednesday")
    - GST/HST credit: Quarterly (Jan, Apr, Jul, Oct)
    - Canada Child Benefit: ~20th of month
    - CPP/OAS: ~last business day of month
    """
    
    @staticmethod
    def get_payment_date_type(dt: date) -> Tuple[str, float]:
        """
        Check if date is a government payment date.
        
        Returns (payment_type, boost_factor)
        """
        dow = dt.weekday()
        day = dt.day
        month = dt.month
        
        # Welfare Wednesday: Last Wednesday of month
        if dow == 2:  # Wednesday
            next_wed = dt + timedelta(days=7)
            if next_wed.month != dt.month:
                return "welfare_wednesday", 0.6
        
        # GST/HST Credit: ~5th of Jan, Apr, Jul, Oct
        if month in (1, 4, 7, 10) and 3 <= day <= 7:
            return "gst_credit", 0.3
        
        # Child benefit: ~20th
        if 18 <= day <= 22:
            return "child_benefit", 0.2
        
        # CPP/OAS: Last business day
        if day >= 25:
            # Check if last business day
            temp = dt
            while temp.month == dt.month:
                temp += timedelta(days=1)
            last_day = temp - timedelta(days=1)
            while last_day.weekday() >= 5:  # Weekend
                last_day -= timedelta(days=1)
            if dt == last_day:
                return "cpp_oas", 0.3
        
        return "", 0.0
    
    @staticmethod
    def get_payment_vibe_signal(dt: date) -> Optional[VibeSignal]:
        """Generate vibe signal for government payment dates."""
        payment_type, boost = GovernmentPayments.get_payment_date_type(dt)
        
        if payment_type == "welfare_wednesday":
            return VibeSignal(
                name="Welfare Wednesday",
                vibe_type=VibeType.COUCH_MODE,
                traffic_impact=TrafficImpact.SUBURBAN_BUSY,
                intensity=boost,
                category_boosts={"value": 0.3, "budget": 0.25, "ounces": 0.2},
                category_demotes={"premium": 0.15},
                description="Monthly income assistance day - budget focus",
            )
        
        if payment_type == "gst_credit":
            return VibeSignal(
                name="GST Credit Day",
                vibe_type=VibeType.COUCH_MODE,
                traffic_impact=TrafficImpact.SUBURBAN_BUSY,
                intensity=boost,
                category_boosts={"value": 0.15, "ounces": 0.1},
                category_demotes={},
                description="Quarterly GST credit payment",
            )
        
        return None


# =============================================================================
# CULTURAL MOMENTS
# =============================================================================

# Major Vancouver cultural events (month, day) -> event info
VANCOUVER_CULTURAL_EVENTS: Dict[Tuple[int, int], Dict[str, Any]] = {
    # Pride Week (late July/early Aug - parade is last Sunday of July)
    (7, 28): {
        "name": "Pride Weekend",
        "duration": 3,  # days
        "vibe": VibeType.PARTY_MODE,
        "traffic": TrafficImpact.DOWNTOWN_BUSY,
        "boosts": {"pre-rolls": 0.3, "edibles": 0.2, "beverages": 0.2},
    },
    
    # Celebration of Light (late July - 3 nights)
    (7, 20): {
        "name": "Celebration of Light (Start)",
        "duration": 10,  # spread over ~2 weeks
        "vibe": VibeType.OUTDOOR_ACTIVE,
        "traffic": TrafficImpact.HIGH_FOOT_TRAFFIC,
        "boosts": {"pre-rolls": 0.25, "beverages": 0.15},
    },
    
    # Halloween weekend
    (10, 29): {
        "name": "Halloween Weekend",
        "duration": 4,
        "vibe": VibeType.PARTY_MODE,
        "traffic": TrafficImpact.HIGH_FOOT_TRAFFIC,
        "boosts": {"edibles": 0.4, "pre-rolls": 0.2, "party-packs": 0.3},
    },
    
    # St. Patrick's Day
    (3, 17): {
        "name": "St. Patrick's Day",
        "duration": 1,
        "vibe": VibeType.PARTY_MODE,
        "traffic": TrafficImpact.DOWNTOWN_BUSY,
        "boosts": {"beverages": 0.3, "pre-rolls": 0.2},
    },
    
    # New Year's Eve
    (12, 31): {
        "name": "New Year's Eve",
        "duration": 1,
        "vibe": VibeType.PARTY_MODE,
        "traffic": TrafficImpact.DOWNTOWN_BUSY,
        "boosts": {"edibles": 0.3, "beverages": 0.3, "pre-rolls": 0.25},
    },
    
    # PNE Fair (mid-Aug to Labour Day)
    (8, 17): {
        "name": "PNE Fair",
        "duration": 15,
        "vibe": VibeType.OUTDOOR_ACTIVE,
        "traffic": TrafficImpact.HIGH_FOOT_TRAFFIC,  # East Van especially
        "boosts": {"pre-rolls": 0.15, "edibles": 0.1},
    },
}


# =============================================================================
# CRUISE SHIP SCHEDULE
# =============================================================================

class CruiseSchedule:
    """
    Vancouver cruise ship schedule.
    
    Season: April - October
    Peak: May - September
    
    ~300 ships/year, mostly Alaska-bound.
    Each ship = 2,000-5,000 tourists downtown.
    """
    
    @staticmethod
    def is_cruise_season(dt: date) -> bool:
        """Check if in cruise season."""
        return dt.month in (4, 5, 6, 7, 8, 9, 10)
    
    @staticmethod
    def estimate_cruise_likelihood(dt: date) -> float:
        """
        Estimate likelihood of cruise ship in port.
        
        Peak season: ~1.5 ships/day average
        Shoulder: ~0.5 ships/day
        
        Returns 0.0 to 1.0 likelihood
        """
        if not CruiseSchedule.is_cruise_season(dt):
            return 0.0
        
        month = dt.month
        dow = dt.weekday()
        
        # Monthly intensity
        month_factor = {
            4: 0.3, 5: 0.7, 6: 0.9, 7: 1.0, 8: 1.0, 9: 0.8, 10: 0.4
        }.get(month, 0.0)
        
        # Day of week (more ships on weekends for Alaska runs)
        dow_factor = {
            0: 0.7, 1: 0.6, 2: 0.7, 3: 0.8, 4: 0.9, 5: 1.0, 6: 0.9
        }.get(dow, 0.7)
        
        return month_factor * dow_factor
    
    @staticmethod
    def get_cruise_vibe_signal(dt: date) -> Optional[VibeSignal]:
        """Generate vibe signal for cruise ship days."""
        likelihood = CruiseSchedule.estimate_cruise_likelihood(dt)
        
        if likelihood > 0.5:
            return VibeSignal(
                name="Cruise Ship Day",
                vibe_type=VibeType.TOURIST_INFLUX,
                traffic_impact=TrafficImpact.DOWNTOWN_BUSY,
                intensity=likelihood * 0.4,
                category_boosts={"pre-rolls": 0.1, "souvenirs": 0.2},
                category_demotes={"ounces": 0.1},  # Tourists buy small
                description=f"Likely cruise ship(s) in port ({likelihood:.0%} confidence)",
            )
        return None


# =============================================================================
# ASTRONOMICAL SIGNALS
# =============================================================================

class AstronomicalSignals:
    """
    Astronomical events that correlate with behavior.
    
    Yes, full moons affect retail. Multiple studies confirm this.
    Also: daylight hours affect mood and energy.
    """
    
    @staticmethod
    def get_moon_phase(dt: date) -> Tuple[str, float]:
        """
        Calculate moon phase.
        
        Returns (phase_name, fullness) where fullness is 0.0-1.0
        """
        # Simple lunar cycle calculation
        # New moon reference: Jan 6, 2000
        reference = date(2000, 1, 6)
        days_since = (dt - reference).days
        lunar_cycle = 29.53058867  # days
        
        phase = (days_since % lunar_cycle) / lunar_cycle
        
        if phase < 0.03 or phase > 0.97:
            return "new_moon", 0.0
        elif 0.22 < phase < 0.28:
            return "first_quarter", 0.5
        elif 0.47 < phase < 0.53:
            return "full_moon", 1.0
        elif 0.72 < phase < 0.78:
            return "last_quarter", 0.5
        elif phase < 0.5:
            return "waxing", phase * 2
        else:
            return "waning", (1 - phase) * 2
    
    @staticmethod
    def get_daylight_hours(dt: date, lat: float = 49.28) -> float:
        """
        Estimate daylight hours for Vancouver area.
        
        Rough calculation based on latitude and day of year.
        """
        day_of_year = dt.timetuple().tm_yday
        
        # Simplified daylight calculation
        # Vancouver: ~8 hours in December, ~16 hours in June
        declination = 23.45 * math.sin(math.radians((360/365) * (day_of_year - 81)))
        lat_rad = math.radians(lat)
        decl_rad = math.radians(declination)
        
        # Hour angle at sunrise/sunset
        cos_hour = -math.tan(lat_rad) * math.tan(decl_rad)
        cos_hour = max(-1, min(1, cos_hour))  # Clamp
        hour_angle = math.degrees(math.acos(cos_hour))
        
        daylight = 2 * hour_angle / 15  # Convert to hours
        return daylight
    
    @staticmethod
    def get_astronomical_vibe_signal(dt: date) -> Optional[VibeSignal]:
        """Generate vibe signal for astronomical events."""
        phase, fullness = AstronomicalSignals.get_moon_phase(dt)
        
        if phase == "full_moon":
            return VibeSignal(
                name="Full Moon",
                vibe_type=VibeType.PARTY_MODE,
                traffic_impact=TrafficImpact.HIGH_FOOT_TRAFFIC,
                intensity=0.15,  # Subtle but real effect
                category_boosts={"sativa": 0.05, "edibles": 0.05},
                category_demotes={},
                description="Full moon - historically elevated retail activity",
            )
        
        return None


# =============================================================================
# VIBE AGGREGATOR
# =============================================================================

class VibeEngine:
    """
    Aggregate all vibe signals for a given date.
    
    Combines:
    - Sports schedules
    - Weather patterns
    - Academic calendar
    - Government payments
    - Cultural events
    - Cruise ships
    - Astronomical
    """
    
    def __init__(self):
        self.sports = SportsSchedule()
        self.weather = WeatherVibe()
        self.academic = AcademicCalendar()
        self.payments = GovernmentPayments()
        self.cruises = CruiseSchedule()
        self.astro = AstronomicalSignals()
    
    def get_day_vibe(
        self,
        dt: date,
        weather_data: Optional[Dict[str, float]] = None,
        precip_history: Optional[List[float]] = None,
    ) -> DayVibe:
        """
        Calculate comprehensive vibe for a date.
        
        Parameters
        ----------
        dt : date
            Target date
        weather_data : dict, optional
            Today's weather: {temp_c, precip_mm, wind_kmh, cloud_cover}
        precip_history : list, optional
            Last 7 days of precipitation for rain streak detection
        """
        signals = []
        
        # Sports signals
        is_nfl, nfl_type = SportsSchedule.is_nfl_game_window(dt)
        if is_nfl and nfl_type:
            intensity = 0.7 if nfl_type == "super_bowl" else 0.4
            signals.append(VibeSignal(
                name=f"NFL: {nfl_type.replace('_', ' ').title()}",
                vibe_type=VibeType.COUCH_MODE,
                traffic_impact=TrafficImpact.EVERYONE_HOME,
                intensity=intensity,
                category_boosts={"edibles": 0.2, "pre-rolls": 0.15, "beverages": 0.1},
                description=f"NFL game window: {nfl_type}",
            ))
        
        is_canucks, prob = SportsSchedule.is_likely_canucks_home_game(dt)
        if is_canucks and prob > 0.2:
            signals.append(VibeSignal(
                name="Canucks Home Game (Likely)",
                vibe_type=VibeType.COUCH_MODE,
                traffic_impact=TrafficImpact.DOWNTOWN_BUSY,
                intensity=prob * 0.5,
                category_boosts={"beverages": 0.1, "pre-rolls": 0.1},
                description=f"Likely Canucks game ({prob:.0%} probability)",
            ))
        
        # Weather signals
        if weather_data:
            rain_streak = 0
            is_first_nice = False
            if precip_history:
                rain_streak, is_first_nice = WeatherVibe.calculate_rain_streak(precip_history)
            
            weather_signal = WeatherVibe.get_weather_vibe_signal(
                weather_data.get("temp_c", 15),
                weather_data.get("precip_mm", 0),
                rain_streak,
                is_first_nice,
            )
            if weather_signal:
                signals.append(weather_signal)
        
        # Academic signals
        academic_signal = AcademicCalendar.get_academic_vibe_signal(dt)
        if academic_signal:
            signals.append(academic_signal)
        
        # Payment signals
        payment_signal = GovernmentPayments.get_payment_vibe_signal(dt)
        if payment_signal:
            signals.append(payment_signal)
        
        # Cultural events
        for (m, d), event_info in VANCOUVER_CULTURAL_EVENTS.items():
            event_start = date(dt.year, m, d)
            event_end = event_start + timedelta(days=event_info.get("duration", 1))
            if event_start <= dt < event_end:
                signals.append(VibeSignal(
                    name=event_info["name"],
                    vibe_type=event_info["vibe"],
                    traffic_impact=event_info["traffic"],
                    intensity=0.5,
                    category_boosts=event_info.get("boosts", {}),
                ))
        
        # Cruise ships
        cruise_signal = CruiseSchedule.get_cruise_vibe_signal(dt)
        if cruise_signal:
            signals.append(cruise_signal)
        
        # Astronomical
        astro_signal = AstronomicalSignals.get_astronomical_vibe_signal(dt)
        if astro_signal:
            signals.append(astro_signal)
        
        # Calculate aggregate indices
        couch_index = 0.5
        party_index = 0.0
        stress_index = 0.0
        foot_traffic = 0.5
        
        for sig in signals:
            if sig.vibe_type == VibeType.COUCH_MODE:
                couch_index += sig.intensity * 0.3
            elif sig.vibe_type == VibeType.PARTY_MODE:
                party_index += sig.intensity * 0.4
            elif sig.vibe_type == VibeType.STRESS_MODE:
                stress_index += sig.intensity * 0.5
            
            if sig.traffic_impact in (TrafficImpact.HIGH_FOOT_TRAFFIC, TrafficImpact.DOWNTOWN_BUSY):
                foot_traffic += sig.intensity * 0.2
            elif sig.traffic_impact == TrafficImpact.LOW_FOOT_TRAFFIC:
                foot_traffic -= sig.intensity * 0.2
        
        # Weather cozy
        weather_cozy = 0.5
        if weather_data:
            weather_cozy = WeatherVibe.calculate_cozy_index(
                weather_data.get("temp_c", 15),
                weather_data.get("precip_mm", 0),
                weather_data.get("wind_kmh", 0),
                weather_data.get("cloud_cover", 50),
            )
        
        return DayVibe(
            date=dt,
            signals=signals,
            couch_index=max(0, min(1, couch_index)),
            party_index=max(0, min(1, party_index)),
            stress_index=max(0, min(1, stress_index)),
            weather_cozy=weather_cozy,
            foot_traffic=max(0, min(1, foot_traffic)),
        )
    
    def get_category_adjustments(
        self, 
        day_vibe: DayVibe
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """
        Aggregate category boosts/demotes from all signals.
        
        Returns (boosts, demotes) dictionaries.
        """
        boosts: Dict[str, float] = {}
        demotes: Dict[str, float] = {}
        
        for signal in day_vibe.signals:
            for cat, val in signal.category_boosts.items():
                boosts[cat] = boosts.get(cat, 0) + val * signal.intensity
            for cat, val in signal.category_demotes.items():
                demotes[cat] = demotes.get(cat, 0) + val * signal.intensity
        
        return boosts, demotes


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_vibe_for_date(
    dt: date,
    weather_data: Optional[Dict[str, float]] = None,
    precip_history: Optional[List[float]] = None,
    use_cache: bool = True,
) -> DayVibe:
    """
    Convenience function to get vibe for a single date.
    
    Uses module-level cache when use_cache=True and no weather data provided.
    """
    # Fast path: check cache for date-only lookups (no weather)
    if use_cache and weather_data is None and dt in _VIBE_CACHE:
        return _VIBE_CACHE[dt]
    
    engine = VibeEngine()
    vibe = engine.get_day_vibe(dt, weather_data, precip_history)
    
    # Cache if no weather data (deterministic result)
    if use_cache and weather_data is None:
        _VIBE_CACHE[dt] = vibe
    
    return vibe


def get_vibe_for_date_cached(dt: date) -> DayVibe:
    """
    Get vibe for a date, always using cache (no weather).
    
    This is the fastest path for backtesting.
    """
    if dt in _VIBE_CACHE:
        return _VIBE_CACHE[dt]
    
    engine = VibeEngine()
    vibe = engine.get_day_vibe(dt, None, None)
    _VIBE_CACHE[dt] = vibe
    return vibe


def preload_vibe_cache(start_date: date, end_date: date) -> int:
    """
    Preload vibe cache for a date range.
    
    Use this before backtesting to avoid repeated calculations.
    Returns number of dates loaded.
    """
    engine = VibeEngine()
    count = 0
    current = start_date
    while current <= end_date:
        if current not in _VIBE_CACHE:
            _VIBE_CACHE[current] = engine.get_day_vibe(current, None, None)
            count += 1
        current += timedelta(days=1)
    logger.info(f"Preloaded {count} dates into vibe cache ({start_date} to {end_date})")
    return count


def clear_vibe_cache():
    """Clear the vibe cache."""
    _VIBE_CACHE.clear()


def get_vibe_features(day_vibe: DayVibe) -> Dict[str, float]:
    """
    Extract numeric features from DayVibe for ML/similarity matching.
    
    Returns dict of features ready for use in forecasting.
    """
    return {
        "couch_index": day_vibe.couch_index,
        "party_index": day_vibe.party_index,
        "stress_index": day_vibe.stress_index,
        "weather_cozy": day_vibe.weather_cozy,
        "foot_traffic": day_vibe.foot_traffic,
        "n_signals": len(day_vibe.signals),
        "has_sports": any(s.name.startswith(("NFL", "Canucks")) for s in day_vibe.signals),
        "has_cultural": any(s.vibe_type == VibeType.PARTY_MODE for s in day_vibe.signals),
    }
