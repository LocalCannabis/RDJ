"""
AOC Store-Specific Regime Configurations

Per-store category adjustments based on data analysis of 538K+ transactions.
Each store has unique customer demographics and shopping patterns.

Data Source: Weather × Category correlation analysis per location
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any

logger = logging.getLogger(__name__)


# =============================================================================
# STORE IDS AND NAMES
# =============================================================================

STORE_IDS = {
    "parksville": "Parksville",
    "kingsway": "Kingsway", 
    "victoria_drive": "Victoria Drive",
}


# =============================================================================
# STORE-SPECIFIC WEATHER SENSITIVITY
# Data-validated from correlation analysis
# =============================================================================

STORE_WEATHER_SENSITIVITY: Dict[str, Dict[str, Dict[str, float]]] = {
    # Parksville - Vancouver Island, older demographic, more extreme weather patterns
    "parksville": {
        "cold_boosts": {
            "indica": 1.0,          # +1.0pp on cold days (strongest in chain)
            "flower": 0.5,
            "edibles": 0.3,
        },
        "cold_demotes": {
            "pre-rolls": 2.5,       # -2.5pp on cold days (HUGE swing)
            "beverages": 0.8,
        },
        "warm_boosts": {
            "pre-rolls": 2.5,       # Strong recovery on warm days
            "sativa": 0.8,
            "beverages": 0.5,
        },
        "warm_demotes": {
            "indica": 0.8,
            "flower": 0.5,
        },
        "rain_boosts": {
            "flower": 0.8,
            "indica": 0.6,
        },
        "rain_demotes": {
            "pre-rolls": 1.5,
        },
    },
    
    # Kingsway - Vancouver urban, younger demographic, commuter traffic
    "kingsway": {
        "cold_boosts": {
            "rosin-gummies": 0.6,   # Unique: premium edibles peak on cold days
            "indica": 0.5,
            "extracts": 0.4,
        },
        "cold_demotes": {
            "beverages": 1.0,       # -1.0pp on cold days
            "pre-rolls": 0.8,
        },
        "warm_boosts": {
            "beverages": 1.0,       # Strong recovery
            "sativa": 0.6,
            "pre-rolls": 0.5,
        },
        "warm_demotes": {
            "indica": 0.5,
            "extracts": 0.3,
        },
        "rain_boosts": {
            "edibles": 0.5,
            "indica": 0.4,
        },
        "rain_demotes": {
            "pre-rolls": 0.6,
        },
    },
    
    # Victoria Drive - Vancouver urban, diverse demographic, evening traffic
    "victoria_drive": {
        "cold_boosts": {
            "pre-roll-hybrid": 1.2,  # Unique: hybrid pre-rolls UP on cold days
            "accessories": 1.0,      # Unique: accessories UP on cold days
            "indica": 0.6,
        },
        "cold_demotes": {
            "beverages": 0.5,
            "sativa": 0.4,
        },
        "warm_boosts": {
            "sativa": 0.8,
            "beverages": 0.6,
        },
        "warm_demotes": {
            "pre-roll-hybrid": 1.0,
            "accessories": 0.8,
        },
        "rain_boosts": {
            "flower": 0.6,
            "edibles": 0.4,
        },
        "rain_demotes": {
            "pre-rolls": 0.5,
        },
    },
}


# =============================================================================
# STORE-SPECIFIC TIME PATTERNS
# =============================================================================

STORE_TIME_PATTERNS: Dict[str, Dict[str, Dict[str, float]]] = {
    "parksville": {
        # Older demographic, earlier shopping patterns
        "morning_boosts": {
            "pre-rolls": 0.8,       # Strong morning pre-roll sales
            "CBD": 0.5,
        },
        "morning_demotes": {
            "edibles": 0.4,
        },
        "evening_boosts": {
            "indica": 0.6,
            "edibles": 0.8,
        },
        "evening_demotes": {
            "sativa": 0.3,
        },
    },
    
    "kingsway": {
        # Commuter traffic, lunch and after-work peaks
        "morning_boosts": {
            "CBD": 0.4,
            "sativa": 0.3,
        },
        "morning_demotes": {
            "edibles": 0.3,
        },
        "evening_boosts": {
            "edibles": 1.0,         # Strong evening edibles
            "beverages": 0.8,
            "indica": 0.5,
        },
        "evening_demotes": {
            "sativa": 0.4,
        },
    },
    
    "victoria_drive": {
        # Evening-heavy traffic
        "morning_boosts": {
            "pre-rolls": 0.5,
        },
        "morning_demotes": {
            "edibles": 0.3,
        },
        "evening_boosts": {
            "edibles": 0.9,
            "indica": 0.7,
            "beverages": 0.6,
        },
        "evening_demotes": {
            "flower": 0.3,
        },
    },
}


# =============================================================================
# STORE-SPECIFIC WEEKEND PATTERNS
# =============================================================================

STORE_WEEKEND_PATTERNS: Dict[str, Dict[str, float]] = {
    "parksville": {
        # Tourist traffic on weekends
        "pre-rolls": 0.8,           # Tourists buy grab-and-go
        "edibles": 0.5,
        "beverages": 0.4,
    },
    "kingsway": {
        # Local regulars on weekends
        "edibles": 1.0,
        "beverages": 0.6,
        "ounces": 0.4,              # Bulk buying on weekends
    },
    "victoria_drive": {
        # Mixed traffic
        "edibles": 0.8,
        "beverages": 0.5,
    },
}


# =============================================================================
# STORE PROFILE CLASS
# =============================================================================

@dataclass
class StoreProfile:
    """Complete store profile with all adjustments."""
    store_id: str
    store_name: str
    weather_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    time_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weekend_boosts: Dict[str, float] = field(default_factory=dict)
    
    def get_weather_adjustments(
        self, 
        is_cold: bool = False, 
        is_warm: bool = False, 
        is_rainy: bool = False
    ) -> Dict[str, float]:
        """Get aggregated weather-based category adjustments."""
        adjustments: Dict[str, float] = {}
        
        if is_cold:
            for cat, weight in self.weather_sensitivity.get("cold_boosts", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) + weight
            for cat, weight in self.weather_sensitivity.get("cold_demotes", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) - weight
        
        if is_warm:
            for cat, weight in self.weather_sensitivity.get("warm_boosts", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) + weight
            for cat, weight in self.weather_sensitivity.get("warm_demotes", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) - weight
        
        if is_rainy:
            for cat, weight in self.weather_sensitivity.get("rain_boosts", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) + weight
            for cat, weight in self.weather_sensitivity.get("rain_demotes", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) - weight
        
        return adjustments
    
    def get_time_adjustments(
        self, 
        is_morning: bool = False, 
        is_evening: bool = False,
        is_weekend: bool = False
    ) -> Dict[str, float]:
        """Get aggregated time-based category adjustments."""
        adjustments: Dict[str, float] = {}
        
        if is_morning:
            for cat, weight in self.time_patterns.get("morning_boosts", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) + weight
            for cat, weight in self.time_patterns.get("morning_demotes", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) - weight
        
        if is_evening:
            for cat, weight in self.time_patterns.get("evening_boosts", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) + weight
            for cat, weight in self.time_patterns.get("evening_demotes", {}).items():
                adjustments[cat] = adjustments.get(cat, 0) - weight
        
        if is_weekend:
            for cat, weight in self.weekend_boosts.items():
                adjustments[cat] = adjustments.get(cat, 0) + weight
        
        return adjustments
    
    def get_combined_adjustments(
        self,
        is_cold: bool = False,
        is_warm: bool = False,
        is_rainy: bool = False,
        is_morning: bool = False,
        is_evening: bool = False,
        is_weekend: bool = False,
    ) -> Dict[str, float]:
        """Get all adjustments combined."""
        weather = self.get_weather_adjustments(is_cold, is_warm, is_rainy)
        time = self.get_time_adjustments(is_morning, is_evening, is_weekend)
        
        combined: Dict[str, float] = {}
        for cat, weight in weather.items():
            combined[cat] = combined.get(cat, 0) + weight
        for cat, weight in time.items():
            combined[cat] = combined.get(cat, 0) + weight
        
        return combined


# =============================================================================
# STORE PROFILES REGISTRY
# =============================================================================

def get_store_profile(store_id: str) -> StoreProfile:
    """Get the profile for a specific store."""
    store_id = store_id.lower().replace(" ", "_")
    
    if store_id not in STORE_IDS:
        # Return generic profile for unknown stores
        logger.warning(f"Unknown store ID: {store_id}, using generic profile")
        return StoreProfile(
            store_id=store_id,
            store_name=store_id.title(),
        )
    
    return StoreProfile(
        store_id=store_id,
        store_name=STORE_IDS[store_id],
        weather_sensitivity=STORE_WEATHER_SENSITIVITY.get(store_id, {}),
        time_patterns=STORE_TIME_PATTERNS.get(store_id, {}),
        weekend_boosts=STORE_WEEKEND_PATTERNS.get(store_id, {}),
    )


def get_all_store_profiles() -> Dict[str, StoreProfile]:
    """Get all store profiles."""
    return {
        store_id: get_store_profile(store_id) 
        for store_id in STORE_IDS.keys()
    }


# =============================================================================
# INTEGRATION HELPER
# =============================================================================

def get_store_regime_adjustments(
    store_id: str,
    temp_c: Optional[float] = None,
    is_rainy: bool = False,
    hour: Optional[int] = None,
    is_weekend: bool = False,
) -> Dict[str, Any]:
    """
    Get complete regime adjustments for a store given current conditions.
    
    This is the main integration point for the decision router.
    
    Returns:
        {
            "store_name": str,
            "category_boosts": List[str],
            "category_demotes": List[str],
            "boost_weights": Dict[str, float],
            "demote_weights": Dict[str, float],
            "explanation": str,
        }
    """
    profile = get_store_profile(store_id)
    
    # Determine conditions
    is_cold = temp_c is not None and temp_c < 10
    is_warm = temp_c is not None and temp_c > 18
    is_morning = hour is not None and 6 <= hour < 12
    is_evening = hour is not None and 18 <= hour < 22
    
    # Get combined adjustments
    adjustments = profile.get_combined_adjustments(
        is_cold=is_cold,
        is_warm=is_warm,
        is_rainy=is_rainy,
        is_morning=is_morning,
        is_evening=is_evening,
        is_weekend=is_weekend,
    )
    
    # Split into boosts and demotes
    boosts = {cat: weight for cat, weight in adjustments.items() if weight > 0}
    demotes = {cat: abs(weight) for cat, weight in adjustments.items() if weight < 0}
    
    # Build explanation
    conditions = []
    if is_cold:
        conditions.append(f"cold ({temp_c:.0f}°C)")
    if is_warm:
        conditions.append(f"warm ({temp_c:.0f}°C)")
    if is_rainy:
        conditions.append("rainy")
    if is_morning:
        conditions.append("morning")
    if is_evening:
        conditions.append("evening")
    if is_weekend:
        conditions.append("weekend")
    
    explanation = f"{profile.store_name} adjustments for {', '.join(conditions) or 'baseline'}"
    
    return {
        "store_name": profile.store_name,
        "category_boosts": list(boosts.keys()),
        "category_demotes": list(demotes.keys()),
        "boost_weights": boosts,
        "demote_weights": demotes,
        "explanation": explanation,
    }


# =============================================================================
# CLI TEST
# =============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("AOC Store-Specific Regimes Test")
    print("=" * 70)
    
    # Test each store
    for store_id in STORE_IDS.keys():
        profile = get_store_profile(store_id)
        print(f"\n📍 {profile.store_name}")
        print("-" * 40)
        
        # Cold evening scenario
        result = get_store_regime_adjustments(
            store_id=store_id,
            temp_c=5,
            is_rainy=False,
            hour=19,
            is_weekend=False,
        )
        print(f"   Cold Evening (5°C, 7pm, weekday):")
        print(f"   Boosts: {result['boost_weights']}")
        print(f"   Demotes: {result['demote_weights']}")
        
        # Warm weekend scenario
        result = get_store_regime_adjustments(
            store_id=store_id,
            temp_c=22,
            is_rainy=False,
            hour=14,
            is_weekend=True,
        )
        print(f"   Warm Weekend (22°C, 2pm, Saturday):")
        print(f"   Boosts: {result['boost_weights']}")
        print(f"   Demotes: {result['demote_weights']}")
        
        # Rainy morning
        result = get_store_regime_adjustments(
            store_id=store_id,
            temp_c=12,
            is_rainy=True,
            hour=10,
            is_weekend=False,
        )
        print(f"   Rainy Morning (12°C, 10am, rain):")
        print(f"   Boosts: {result['boost_weights']}")
        print(f"   Demotes: {result['demote_weights']}")
