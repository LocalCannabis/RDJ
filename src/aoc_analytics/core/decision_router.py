"""
AOC Decision Router

The "brain switch" that selects the optimal analysis lenses and weights
based on screen purpose, current context (weather, time, signals), and constraints.

This is the heart of AUTO mode - it decides what analytical approach
to use for generating recommendations.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# =============================================================================
# ENUMS & CONSTANTS
# =============================================================================

class ScreenPurpose(str, Enum):
    SIGNAGE = "SIGNAGE"
    ORDERING = "ORDERING"
    PROMO = "PROMO"
    STAFF_PICKS = "STAFF_PICKS"


class AnalysisLens(str, Enum):
    FUNNEL = "FUNNEL"           # Conversion/velocity weighted
    MARGIN_MIX = "MARGIN_MIX"   # Profit optimized
    BASKET = "BASKET"           # Cross-sell / pairing
    ELASTICITY = "ELASTICITY"   # Price sensitivity
    ABCXYZ = "ABCXYZ"           # Velocity + volatility matrix
    RFM = "RFM"                 # Customer segmentation (requires customer IDs)
    CLV = "CLV"                 # Customer lifetime value


class Regime(str, Enum):
    COZY_INDOOR = "cozy_indoor"
    SUNNY_OUTDOOR = "sunny_outdoor"
    RAINY_DAY = "rainy_day"  # Data-validated: flower +0.6pp, sativa -0.6pp
    PAYDAY_RUSH = "payday_rush"
    EVENING_WIND_DOWN = "evening_wind_down"
    WEEKEND_SOCIAL = "weekend_social"
    MORNING_FUNCTIONAL = "morning_functional"
    HOLIDAY_GIFTING = "holiday_gifting"
    BASELINE = "baseline"


class TimeHorizon(str, Enum):
    NOW = "NOW"
    TODAY = "TODAY"
    NEXT_7_DAYS = "NEXT_7_DAYS"


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class WeatherContext:
    """Current weather conditions."""
    temp_c: float
    feels_like_c: float
    precip_mm: float
    precip_type: str  # none, rain, snow, drizzle
    cloud_cover_pct: float
    humidity_pct: float
    wind_kph: float
    condition: str  # Clear sky, Rain, Snow, etc.
    
    @property
    def is_rainy(self) -> bool:
        return self.precip_mm > 0 or self.precip_type in ("rain", "drizzle")
    
    @property
    def is_snowy(self) -> bool:
        return self.precip_type == "snow"
    
    @property
    def is_cold(self) -> bool:
        return self.temp_c < 10
    
    @property
    def is_warm(self) -> bool:
        return self.temp_c > 18
    
    @property
    def is_clear(self) -> bool:
        return self.cloud_cover_pct < 30 and self.precip_mm == 0


@dataclass
class TimeContext:
    """Current time context."""
    hour: int
    day_of_week: int  # 0=Monday, 6=Sunday
    day_of_month: int
    month: int
    year: int
    is_holiday: bool = False
    holiday_name: Optional[str] = None
    
    @property
    def is_morning(self) -> bool:
        return 6 <= self.hour < 12
    
    @property
    def is_afternoon(self) -> bool:
        return 12 <= self.hour < 18
    
    @property
    def is_evening(self) -> bool:
        return 18 <= self.hour < 22
    
    @property
    def is_night(self) -> bool:
        return self.hour >= 22 or self.hour < 6
    
    @property
    def is_weekend(self) -> bool:
        return self.day_of_week >= 5  # Saturday=5, Sunday=6
    
    @property
    def is_end_of_month(self) -> bool:
        return self.day_of_month >= 25 or self.day_of_month <= 3
    
    @classmethod
    def now(cls) -> "TimeContext":
        dt = datetime.now()
        return cls(
            hour=dt.hour,
            day_of_week=dt.weekday(),
            day_of_month=dt.day,
            month=dt.month,
            year=dt.year,
        )


@dataclass
class SignalContext:
    """Behavioral signals from AOC."""
    payday_index: float = 0.5       # 0-1, higher = likely payday period
    at_home_index: float = 0.5      # 0-1, higher = stay-at-home behavior
    out_about_index: float = 0.5    # 0-1, higher = going-out behavior
    holiday_index: float = 0.0      # 0-1, higher = holiday shopping behavior
    
    @classmethod
    def default(cls) -> "SignalContext":
        return cls()


@dataclass
class RegimeConfig:
    """Configuration for a detected regime."""
    name: str
    category_boosts: List[str] = field(default_factory=list)
    category_demotes: List[str] = field(default_factory=list)
    lens_adjustments: Dict[str, float] = field(default_factory=dict)
    drivers: List[str] = field(default_factory=list)
    confidence: float = 0.5


@dataclass
class DecisionResult:
    """Output of the decision router."""
    selected_lenses: List[str]
    lens_weights: Dict[str, float]
    regime: RegimeConfig
    confidence: float
    explanation: str
    why_selected: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "selected_lenses": self.selected_lenses,
            "lens_weights": self.lens_weights,
            "regime": {
                "name": self.regime.name,
                "category_boosts": self.regime.category_boosts,
                "category_demotes": self.regime.category_demotes,
                "drivers": self.regime.drivers,
            },
            "confidence": self.confidence,
            "explanation": self.explanation,
            "why_selected": self.why_selected,
        }


# =============================================================================
# LENS DEFAULTS BY PURPOSE
# =============================================================================

PURPOSE_LENS_DEFAULTS: Dict[str, Dict[str, List[str]]] = {
    ScreenPurpose.SIGNAGE.value: {
        "primary": [AnalysisLens.FUNNEL.value, AnalysisLens.MARGIN_MIX.value],
        "secondary": [AnalysisLens.BASKET.value],
        "filter": [AnalysisLens.ABCXYZ.value],
    },
    ScreenPurpose.ORDERING.value: {
        "primary": [AnalysisLens.ABCXYZ.value, AnalysisLens.FUNNEL.value],
        "secondary": [AnalysisLens.ELASTICITY.value],
        "filter": [],
    },
    ScreenPurpose.PROMO.value: {
        "primary": [AnalysisLens.ELASTICITY.value, AnalysisLens.MARGIN_MIX.value],
        "secondary": [AnalysisLens.BASKET.value],
        "filter": [AnalysisLens.ABCXYZ.value],
    },
    ScreenPurpose.STAFF_PICKS.value: {
        "primary": [AnalysisLens.MARGIN_MIX.value, AnalysisLens.BASKET.value],
        "secondary": [AnalysisLens.FUNNEL.value],
        "filter": [],
    },
}


# =============================================================================
# REGIME DEFINITIONS
# Data-driven weights validated against 538,623 transactions (2021-2025)
# pp = percentage point lift vs baseline
# =============================================================================

REGIME_DEFINITIONS: Dict[str, Dict[str, Any]] = {
    # Cold weather (temp < 10°C) - 232K transactions analyzed
    Regime.COZY_INDOOR.value: {
        # VALIDATED: indica +0.7pp, extracts +0.3pp
        # INVALIDATED: flower/edibles were NOT significantly boosted
        "category_boosts": ["indica", "extracts", "sleep"],
        # VALIDATED: pre-rolls -1.2pp, beverages -0.3pp
        "category_demotes": ["pre-rolls", "beverages", "sativa"],
        "boost_weights": {"indica": 0.7, "extracts": 0.3},
        "demote_weights": {"pre-rolls": 1.2, "beverages": 0.3},
        "lens_adjustments": {
            AnalysisLens.MARGIN_MIX.value: 0.1,
            AnalysisLens.BASKET.value: 0.1,
        },
    },
    # Warm & dry weather (temp > 18°C, no rain) - 133K transactions analyzed
    Regime.SUNNY_OUTDOOR.value: {
        # VALIDATED: sativa +0.6pp, pre-rolls +0.5pp, beverages +0.3pp
        "category_boosts": ["sativa", "pre-rolls", "beverages", "energy"],
        # VALIDATED: flower -0.6pp, indica -0.5pp
        "category_demotes": ["flower", "indica"],
        "boost_weights": {"sativa": 0.6, "pre-rolls": 0.5, "beverages": 0.3},
        "demote_weights": {"flower": 0.6, "indica": 0.5},
        "lens_adjustments": {
            AnalysisLens.FUNNEL.value: 0.1,
        },
    },
    # Payday periods (1st, 15th, last Friday) - behavioral signal driven
    Regime.PAYDAY_RUSH.value: {
        "category_boosts": ["ounces", "bulk", "premium", "high-margin"],
        "category_demotes": [],
        "boost_weights": {"ounces": 1.0, "premium": 0.5},
        "demote_weights": {},
        "lens_adjustments": {
            AnalysisLens.FUNNEL.value: 0.15,
            AnalysisLens.MARGIN_MIX.value: 0.1,
        },
    },
    # Weekday evenings (5pm-9pm Mon-Fri) - 172K transactions analyzed
    Regime.EVENING_WIND_DOWN.value: {
        # VALIDATED: edibles +0.9pp, beverages +0.6pp, indica +0.4pp, pre-rolls +0.3pp
        "category_boosts": ["edibles", "beverages", "indica", "pre-rolls", "sleep"],
        # VALIDATED: sativa -0.4pp, flower -0.4pp, extracts -0.3pp
        "category_demotes": ["sativa", "flower", "extracts"],
        "boost_weights": {"edibles": 0.9, "beverages": 0.6, "indica": 0.4, "pre-rolls": 0.3},
        "demote_weights": {"sativa": 0.4, "flower": 0.4, "extracts": 0.3},
        "lens_adjustments": {
            AnalysisLens.BASKET.value: 0.1,
        },
    },
    # Weekend afternoon/evening (Sat/Sun 12pm+) - 132K transactions analyzed
    Regime.WEEKEND_SOCIAL.value: {
        # VALIDATED: edibles +1.3pp, beverages +0.8pp
        "category_boosts": ["edibles", "beverages", "party-packs"],
        # VALIDATED: flower -0.5pp, indica -0.5pp, pre-rolls -0.4pp
        "category_demotes": ["flower", "indica", "pre-rolls", "extracts"],
        "boost_weights": {"edibles": 1.3, "beverages": 0.8},
        "demote_weights": {"flower": 0.5, "indica": 0.5, "pre-rolls": 0.4, "extracts": 0.3},
        "lens_adjustments": {
            AnalysisLens.BASKET.value: 0.1,
            AnalysisLens.FUNNEL.value: 0.05,
        },
    },
    # Morning functional (6am-12pm weekdays) - morning pre-roll peak observed
    Regime.MORNING_FUNCTIONAL.value: {
        # Pre-rolls peak in morning (23.9% share)
        "category_boosts": ["pre-rolls", "sativa", "CBD", "energy"],
        "category_demotes": ["edibles", "indica", "sleep"],
        "boost_weights": {"pre-rolls": 0.5, "sativa": 0.3},
        "demote_weights": {"edibles": 0.5, "indica": 0.3},
        "lens_adjustments": {},
    },
    # Rainy day (precip > 0.5mm) - 51K transactions analyzed
    Regime.RAINY_DAY.value: {
        # VALIDATED: flower +0.6pp, edibles +0.3pp
        "category_boosts": ["flower", "edibles", "indica"],
        # VALIDATED: sativa -0.6pp, pre-rolls -0.4pp
        "category_demotes": ["sativa", "pre-rolls"],
        "boost_weights": {"flower": 0.6, "edibles": 0.3},
        "demote_weights": {"sativa": 0.6, "pre-rolls": 0.4},
        "lens_adjustments": {
            AnalysisLens.MARGIN_MIX.value: 0.1,
        },
    },
    Regime.HOLIDAY_GIFTING.value: {
        "category_boosts": ["gift-sets", "premium", "bundles", "accessories"],
        "category_demotes": ["value-packs", "budget"],
        "boost_weights": {"premium": 1.0, "gift-sets": 0.5},
        "demote_weights": {},
        "lens_adjustments": {
            AnalysisLens.MARGIN_MIX.value: 0.15,
        },
    },
    Regime.BASELINE.value: {
        "category_boosts": [],
        "category_demotes": [],
        "boost_weights": {},
        "demote_weights": {},
        "lens_adjustments": {},
    },
}


# =============================================================================
# DECISION ROUTER
# =============================================================================

class AOCDecisionRouter:
    """
    The brain switch - selects optimal analysis strategy based on context.
    
    Usage:
        router = AOCDecisionRouter()
        
        # With full context
        result = router.decide(
            purpose="SIGNAGE",
            weather=WeatherContext(...),
            time=TimeContext.now(),
            signals=SignalContext(...),
        )
        
        # Quick decision (uses defaults)
        result = router.decide(purpose="SIGNAGE")
        
        # With adaptive weights (live from database)
        router = AOCDecisionRouter(use_adaptive_weights=True)
    """
    
    def __init__(self, use_adaptive_weights: bool = False):
        """
        Initialize the decision router.
        
        Args:
            use_adaptive_weights: If True, use live weights from database.
                                  If False, use static REGIME_DEFINITIONS.
        """
        self.purpose_defaults = PURPOSE_LENS_DEFAULTS
        self.regime_definitions = REGIME_DEFINITIONS
        self.use_adaptive_weights = use_adaptive_weights
        self._adaptive_weights = None
        
        if use_adaptive_weights:
            self._load_adaptive_weights()
    
    def _load_adaptive_weights(self):
        """Load adaptive weights from database."""
        try:
            from .adaptive_weights import get_current_weights
            self._adaptive_weights = get_current_weights()
            if self._adaptive_weights:
                logger.info(f"Loaded adaptive weights v{self._adaptive_weights.version}")
            else:
                logger.warning("No adaptive weights found, falling back to static")
        except Exception as e:
            logger.warning(f"Could not load adaptive weights: {e}")
            self._adaptive_weights = None
    
    def get_regime_config(self, regime_name: str) -> Dict[str, Any]:
        """
        Get regime configuration, preferring adaptive weights if available.
        """
        # Try adaptive weights first
        if self.use_adaptive_weights and self._adaptive_weights:
            if regime_name in self._adaptive_weights.regime_weights:
                adaptive = self._adaptive_weights.regime_weights[regime_name]
                return {
                    "category_boosts": list(adaptive.category_boosts.keys()),
                    "category_demotes": list(adaptive.category_demotes.keys()),
                    "boost_weights": adaptive.category_boosts,
                    "demote_weights": adaptive.category_demotes,
                    "lens_adjustments": self.regime_definitions.get(regime_name, {}).get("lens_adjustments", {}),
                    "source": "adaptive",
                    "confidence": adaptive.confidence,
                    "sample_size": adaptive.sample_size,
                }
        
        # Fall back to static definitions
        static = self.regime_definitions.get(regime_name, {})
        return {
            **static,
            "source": "static",
        }
    
    def decide(
        self,
        purpose: str,
        time_horizon: str = TimeHorizon.NOW.value,
        weather: Optional[WeatherContext] = None,
        time: Optional[TimeContext] = None,
        signals: Optional[SignalContext] = None,
        constraints: Optional[Dict[str, Any]] = None,
    ) -> DecisionResult:
        """
        Main entry point - decide which lenses to use and their weights.
        
        Args:
            purpose: Screen purpose (SIGNAGE, ORDERING, PROMO, STAFF_PICKS)
            time_horizon: NOW, TODAY, or NEXT_7_DAYS
            weather: Current weather context
            time: Current time context
            signals: Behavioral signals from AOC
            constraints: Screen constraints (optional)
        
        Returns:
            DecisionResult with selected lenses, weights, and explanation
        """
        # Use defaults if context not provided
        time = time or TimeContext.now()
        signals = signals or SignalContext.default()
        
        # Detect current regime
        regime = self._detect_regime(weather, time, signals)
        
        # Get base lens weights for purpose
        lens_weights = self._calculate_base_weights(purpose)
        
        # Apply regime adjustments
        lens_weights = self._apply_regime_adjustments(lens_weights, regime)
        
        # Apply time horizon adjustments
        lens_weights = self._apply_time_horizon_adjustments(lens_weights, time_horizon)
        
        # Normalize weights
        lens_weights = self._normalize_weights(lens_weights)
        
        # Build explanation
        selected_lenses = [lens for lens in lens_weights.keys() if lens_weights[lens] > 0]
        explanation = self._build_explanation(purpose, regime, time_horizon, weather)
        why_selected = self._build_why_selected(selected_lenses, regime, time)
        
        # Calculate overall confidence
        confidence = self._calculate_confidence(weather, signals, regime)
        
        return DecisionResult(
            selected_lenses=selected_lenses,
            lens_weights=lens_weights,
            regime=regime,
            confidence=confidence,
            explanation=explanation,
            why_selected=why_selected,
        )
    
    def _detect_regime(
        self,
        weather: Optional[WeatherContext],
        time: TimeContext,
        signals: SignalContext,
    ) -> RegimeConfig:
        """
        Detect the current regime based on context.
        Returns the highest-scoring regime.
        """
        regime_scores: Dict[str, Tuple[float, List[str]]] = {}
        
        # Score each regime
        for regime_name, config in self.regime_definitions.items():
            score, drivers = self._score_regime(regime_name, weather, time, signals)
            regime_scores[regime_name] = (score, drivers)
        
        # Find best regime
        best_regime = max(regime_scores.keys(), key=lambda k: regime_scores[k][0])
        best_score, best_drivers = regime_scores[best_regime]
        
        # Fall back to baseline if no strong signal
        if best_score < 0.3:
            best_regime = Regime.BASELINE.value
            best_drivers = ["No strong regime signals detected"]
        
        # Get config (adaptive or static)
        config = self.get_regime_config(best_regime)
        
        # Add source info to drivers
        if config.get("source") == "adaptive":
            best_drivers.append(f"Using adaptive weights v{self._adaptive_weights.version}")
        
        return RegimeConfig(
            name=best_regime,
            category_boosts=config.get("category_boosts", []),
            category_demotes=config.get("category_demotes", []),
            lens_adjustments=config.get("lens_adjustments", {}),
            drivers=best_drivers,
            confidence=best_score,
        )
    
    def _score_regime(
        self,
        regime_name: str,
        weather: Optional[WeatherContext],
        time: TimeContext,
        signals: SignalContext,
    ) -> Tuple[float, List[str]]:
        """Score how well current context matches a regime."""
        triggers_met = 0
        total_triggers = 0
        drivers = []
        
        if regime_name == Regime.COZY_INDOOR.value:
            total_triggers = 3
            if weather:
                if weather.is_cold:
                    triggers_met += 1
                    drivers.append(f"Cold: {weather.temp_c:.0f}°C")
                if weather.is_rainy or weather.is_snowy:
                    triggers_met += 1
                    drivers.append(f"Precipitation: {weather.condition}")
                if weather.cloud_cover_pct > 70:
                    triggers_met += 0.5
                    drivers.append(f"Cloudy: {weather.cloud_cover_pct:.0f}%")
            if signals.at_home_index > 0.6:
                triggers_met += 0.5
                drivers.append(f"At-home index: {signals.at_home_index:.2f}")
        
        elif regime_name == Regime.SUNNY_OUTDOOR.value:
            total_triggers = 3
            if weather:
                if weather.is_warm:
                    triggers_met += 1
                    drivers.append(f"Warm: {weather.temp_c:.0f}°C")
                if weather.precip_mm == 0:
                    triggers_met += 0.5
                    drivers.append("No precipitation")
                if weather.is_clear:
                    triggers_met += 1
                    drivers.append(f"Clear sky: {weather.cloud_cover_pct:.0f}% clouds")
            if signals.out_about_index > 0.6:
                triggers_met += 0.5
                drivers.append(f"Out-about index: {signals.out_about_index:.2f}")
        
        # Data-validated: flower +0.6pp, edibles +0.3pp on rainy days
        elif regime_name == Regime.RAINY_DAY.value:
            total_triggers = 2
            if weather:
                if weather.is_rainy:
                    triggers_met += 1.5
                    drivers.append(f"Rain: {weather.precip_mm:.1f}mm")
                elif weather.precip_mm > 0:
                    triggers_met += 0.5
                    drivers.append(f"Light precip: {weather.precip_mm:.1f}mm")
            if signals.at_home_index > 0.6:
                triggers_met += 0.5
                drivers.append(f"At-home index: {signals.at_home_index:.2f}")
        
        elif regime_name == Regime.PAYDAY_RUSH.value:
            total_triggers = 2
            if signals.payday_index > 0.7:
                triggers_met += 1
                drivers.append(f"Payday index: {signals.payday_index:.2f}")
            if time.is_end_of_month:
                triggers_met += 1
                drivers.append("End of month")
        
        elif regime_name == Regime.EVENING_WIND_DOWN.value:
            total_triggers = 2
            if time.is_evening:
                triggers_met += 1
                drivers.append(f"Evening hour: {time.hour}:00")
            if not time.is_weekend:
                triggers_met += 0.5
                drivers.append("Weekday")
        
        elif regime_name == Regime.WEEKEND_SOCIAL.value:
            total_triggers = 2
            if time.is_weekend:
                triggers_met += 1
                drivers.append("Weekend")
            if time.is_afternoon or time.is_evening:
                triggers_met += 1
                drivers.append(f"Social hours: {time.hour}:00")
        
        elif regime_name == Regime.MORNING_FUNCTIONAL.value:
            total_triggers = 2
            if time.is_morning:
                triggers_met += 1
                drivers.append(f"Morning: {time.hour}:00")
            if not time.is_weekend:
                triggers_met += 1
                drivers.append("Weekday")
        
        elif regime_name == Regime.HOLIDAY_GIFTING.value:
            total_triggers = 2
            if time.is_holiday:
                triggers_met += 1.5
                drivers.append(f"Holiday: {time.holiday_name or 'detected'}")
            if signals.holiday_index > 0.6:
                triggers_met += 0.5
                drivers.append(f"Holiday index: {signals.holiday_index:.2f}")
        
        elif regime_name == Regime.BASELINE.value:
            # Baseline always scores low
            total_triggers = 1
            triggers_met = 0.1
        
        score = triggers_met / total_triggers if total_triggers > 0 else 0
        return score, drivers
    
    def _calculate_base_weights(self, purpose: str) -> Dict[str, float]:
        """Get base lens weights for a purpose."""
        defaults = self.purpose_defaults.get(purpose, self.purpose_defaults[ScreenPurpose.SIGNAGE.value])
        
        weights = {}
        
        # Primary lenses get higher weight
        for lens in defaults.get("primary", []):
            weights[lens] = 0.35
        
        # Secondary lenses get medium weight
        for lens in defaults.get("secondary", []):
            weights[lens] = 0.15
        
        # Filter lenses get low weight (used for diversity/stability)
        for lens in defaults.get("filter", []):
            weights[lens] = 0.05
        
        return weights
    
    def _apply_regime_adjustments(
        self,
        weights: Dict[str, float],
        regime: RegimeConfig,
    ) -> Dict[str, float]:
        """Apply regime-specific lens weight adjustments."""
        for lens, adjustment in regime.lens_adjustments.items():
            if lens in weights:
                weights[lens] += adjustment
            else:
                weights[lens] = adjustment
        
        return weights
    
    def _apply_time_horizon_adjustments(
        self,
        weights: Dict[str, float],
        time_horizon: str,
    ) -> Dict[str, float]:
        """Apply time horizon adjustments."""
        if time_horizon == TimeHorizon.NEXT_7_DAYS.value:
            # Stability matters more for longer horizons
            abcxyz = AnalysisLens.ABCXYZ.value
            if abcxyz in weights:
                weights[abcxyz] += 0.1
            else:
                weights[abcxyz] = 0.1
        
        return weights
    
    def _normalize_weights(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Normalize weights to sum to 1.0."""
        total = sum(weights.values())
        if total == 0:
            return weights
        return {k: v / total for k, v in weights.items()}
    
    def _calculate_confidence(
        self,
        weather: Optional[WeatherContext],
        signals: SignalContext,
        regime: RegimeConfig,
    ) -> float:
        """Calculate overall confidence in the decision."""
        confidence = 0.5  # Base confidence
        
        # Higher confidence with more context
        if weather:
            confidence += 0.15
        
        # Higher confidence with stronger signals
        signal_strength = max(
            signals.payday_index,
            signals.at_home_index,
            signals.out_about_index,
            signals.holiday_index,
        )
        confidence += signal_strength * 0.2
        
        # Higher confidence with stronger regime match
        confidence += regime.confidence * 0.15
        
        return min(confidence, 1.0)
    
    def _build_explanation(
        self,
        purpose: str,
        regime: RegimeConfig,
        time_horizon: str,
        weather: Optional[WeatherContext],
    ) -> str:
        """Build human-readable explanation."""
        parts = []
        
        # Regime explanation
        if regime.name != Regime.BASELINE.value:
            regime_desc = regime.name.replace("_", " ").title()
            parts.append(f"{regime_desc} regime detected")
            
            if weather:
                parts.append(f"({weather.temp_c:.0f}°C, {weather.condition})")
        
        # Boost explanation
        if regime.category_boosts:
            boosts = ", ".join(regime.category_boosts[:3])
            parts.append(f"→ Prioritizing {boosts}")
        
        # Time horizon note
        if time_horizon == TimeHorizon.NEXT_7_DAYS.value:
            parts.append("Favoring stable performers for 7-day horizon")
        
        return ". ".join(parts) if parts else f"Standard {purpose.lower()} recommendations"
    
    def _build_why_selected(
        self,
        lenses: List[str],
        regime: RegimeConfig,
        time: TimeContext,
    ) -> List[str]:
        """Build list of reasons for lens selection."""
        reasons = []
        
        for lens in lenses:
            if lens == AnalysisLens.FUNNEL.value:
                reasons.append("FUNNEL: Prioritizing high-velocity products")
            elif lens == AnalysisLens.MARGIN_MIX.value:
                reasons.append("MARGIN_MIX: Optimizing for profit contribution")
            elif lens == AnalysisLens.BASKET.value:
                reasons.append("BASKET: Including cross-sell opportunities")
            elif lens == AnalysisLens.ELASTICITY.value:
                reasons.append("ELASTICITY: Highlighting price-sensitive items")
            elif lens == AnalysisLens.ABCXYZ.value:
                reasons.append("ABCXYZ: Balancing velocity with stability")
        
        if regime.drivers:
            reasons.append(f"Regime drivers: {', '.join(regime.drivers[:2])}")
        
        return reasons


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def get_decision_for_screen(
    purpose: str,
    store_id: str,
    weather_data: Optional[Dict[str, Any]] = None,
) -> DecisionResult:
    """
    Convenience function to get a decision for a screen.
    
    Args:
        purpose: Screen purpose
        store_id: Store identifier
        weather_data: Optional weather dict with temp_c, precip_mm, etc.
    
    Returns:
        DecisionResult
    """
    router = AOCDecisionRouter()
    
    weather = None
    if weather_data:
        weather = WeatherContext(
            temp_c=weather_data.get("temp_c", 15),
            feels_like_c=weather_data.get("feels_like_c", 15),
            precip_mm=weather_data.get("precip_mm", 0),
            precip_type=weather_data.get("precip_type", "none"),
            cloud_cover_pct=weather_data.get("cloud_cover_pct", 50),
            humidity_pct=weather_data.get("humidity_pct", 60),
            wind_kph=weather_data.get("wind_kph", 10),
            condition=weather_data.get("condition", "Partly cloudy"),
        )
    
    return router.decide(
        purpose=purpose,
        weather=weather,
        time=TimeContext.now(),
    )
