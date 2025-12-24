"""
Core analytics modules.

This is THE KEYSTONE of AOC - the weather-as-a-priori normalization engine.

This package contains the core analytics engines:
- signals/: Behavioral signal computation (weather → propensities)
- predictor: Demand forecasting using weather-aware similarity matching
- hero_signals: SKU behavior analysis for hero product identification
- anomaly_registry: Tracking and down-weighting disrupted periods
- conditions: Unified conditions + outcomes data loading
"""

from .anomaly_registry import (
    Anomaly,
    AnomalyType,
    add_anomaly_flags_to_df,
    compute_anomaly_adjusted_weight,
    create_anomaly,
    delete_anomaly,
    get_anomalies_for_date_range,
    get_anomaly,
    init_anomaly_table,
    list_anomalies,
    seed_anomalies,
    update_anomaly,
)
from .conditions import (
    ConditionsConfig,
    load_category_conditions_df,
    load_conditions_df,
)
from .hero_signals import (
    HeroWeightingConfig,
    build_sku_behavior_signals,
)
from .predictor import (
    SimilarityConfig,
    compute_similarity,
    forecast_demand_for_slot,
)
from .signals.builder import rebuild_behavioral_signals
from .decision_router import (
    AOCDecisionRouter,
    Regime,
    WeatherContext,
    TimeContext,
)
from .recommender import SignageRecommenderV1
from .weather import (
    WeatherClient,
    HourlyWeather,
    DailyWeather,
    STORE_LOCATIONS,
    get_weather_for_location,
    get_all_store_weather,
)

__all__ = [
    # Anomaly registry
    "Anomaly",
    "AnomalyType",
    "add_anomaly_flags_to_df",
    "compute_anomaly_adjusted_weight",
    "create_anomaly",
    "delete_anomaly",
    "get_anomalies_for_date_range",
    "get_anomaly",
    "init_anomaly_table",
    "list_anomalies",
    "seed_anomalies",
    "update_anomaly",
    # Conditions
    "ConditionsConfig",
    "load_category_conditions_df",
    "load_conditions_df",
    # Hero signals
    "HeroWeightingConfig",
    "build_sku_behavior_signals",
    # Predictor
    "SimilarityConfig",
    "compute_similarity",
    "forecast_demand_for_slot",
    # Signals builder (the keystone)
    "rebuild_behavioral_signals",
    # Decision Router (brain switch)
    "AOCDecisionRouter",
    "Regime",
    "WeatherContext",
    "TimeContext",
    # Recommender
    "SignageRecommenderV1",
    # Weather
    "WeatherClient",
    "HourlyWeather",
    "DailyWeather",
    "STORE_LOCATIONS",
    "get_weather_for_location",
    "get_all_store_weather",
]
