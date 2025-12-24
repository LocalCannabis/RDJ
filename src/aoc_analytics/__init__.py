"""
AOC Analytics - Weather-aware retail analytics for cannabis retail.

The keystone philosophy: Weather is not an insight — it's background radiation.
Weather conditions are transformed into behavioral propensities BEFORE any
correlation with sales. This is the a priori normalization that makes 
demand forecasting possible.

Usage:
    # Direct Python imports (for same-process integration)
    from aoc_analytics import rebuild_behavioral_signals, forecast_demand_for_slot
    
    # HTTP client (for cross-process/service integration)
    from aoc_analytics import AOCClient
    client = AOCClient("http://localhost:8081")
    forecast = client.forecast_demand(...)
"""

__version__ = "1.0.0"

# Core analytics functions
from .core import (
    # Signals builder (THE KEYSTONE)
    rebuild_behavioral_signals,
    # Forecasting
    SimilarityConfig,
    forecast_demand_for_slot,
    compute_similarity,
    # Hero scoring
    HeroWeightingConfig,
    build_sku_behavior_signals,
    # Anomaly registry
    Anomaly,
    AnomalyType,
    list_anomalies,
    create_anomaly,
    delete_anomaly,
    get_anomaly,
    get_anomalies_for_date_range,
    init_anomaly_table,
    seed_anomalies,
    add_anomaly_flags_to_df,
    compute_anomaly_adjusted_weight,
    # Conditions loading
    ConditionsConfig,
    load_conditions_df,
    load_category_conditions_df,
)

# HTTP client for external consumers
from .client import AOCClient

__all__ = [
    # Version
    "__version__",
    # Client
    "AOCClient",
    # Signals
    "rebuild_behavioral_signals",
    # Forecasting
    "SimilarityConfig",
    "forecast_demand_for_slot",
    "compute_similarity",
    # Hero scoring
    "HeroWeightingConfig",
    "build_sku_behavior_signals",
    # Anomaly registry
    "Anomaly",
    "AnomalyType",
    "list_anomalies",
    "create_anomaly",
    "delete_anomaly",
    "get_anomaly",
    "get_anomalies_for_date_range",
    "init_anomaly_table",
    "seed_anomalies",
    "add_anomaly_flags_to_df",
    "compute_anomaly_adjusted_weight",
    # Conditions
    "ConditionsConfig",
    "load_conditions_df",
    "load_category_conditions_df",
]
