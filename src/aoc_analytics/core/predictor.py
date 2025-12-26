"""
Matched-Conditions Demand Forecasting Engine

Core forecasting algorithm using weather-aware similarity matching
to predict demand based on historical conditions and outcomes.

This forecaster uses context-normalized signals (from the behavioral
signals builder) to find similar historical periods and predict outcomes.

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
All Rights Reserved. Proprietary and Confidential.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd


@dataclass
class SimilarityConfig:
    """Configuration for building condition vectors and computing similarity.

    This is intentionally conservative for v1. We restrict ourselves to a small
    set of well-understood features and make it easy to extend later.
    """

    # Columns to use for time-of-week encoding
    dow_col: str = "dow"
    hour_col: str = "hour"
    
    # NEW: Seasonality columns (critical for accuracy!)
    month_col: str = "month"
    day_of_year_col: str = "day_of_year"

    # Weather columns
    temp_col: str = "temp_c"
    precip_col: str = "precip_mm"

    # Calendar / event columns
    is_holiday_col: str = "is_holiday"
    is_preholiday_col: str = "is_preholiday"
    is_payday_col: str = "is_payday_window"
    has_home_game_col: str = "has_home_game"
    has_concert_col: str = "has_concert"
    has_festival_col: str = "has_festival"

    # Vibe columns (from behavioral signals - the a priori layer)
    at_home_index_col: str = "local_vibe_at_home_index"
    out_and_about_index_col: str = "local_vibe_out_and_about_index"
    
    # Sunday-specific columns (high-variance day)
    is_sunday_col: str = "is_sunday"
    is_nfl_sunday_col: str = "is_nfl_sunday"
    
    # NEW: Vibe signal columns (from vibe_signals.py)
    couch_index_col: str = "couch_index"
    party_index_col: str = "party_index"
    stress_index_col: str = "stress_index"
    has_major_event_col: str = "has_major_event"

    # Feature weights for similarity
    # TUNED: Based on Parksville backtesting Dec 2024
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "dow": 4.0,          # Day of week is crucial
            "hour": 2.0,
            # NEW: Seasonality weights - critical for accuracy
            "month": 5.0,        # Match same month strongly
            "season": 3.0,       # Cyclical season encoding
            "temp": 1.0,
            "precip": 0.5,
            "holiday": 3.0,
            "preholiday": 2.5,
            "payday": 2.0,
            "home_game": 3.0,
            "concert": 2.0,
            "festival": 2.0,
            "at_home": 1.0,
            "out_and_about": 1.0,
            "sunday": 3.0,       # High weight - Sundays are different
            "nfl_sunday": 2.0,   # NFL Sundays are extra different
            # Vibe signal weights - DISABLED: No significant impact in backtests
            "couch": 0.0,
            "party": 0.0,
            "stress": 0.0,
            "major_event": 0.0,
        }
    )


def _encode_time_of_week(dow: int, hour: int) -> Tuple[float, float, float, float]:
    """Encode day-of-week and hour as simple cyclical features.

    We use sin/cos transforms to make Monday and Sunday close in the embedding
    space. This is a light-weight alternative to one-hot encoding.
    """
    dow_rad = 2 * np.pi * (dow % 7) / 7.0
    hour_rad = 2 * np.pi * (hour % 24) / 24.0
    return np.sin(dow_rad), np.cos(dow_rad), np.sin(hour_rad), np.cos(hour_rad)


def _encode_season(month: int, day_of_year: int) -> Tuple[float, float, float]:
    """Encode seasonality as cyclical features.
    
    Returns:
        month_sin, month_cos: Cyclical month encoding (December close to January)
        season_progress: Linear progress through the year (0-1)
    """
    month_rad = 2 * np.pi * (month - 1) / 12.0
    season_progress = day_of_year / 365.0
    return np.sin(month_rad), np.cos(month_rad), season_progress


def build_condition_vector(row: pd.Series, cfg: SimilarityConfig) -> np.ndarray:
    """Build a numeric condition vector from a row.

    The order of components must match what `compute_similarity` expects.
    Vector is now 23 dimensions (was 20).
    """
    dow = int(row[cfg.dow_col])
    hour = int(row[cfg.hour_col])
    dow_sin, dow_cos, hour_sin, hour_cos = _encode_time_of_week(dow, hour)
    
    # NEW: Seasonality features
    month = int(row.get(cfg.month_col, 6))  # Default to June if missing
    day_of_year = int(row.get(cfg.day_of_year_col, 180))  # Default to mid-year
    month_sin, month_cos, season_progress = _encode_season(month, day_of_year)

    temp = float(row.get(cfg.temp_col, np.nan))
    precip = float(row.get(cfg.precip_col, 0.0) or 0.0)

    is_holiday = float(row.get(cfg.is_holiday_col, 0) or 0)
    is_preholiday = float(row.get(cfg.is_preholiday_col, 0) or 0)
    is_payday = float(row.get(cfg.is_payday_col, 0) or 0)
    has_home_game = float(row.get(cfg.has_home_game_col, 0) or 0)
    has_concert = float(row.get(cfg.has_concert_col, 0) or 0)
    has_festival = float(row.get(cfg.has_festival_col, 0) or 0)

    at_home = float(row.get(cfg.at_home_index_col, 0.0) or 0.0)
    out_and_about = float(row.get(cfg.out_and_about_index_col, 0.0) or 0.0)
    
    # Sunday-specific features
    is_sunday = float(row.get(cfg.is_sunday_col, 0) or 0)
    is_nfl_sunday = float(row.get(cfg.is_nfl_sunday_col, 0) or 0)
    
    # Vibe signal features
    couch_index = float(row.get(cfg.couch_index_col, 0.5) or 0.5)
    party_index = float(row.get(cfg.party_index_col, 0.0) or 0.0)
    stress_index = float(row.get(cfg.stress_index_col, 0.0) or 0.0)
    has_major_event = float(row.get(cfg.has_major_event_col, 0) or 0)

    return np.array(
        [
            dow_sin,
            dow_cos,
            hour_sin,
            hour_cos,
            # NEW: Seasonality (3 features)
            month_sin,
            month_cos,
            season_progress,
            # Weather
            temp,
            precip,
            # Calendar/events
            is_holiday,
            is_preholiday,
            is_payday,
            has_home_game,
            has_concert,
            has_festival,
            at_home,
            out_and_about,
            is_sunday,
            is_nfl_sunday,
            # Vibe features
            couch_index,
            party_index,
            stress_index,
            has_major_event,
        ],
        dtype=float,
    )


def compute_similarity(
    past_vec: np.ndarray,
    future_vec: np.ndarray,
    cfg: SimilarityConfig,
) -> float:
    """Compute a similarity score between two condition vectors.

    We currently implement a negative weighted L1 distance: higher scores
    indicate *more* similar conditions.
    
    Vector is now 23 dimensions (was 20).
    """
    if past_vec.shape != future_vec.shape:
        raise ValueError("Condition vectors must have the same shape.")

    # Map positions to logical feature groups to apply weights.
    # Index layout must stay in sync with `build_condition_vector`.
    # NEW: Added month_sin, month_cos, season_progress at indices 4-6
    idx = {
        "dow_sin": 0,
        "dow_cos": 1,
        "hour_sin": 2,
        "hour_cos": 3,
        # NEW: Seasonality
        "month_sin": 4,
        "month_cos": 5,
        "season_progress": 6,
        # Weather (shifted by 3)
        "temp": 7,
        "precip": 8,
        # Calendar/events (shifted by 3)
        "is_holiday": 9,
        "is_preholiday": 10,
        "is_payday": 11,
        "has_home_game": 12,
        "has_concert": 13,
        "has_festival": 14,
        "at_home": 15,
        "out_and_about": 16,
        "is_sunday": 17,
        "is_nfl_sunday": 18,
        # Vibe signals (shifted by 3)
        "couch_index": 19,
        "party_index": 20,
        "stress_index": 21,
        "has_major_event": 22,
    }

    weights = cfg.weights

    distance = 0.0
    
    # Time-of-week (combined weight across sin/cos pairs)
    distance += weights.get("dow", 1.0) * (
        abs(past_vec[idx["dow_sin"]] - future_vec[idx["dow_sin"]])
        + abs(past_vec[idx["dow_cos"]] - future_vec[idx["dow_cos"]])
    ) / 2.0
    distance += weights.get("hour", 1.0) * (
        abs(past_vec[idx["hour_sin"]] - future_vec[idx["hour_sin"]])
        + abs(past_vec[idx["hour_cos"]] - future_vec[idx["hour_cos"]])
    ) / 2.0
    
    # NEW: Seasonality (critical for accuracy!)
    distance += weights.get("month", 5.0) * (
        abs(past_vec[idx["month_sin"]] - future_vec[idx["month_sin"]])
        + abs(past_vec[idx["month_cos"]] - future_vec[idx["month_cos"]])
    ) / 2.0
    distance += weights.get("season", 3.0) * abs(
        past_vec[idx["season_progress"]] - future_vec[idx["season_progress"]]
    )

    # Weather
    distance += weights.get("temp", 1.0) * abs(past_vec[idx["temp"]] - future_vec[idx["temp"]])
    distance += weights.get("precip", 0.5) * abs(past_vec[idx["precip"]] - future_vec[idx["precip"]])

    # Calendar / events
    distance += weights.get("holiday", 1.0) * abs(
        past_vec[idx["is_holiday"]] - future_vec[idx["is_holiday"]]
    )
    distance += weights.get("preholiday", 1.0) * abs(
        past_vec[idx["is_preholiday"]] - future_vec[idx["is_preholiday"]]
    )
    distance += weights.get("payday", 1.0) * abs(
        past_vec[idx["is_payday"]] - future_vec[idx["is_payday"]]
    )
    distance += weights.get("home_game", 1.0) * abs(
        past_vec[idx["has_home_game"]] - future_vec[idx["has_home_game"]]
    )
    distance += weights.get("concert", 1.0) * abs(
        past_vec[idx["has_concert"]] - future_vec[idx["has_concert"]]
    )
    distance += weights.get("festival", 1.0) * abs(
        past_vec[idx["has_festival"]] - future_vec[idx["has_festival"]]
    )

    # Vibe indices (from a priori weather normalization)
    distance += weights.get("at_home", 1.0) * abs(
        past_vec[idx["at_home"]] - future_vec[idx["at_home"]]
    )
    distance += weights.get("out_and_about", 1.0) * abs(
        past_vec[idx["out_and_about"]] - future_vec[idx["out_and_about"]]
    )
    
    # Sunday-specific features (high variance day)
    distance += weights.get("sunday", 3.0) * abs(
        past_vec[idx["is_sunday"]] - future_vec[idx["is_sunday"]]
    )
    distance += weights.get("nfl_sunday", 2.0) * abs(
        past_vec[idx["is_nfl_sunday"]] - future_vec[idx["is_nfl_sunday"]]
    )
    
    # NEW: Vibe signal features
    distance += weights.get("couch", 1.5) * abs(
        past_vec[idx["couch_index"]] - future_vec[idx["couch_index"]]
    )
    distance += weights.get("party", 2.0) * abs(
        past_vec[idx["party_index"]] - future_vec[idx["party_index"]]
    )
    distance += weights.get("stress", 1.0) * abs(
        past_vec[idx["stress_index"]] - future_vec[idx["stress_index"]]
    )
    distance += weights.get("major_event", 2.5) * abs(
        past_vec[idx["has_major_event"]] - future_vec[idx["has_major_event"]]
    )

    return -float(distance)


def forecast_demand_for_slot(
    conditions_df: pd.DataFrame,
    future_row: pd.Series | Dict[str, Any],
    *,
    k_neighbors: int = 100,
    outcome_cols: Optional[Sequence[str]] = None,
    cfg: Optional[SimilarityConfig] = None,
    anomaly_weight_func: Optional[callable] = None,
) -> Dict[str, float]:
    """Forecast demand for a single future time bucket via matched conditions.

    Parameters
    ----------
    conditions_df:
        Historical conditions + outcomes table (from `load_conditions_df`).
    future_row:
        Dict-like object with at least the columns required by
        `SimilarityConfig` (dow, hour, temp, precip, calendar + vibe flags).
    k_neighbors:
        How many of the most similar past rows to use when aggregating.
    outcome_cols:
        Columns in `conditions_df` to aggregate as demand signals. If omitted,
        defaults to `['sales_units', 'sales_revenue']` when available.
    cfg:
        Similarity configuration; defaults to `SimilarityConfig()`.
    anomaly_weight_func:
        Optional function to compute anomaly-adjusted weights.

    Returns
    -------
    dict
        A flat dict mapping outcome names to aggregated forecast values
        (currently simple means), with a few basic diagnostics.
    """
    if cfg is None:
        cfg = SimilarityConfig()

    if conditions_df.empty:
        raise ValueError("conditions_df is empty; cannot forecast demand.")

    if outcome_cols is None:
        default_cols: List[str] = []
        for col in ("sales_units", "sales_revenue"):
            if col in conditions_df.columns:
                default_cols.append(col)
        if not default_cols:
            raise ValueError(
                "No outcome_cols provided and no default outcome columns "
                "('sales_units', 'sales_revenue') present in conditions_df."
            )
        outcome_cols = default_cols

    # Build vectors for all past rows and the future scenario.
    past_vectors = []
    for _, row in conditions_df.iterrows():
        past_vectors.append(build_condition_vector(row, cfg))
    past_vectors_arr = np.vstack(past_vectors)

    if not isinstance(future_row, pd.Series):
        future_row = pd.Series(future_row)
    future_vec = build_condition_vector(future_row, cfg)

    # Compute similarity scores.
    scores = np.apply_along_axis(
        lambda v: compute_similarity(v, future_vec, cfg), 1, past_vectors_arr
    )

    # Select top-k neighbors.
    k = min(k_neighbors, len(scores))
    neighbor_idx = np.argpartition(scores, -k)[-k:]
    neighbor_df = conditions_df.iloc[neighbor_idx].copy()
    neighbor_scores = scores[neighbor_idx]

    result: Dict[str, float] = {}

    # Check if anomaly columns exist for weighted aggregation
    has_anomaly_info = (
        "is_anomaly_period" in neighbor_df.columns 
        and "anomaly_severity" in neighbor_df.columns
        and anomaly_weight_func is not None
    )

    if has_anomaly_info:
        # Apply anomaly-adjusted weighting
        min_score = neighbor_scores.min()
        shifted_scores = neighbor_scores - min_score + 1e-6

        anomaly_weights = neighbor_df.apply(
            lambda row: anomaly_weight_func(
                base_weight=1.0,
                anomaly_severity=row.get("anomaly_severity", 1.0),
                is_anomaly_period=row.get("is_anomaly_period", False),
            ),
            axis=1
        ).values

        combined_weights = shifted_scores * anomaly_weights

        weight_sum = combined_weights.sum()
        if weight_sum > 0:
            normalized_weights = combined_weights / weight_sum
        else:
            normalized_weights = np.ones(len(combined_weights)) / len(combined_weights)

        for col in outcome_cols:
            result[f"expected_{col}"] = float((neighbor_df[col].values * normalized_weights).sum())

        anomaly_mask = neighbor_df["is_anomaly_period"].values
        result["anomaly_neighbors"] = float(anomaly_mask.sum())
        result["anomaly_weight_fraction"] = (
            float(normalized_weights[anomaly_mask].sum()) if anomaly_mask.any() else 0.0
        )
    else:
        # No anomaly info - use simple mean
        for col in outcome_cols:
            result[f"expected_{col}"] = float(neighbor_df[col].mean())
        result["anomaly_neighbors"] = 0.0
        result["anomaly_weight_fraction"] = 0.0

    # Diagnostics
    result["neighbors_used"] = float(len(neighbor_df))
    result["similarity_mean"] = float(neighbor_scores.mean())
    result["similarity_min"] = float(neighbor_scores.min())
    result["similarity_max"] = float(neighbor_scores.max())

    return result
