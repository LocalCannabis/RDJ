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

    # Feature weights for similarity
    weights: Dict[str, float] = field(
        default_factory=lambda: {
            "dow": 2.0,
            "hour": 2.0,
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


def build_condition_vector(row: pd.Series, cfg: SimilarityConfig) -> np.ndarray:
    """Build a numeric condition vector from a row.

    The order of components must match what `compute_similarity` expects.
    """
    dow = int(row[cfg.dow_col])
    hour = int(row[cfg.hour_col])
    dow_sin, dow_cos, hour_sin, hour_cos = _encode_time_of_week(dow, hour)

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

    return np.array(
        [
            dow_sin,
            dow_cos,
            hour_sin,
            hour_cos,
            temp,
            precip,
            is_holiday,
            is_preholiday,
            is_payday,
            has_home_game,
            has_concert,
            has_festival,
            at_home,
            out_and_about,
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
    """
    if past_vec.shape != future_vec.shape:
        raise ValueError("Condition vectors must have the same shape.")

    # Map positions to logical feature groups to apply weights.
    # Index layout must stay in sync with `build_condition_vector`.
    idx = {
        "dow_sin": 0,
        "dow_cos": 1,
        "hour_sin": 2,
        "hour_cos": 3,
        "temp": 4,
        "precip": 5,
        "is_holiday": 6,
        "is_preholiday": 7,
        "is_payday": 8,
        "has_home_game": 9,
        "has_concert": 10,
        "has_festival": 11,
        "at_home": 12,
        "out_and_about": 13,
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
