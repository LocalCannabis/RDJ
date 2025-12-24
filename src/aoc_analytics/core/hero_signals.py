"""
Hero SKU identification and scoring.

Identifies products with stable, predictable demand patterns
that make them reliable "heroes" for promotional displays.
"""

from __future__ import annotations

import contextlib
import logging
import math
import warnings
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)

DAY_LOOKUP = {
    "Monday": 0,
    "Tuesday": 1,
    "Wednesday": 2,
    "Thursday": 3,
    "Friday": 4,
    "Saturday": 5,
    "Sunday": 6,
}


@dataclass
class HeroWeightingConfig:
    """Configuration for hero signal weighting."""

    min_observations: int = 5
    interval_std_target: float = 2.0
    hour_std_target: float = 3.0
    dow_std_target: float = 1.5
    entropy_floor: float = 0.1
    weather_corr_target: float = 0.3
    weekly_cv_target: float = 0.3

    interval_weight: float = 0.2
    autocorr_weight: float = 0.15
    entropy_weight: float = 0.15
    hour_weight: float = 0.2
    dow_weight: float = 0.1
    weather_weight: float = 0.1
    weekly_weight: float = 0.1

    def weight_map(self) -> dict[str, float]:
        return {
            "interval_score": self.interval_weight,
            "autocorr_score": self.autocorr_weight,
            "entropy_score": self.entropy_weight,
            "hour_score": self.hour_weight,
            "dow_score": self.dow_weight,
            "weather_score": self.weather_weight,
            "weekly_score": self.weekly_weight,
        }


class SkuBehaviorSignalProvider(Protocol):
    """Protocol for SKU behavior signal data source."""

    def get_signals_df(self) -> pd.DataFrame:
        """Return DataFrame with sku, datetime_local, quantity, date, hour, day_of_week, temp_c columns."""
        ...


def build_sku_behavior_signals(
    df: pd.DataFrame,
    config: HeroWeightingConfig | None = None,
) -> pd.DataFrame:
    """
    Compute per-SKU behavior signals for hero scoring.

    Returns DataFrame with columns:
    - sku
    - observation_count
    - interval_mean, interval_std, interval_score
    - lag1, lag7, lag14, autocorr_score
    - entropy_value, entropy_score
    - hour_std, hour_score
    - dow_score
    - weather_corr, weather_score
    - weekly_cv, weekly_score
    - hero_score (weighted combination)
    - confidence (based on observation count)
    - contextual_weight (hero_score * confidence)
    """
    config = config or HeroWeightingConfig()
    if df.empty:
        return pd.DataFrame()

    rows: list[dict[str, Any]] = []
    for sku, group in df.groupby("sku"):
        obs_count = len(group)
        if obs_count < config.min_observations:
            continue

        interval_score, interval_mean, interval_std = _interval_scores(group, config)
        autocorr = _autocorr_scores(group)
        entropy_score, entropy_value = _entropy_score(group, config)
        hour_score, hour_std = _hour_score(group, config)
        dow_score = _dow_score(group, config)
        weather_score, weather_corr = _weather_corr_score(group, config)
        weekly_score, weekly_cv = _weekly_score(group, config)

        components = {
            "interval_score": interval_score,
            "autocorr_score": autocorr.get("autocorr_score"),
            "entropy_score": entropy_score,
            "hour_score": hour_score,
            "dow_score": dow_score,
            "weather_score": weather_score,
            "weekly_score": weekly_score,
        }
        hero_score = _combine_scores(components, config)
        confidence = min(1.0, obs_count / 30)
        contextual_weight = hero_score * confidence

        rows.append(
            {
                "sku": sku,
                "observation_count": obs_count,
                "interval_mean": interval_mean,
                "interval_std": interval_std,
                "interval_score": interval_score,
                **{k: v for k, v in autocorr.items()},
                "entropy_value": entropy_value,
                "entropy_score": entropy_score,
                "hour_std": hour_std,
                "hour_score": hour_score,
                "dow_score": dow_score,
                "weather_corr": weather_corr,
                "weather_score": weather_score,
                "weekly_cv": weekly_cv,
                "weekly_score": weekly_score,
                "hero_score": hero_score,
                "confidence": confidence,
                "contextual_weight": contextual_weight,
            }
        )

    return pd.DataFrame(rows)


def _interval_scores(group: pd.DataFrame, config: HeroWeightingConfig) -> tuple[float, float | None, float | None]:
    timestamps = group["datetime_local"].sort_values()
    deltas = timestamps.diff().dt.total_seconds().dropna()
    if deltas.empty:
        return 0.0, None, None
    days = deltas / 86400.0
    interval_mean = float(days.mean())
    interval_std = float(days.std()) if len(days) > 1 else 0.0
    score = _inverse_score(interval_std, config.interval_std_target)
    return score, interval_mean, interval_std


def _autocorr_scores(group: pd.DataFrame) -> dict:
    daily = (
        group.groupby(group["date"].dt.date)["quantity"].sum().sort_index()
    )
    metrics = {"lag1": None, "lag7": None, "lag14": None, "autocorr_score": 0.0}
    if len(daily) < 2:
        return metrics
    for lag in (1, 7, 14):
        if len(daily) > lag:
            with _suppress_numpy_runtime_warnings(), np.errstate(divide="ignore", invalid="ignore"):
                metrics[f"lag{lag}"] = float(daily.autocorr(lag))
    best = max(abs(value) for value in metrics.values() if isinstance(value, (int, float))) if any(
        isinstance(value, (int, float)) for value in metrics.values()
    ) else 0.0
    metrics["autocorr_score"] = max(0.0, min(1.0, best))
    return metrics


def _entropy_score(group: pd.DataFrame, config: HeroWeightingConfig) -> tuple[float, float | None]:
    counts = group["quantity"].round(2).value_counts()
    if counts.empty:
        return 0.0, None
    probs = counts / counts.sum()
    entropy = -float((probs * np.log(probs)).sum())
    max_entropy = math.log(len(counts)) if len(counts) > 1 else 1.0
    normalized = entropy / max_entropy if max_entropy else 0.0
    score = max(0.0, 1.0 - max(0.0, normalized - config.entropy_floor) / (1 - config.entropy_floor))
    return score, entropy


def _hour_score(group: pd.DataFrame, config: HeroWeightingConfig) -> tuple[float, float | None]:
    if "hour" not in group:
        return 0.0, None
    hour_std = float(group["hour"].std()) if len(group) > 1 else 0.0
    score = _inverse_score(hour_std, config.hour_std_target)
    return score, hour_std


def _dow_score(group: pd.DataFrame, config: HeroWeightingConfig) -> float:
    if "day_of_week" not in group:
        return 0.0
    dow_numeric = group["day_of_week"].map(DAY_LOOKUP).dropna()
    if dow_numeric.empty:
        return 0.0
    std = float(dow_numeric.std()) if len(dow_numeric) > 1 else 0.0
    return _inverse_score(std, config.dow_std_target)


def _weather_corr_score(group: pd.DataFrame, config: HeroWeightingConfig) -> tuple[float, float | None]:
    if "temp_c" not in group or group["temp_c"].notna().sum() < config.min_observations:
        return 0.0, None
    with _suppress_numpy_runtime_warnings(), np.errstate(divide="ignore", invalid="ignore"):
        corr = group["quantity"].corr(group["temp_c"])
    if pd.isna(corr):
        return 0.0, None
    score = max(0.0, 1.0 - min(abs(corr) / config.weather_corr_target, 1.0))
    return score, float(corr)


def _weekly_score(group: pd.DataFrame, config: HeroWeightingConfig) -> tuple[float, float | None]:
    weekly = (
        group.set_index("date").resample("W")["quantity"].sum().dropna()
    )
    if weekly.empty or weekly.mean() == 0:
        return 0.0, None
    cv = float(weekly.std() / weekly.mean()) if weekly.mean() else None
    if cv is None:
        return 0.0, None
    score = _inverse_score(cv, config.weekly_cv_target)
    return score, cv


def _inverse_score(value: float | None, target: float) -> float:
    if value is None or target <= 0:
        return 0.0
    ratio = min(value / target, 1.0)
    return max(0.0, 1.0 - ratio)


def _combine_scores(components: dict[str, float], config: HeroWeightingConfig) -> float:
    weights = config.weight_map()
    total = 0.0
    score = 0.0
    for key, weight in weights.items():
        value = components.get(key)
        if value is None:
            continue
        score += weight * value
        total += weight
    if total == 0:
        return 0.0
    return score / total


def json_dumps(data: dict) -> str:
    import json

    return json.dumps(data, separators=(",", ":"))


@contextlib.contextmanager
def _suppress_numpy_runtime_warnings():
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="Degrees of freedom <= 0 for slice",
            module="numpy.lib._function_base_impl",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="divide by zero encountered",
            module="numpy.lib._function_base_impl",
        )
        warnings.filterwarnings(
            "ignore",
            category=RuntimeWarning,
            message="invalid value encountered in true_divide",
            module="numpy.lib._function_base_impl",
        )
        yield
