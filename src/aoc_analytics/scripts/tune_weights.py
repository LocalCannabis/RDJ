#!/usr/bin/env python3
"""
Feature Weight Tuning Script

Tests different weight configurations to find optimal values.
Uses grid search over key features.
"""

import pandas as pd
import sqlite3
import numpy as np
from datetime import date
from typing import Dict, List, Tuple
import time
from dataclasses import dataclass

from aoc_analytics.core.predictor import SimilarityConfig
from aoc_analytics.core.forecast_engine import ForecastConfig
from aoc_analytics.core.backtesting import BacktestRunner


@dataclass
class TuningResult:
    """Result of a weight configuration test."""
    weights: Dict[str, float]
    mape: float
    mae: float
    bias: float
    runtime: float


def run_quick_backtest(
    sales_df: pd.DataFrame,
    store: str,
    weights: Dict[str, float],
) -> TuningResult:
    """Run a quick backtest with custom weights."""
    start = time.time()
    
    # Create config with custom weights
    sim_cfg = SimilarityConfig(weights=weights)
    forecast_cfg = ForecastConfig(
        similarity_config=sim_cfg,
        skip_categories=True,
    )
    
    runner = BacktestRunner(
        store=store,
        train_window_days=365,
        test_window_days=14,  # Larger test window for speed
        config=forecast_cfg,
    )
    
    result = runner.run(sales_df)
    runtime = time.time() - start
    
    return TuningResult(
        weights=weights,
        mape=result.metrics.revenue_mape,
        mae=result.metrics.revenue_mae,
        bias=result.metrics.revenue_bias,
        runtime=runtime,
    )


def grid_search_weights(
    sales_df: pd.DataFrame,
    store: str,
    feature_ranges: Dict[str, List[float]],
    base_weights: Dict[str, float],
) -> List[TuningResult]:
    """
    Grid search over feature weight combinations.
    
    Only varies specified features, keeps others at base values.
    """
    results = []
    
    # Generate all combinations
    from itertools import product
    
    features = list(feature_ranges.keys())
    value_lists = [feature_ranges[f] for f in features]
    
    total = np.prod([len(v) for v in value_lists])
    print(f"Testing {total} weight combinations...")
    
    for i, values in enumerate(product(*value_lists)):
        weights = base_weights.copy()
        for feat, val in zip(features, values):
            weights[feat] = val
        
        result = run_quick_backtest(sales_df, store, weights)
        results.append(result)
        
        if (i + 1) % 5 == 0:
            print(f"  {i+1}/{total}: MAPE={result.mape:.1f}%")
    
    return results


def analyze_feature_importance(results: List[TuningResult]) -> Dict[str, float]:
    """
    Analyze which features have most impact on MAPE.
    
    Returns correlation between feature weight and MAPE reduction.
    """
    if not results:
        return {}
    
    # Get all features that were varied
    all_weights = [r.weights for r in results]
    mapes = [r.mape for r in results]
    
    # Find features with variance
    features = list(all_weights[0].keys())
    importance = {}
    
    for feat in features:
        values = [w[feat] for w in all_weights]
        if len(set(values)) > 1:  # Feature was varied
            # Correlation between weight and MAPE (negative = higher weight helps)
            corr = np.corrcoef(values, mapes)[0, 1]
            importance[feat] = -corr  # Flip so positive = important
    
    return importance


def main():
    """Run weight tuning."""
    print("=" * 70)
    print("AOC FEATURE WEIGHT TUNING")
    print("=" * 70)
    
    # Load Parksville data (most stable store)
    conn = sqlite3.connect('aoc_sales.db')
    sales_df = pd.read_sql_query('''
        SELECT * FROM sales 
        WHERE location = 'Parksville' 
        AND date >= '2023-06-01' AND date <= '2024-12-31'
    ''', conn)
    conn.close()
    
    print(f"Loaded {len(sales_df)} Parksville records")
    
    # Current baseline
    base_weights = {
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
        "sunday": 3.0,
        "nfl_sunday": 2.0,
        "couch": 1.5,
        "party": 2.0,
        "stress": 1.0,
        "major_event": 2.5,
    }
    
    print("\n1. BASELINE TEST")
    print("-" * 50)
    baseline = run_quick_backtest(sales_df, "Parksville", base_weights)
    print(f"Baseline MAPE: {baseline.mape:.1f}%")
    print(f"Baseline MAE:  ${baseline.mae:,.0f}")
    print(f"Runtime: {baseline.runtime:.1f}s")
    
    # Test key features
    print("\n2. TESTING VIBE FEATURE WEIGHTS")
    print("-" * 50)
    
    vibe_ranges = {
        "couch": [0.5, 1.0, 1.5, 2.0, 2.5],
        "party": [1.0, 1.5, 2.0, 2.5, 3.0],
        "stress": [0.5, 1.0, 1.5, 2.0],
        "major_event": [1.5, 2.0, 2.5, 3.0, 3.5],
    }
    
    # Test each feature individually first
    for feat, values in vibe_ranges.items():
        print(f"\nTesting {feat}:")
        best_mape = baseline.mape
        best_val = base_weights[feat]
        
        for val in values:
            test_weights = base_weights.copy()
            test_weights[feat] = val
            result = run_quick_backtest(sales_df, "Parksville", test_weights)
            
            marker = "★" if result.mape < best_mape else " "
            print(f"  {feat}={val:.1f}: MAPE={result.mape:.1f}% {marker}")
            
            if result.mape < best_mape:
                best_mape = result.mape
                best_val = val
        
        print(f"  Best: {feat}={best_val:.1f} (MAPE={best_mape:.1f}%)")
        base_weights[feat] = best_val  # Update for next iteration
    
    print("\n3. OPTIMIZED WEIGHTS")
    print("-" * 50)
    final = run_quick_backtest(sales_df, "Parksville", base_weights)
    print(f"Final MAPE: {final.mape:.1f}%")
    print(f"Final MAE:  ${final.mae:,.0f}")
    print(f"Improvement: {baseline.mape - final.mape:+.1f}%")
    
    print("\nOptimized weight values:")
    for feat in ["couch", "party", "stress", "major_event"]:
        print(f"  {feat}: {base_weights[feat]:.1f}")
    
    return base_weights


if __name__ == "__main__":
    optimized = main()
