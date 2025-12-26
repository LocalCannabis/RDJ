"""
Fast GPU/Vectorized Demand Forecasting Engine

Drop-in replacement for predictor.py with:
1. Vectorized numpy operations (no row-by-row loops)
2. PyTorch GPU acceleration for similarity search
3. FAISS fallback for CPU-only systems

Provides 10-100x speedup over the original implementation.

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
All Rights Reserved. Proprietary and Confidential.
"""

from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .predictor import SimilarityConfig

logger = logging.getLogger(__name__)

# Try to import torch for GPU
try:
    import torch
    TORCH_AVAILABLE = torch.cuda.is_available()
    if TORCH_AVAILABLE:
        GPU_NAME = torch.cuda.get_device_name(0)
        logger.info(f"PyTorch CUDA available: {GPU_NAME}")
    else:
        GPU_NAME = None
        logger.info("PyTorch available but no CUDA GPU detected")
except ImportError:
    TORCH_AVAILABLE = False
    GPU_NAME = None
    logger.warning("PyTorch not available")

# Try to import faiss as fallback
try:
    import faiss
    FAISS_AVAILABLE = True
    logger.info("FAISS available as fallback")
except ImportError:
    FAISS_AVAILABLE = False


def build_condition_vectors_vectorized(
    df: pd.DataFrame,
    cfg: SimilarityConfig,
) -> np.ndarray:
    """
    Build condition vectors for entire dataframe at once (vectorized).
    
    ~100x faster than row-by-row iteration.
    Vector is now 23 dimensions (was 20).
    """
    n_rows = len(df)
    
    # Time-of-week encoding (vectorized)
    dow = df[cfg.dow_col].values.astype(float)
    hour = df[cfg.hour_col].values.astype(float)
    
    dow_rad = 2 * np.pi * (dow % 7) / 7.0
    hour_rad = 2 * np.pi * (hour % 24) / 24.0
    
    dow_sin = np.sin(dow_rad)
    dow_cos = np.cos(dow_rad)
    hour_sin = np.sin(hour_rad)
    hour_cos = np.cos(hour_rad)
    
    # NEW: Seasonality encoding
    month = df.get(cfg.month_col, pd.Series([6]*n_rows)).fillna(6).values.astype(float)
    day_of_year = df.get(cfg.day_of_year_col, pd.Series([180]*n_rows)).fillna(180).values.astype(float)
    
    month_rad = 2 * np.pi * (month - 1) / 12.0
    month_sin = np.sin(month_rad)
    month_cos = np.cos(month_rad)
    season_progress = day_of_year / 365.0
    
    # Weather (with defaults for missing)
    temp = df[cfg.temp_col].fillna(15.0).values.astype(float)
    precip = df[cfg.precip_col].fillna(0.0).values.astype(float)
    
    # Calendar/event flags (with defaults)
    is_holiday = df.get(cfg.is_holiday_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    is_preholiday = df.get(cfg.is_preholiday_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    is_payday = df.get(cfg.is_payday_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    has_home_game = df.get(cfg.has_home_game_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    has_concert = df.get(cfg.has_concert_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    has_festival = df.get(cfg.has_festival_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    
    # Vibe indices
    at_home = df.get(cfg.at_home_index_col, pd.Series([0.0]*n_rows)).fillna(0.0).values.astype(float)
    out_and_about = df.get(cfg.out_and_about_index_col, pd.Series([0.0]*n_rows)).fillna(0.0).values.astype(float)
    
    # Sunday-specific features
    is_sunday = df.get(cfg.is_sunday_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    is_nfl_sunday = df.get(cfg.is_nfl_sunday_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    
    # Vibe signal features
    couch_index = df.get(cfg.couch_index_col, pd.Series([0.5]*n_rows)).fillna(0.5).values.astype(float)
    party_index = df.get(cfg.party_index_col, pd.Series([0.0]*n_rows)).fillna(0.0).values.astype(float)
    stress_index = df.get(cfg.stress_index_col, pd.Series([0.0]*n_rows)).fillna(0.0).values.astype(float)
    has_major_event = df.get(cfg.has_major_event_col, pd.Series([0]*n_rows)).fillna(0).values.astype(float)
    
    # Stack into matrix (n_rows x 23 features)
    vectors = np.column_stack([
        dow_sin, dow_cos, hour_sin, hour_cos,
        # NEW: Seasonality (3 features)
        month_sin, month_cos, season_progress,
        # Weather
        temp, precip,
        # Calendar/events
        is_holiday, is_preholiday, is_payday,
        has_home_game, has_concert, has_festival,
        at_home, out_and_about,
        is_sunday, is_nfl_sunday,
        # Vibe features
        couch_index, party_index, stress_index, has_major_event,
    ])
    
    return vectors.astype(np.float32)


def build_single_condition_vector(
    row: Dict[str, Any],
    cfg: SimilarityConfig,
) -> np.ndarray:
    """Build condition vector for a single row (dict or Series). 23 dimensions."""
    dow = float(row.get(cfg.dow_col, 0))
    hour = float(row.get(cfg.hour_col, 12))
    
    dow_rad = 2 * np.pi * (dow % 7) / 7.0
    hour_rad = 2 * np.pi * (hour % 24) / 24.0
    
    # NEW: Seasonality
    month = float(row.get(cfg.month_col, 6))
    day_of_year = float(row.get(cfg.day_of_year_col, 180))
    month_rad = 2 * np.pi * (month - 1) / 12.0
    
    return np.array([
        np.sin(dow_rad), np.cos(dow_rad),
        np.sin(hour_rad), np.cos(hour_rad),
        # NEW: Seasonality
        np.sin(month_rad), np.cos(month_rad), day_of_year / 365.0,
        # Weather
        float(row.get(cfg.temp_col, 15.0) or 15.0),
        float(row.get(cfg.precip_col, 0.0) or 0.0),
        # Calendar/events
        float(row.get(cfg.is_holiday_col, 0) or 0),
        float(row.get(cfg.is_preholiday_col, 0) or 0),
        float(row.get(cfg.is_payday_col, 0) or 0),
        float(row.get(cfg.has_home_game_col, 0) or 0),
        float(row.get(cfg.has_concert_col, 0) or 0),
        float(row.get(cfg.has_festival_col, 0) or 0),
        float(row.get(cfg.at_home_index_col, 0.0) or 0.0),
        float(row.get(cfg.out_and_about_index_col, 0.0) or 0.0),
        float(row.get(cfg.is_sunday_col, 0) or 0),
        float(row.get(cfg.is_nfl_sunday_col, 0) or 0),
        # Vibe features
        float(row.get(cfg.couch_index_col, 0.5) or 0.5),
        float(row.get(cfg.party_index_col, 0.0) or 0.0),
        float(row.get(cfg.stress_index_col, 0.0) or 0.0),
        float(row.get(cfg.has_major_event_col, 0) or 0),
    ], dtype=np.float32)


def apply_feature_weights(vectors: np.ndarray, cfg: SimilarityConfig) -> np.ndarray:
    """Apply feature weights to vectors for weighted similarity search. 23 features."""
    weights = cfg.weights
    
    # Weight vector matching feature order (23 features)
    weight_vec = np.array([
        weights.get("dow", 1.0) / 2,    # dow_sin (split dow weight)
        weights.get("dow", 1.0) / 2,    # dow_cos
        weights.get("hour", 1.0) / 2,   # hour_sin (split hour weight)
        weights.get("hour", 1.0) / 2,   # hour_cos
        # NEW: Seasonality (3 features)
        weights.get("month", 5.0) / 2,  # month_sin (critical!)
        weights.get("month", 5.0) / 2,  # month_cos
        weights.get("season", 3.0),     # season_progress
        # Weather
        weights.get("temp", 1.0),       # temp
        weights.get("precip", 0.5),     # precip
        # Calendar/events
        weights.get("holiday", 3.0),    # is_holiday
        weights.get("preholiday", 2.5), # is_preholiday
        weights.get("payday", 2.0),     # is_payday
        weights.get("home_game", 3.0),  # has_home_game
        weights.get("concert", 2.0),    # has_concert
        weights.get("festival", 2.0),   # has_festival
        weights.get("at_home", 1.0),    # at_home
        weights.get("out_and_about", 1.0),  # out_and_about
        weights.get("sunday", 3.0),     # is_sunday (high weight!)
        weights.get("nfl_sunday", 2.0), # is_nfl_sunday
        # Vibe signal weights
        weights.get("couch", 0.0),      # couch_index (disabled)
        weights.get("party", 0.0),      # party_index (disabled)
        weights.get("stress", 0.0),     # stress_index (disabled)
        weights.get("major_event", 0.0),  # has_major_event (disabled)
    ], dtype=np.float32)
    
    # Apply sqrt of weights (since L2 distance squares them)
    return vectors * np.sqrt(weight_vec)


class FastSimilarityIndex:
    """
    Fast similarity search using PyTorch GPU or FAISS/numpy fallback.
    
    Uses GPU for massive parallelization of distance calculations.
    RTX 3060 can compute millions of distances per second.
    """
    
    def __init__(
        self,
        conditions_df: pd.DataFrame,
        cfg: Optional[SimilarityConfig] = None,
        use_gpu: bool = True,
    ):
        self.cfg = cfg or SimilarityConfig()
        self.conditions_df = conditions_df
        self.use_gpu = use_gpu and TORCH_AVAILABLE
        self.device = torch.device("cuda" if self.use_gpu else "cpu") if TORCH_AVAILABLE else None
        
        # Build vectors (numpy)
        logger.debug("Building condition vectors...")
        self.vectors = build_condition_vectors_vectorized(conditions_df, self.cfg)
        
        # Apply feature weights
        self.weighted_vectors = apply_feature_weights(self.vectors, self.cfg)
        
        # Move to GPU if available
        if self.use_gpu and TORCH_AVAILABLE:
            self.vectors_gpu = torch.from_numpy(self.weighted_vectors).to(self.device)
            logger.info(f"Loaded {len(self.vectors_gpu)} vectors to GPU ({GPU_NAME})")
        else:
            self.vectors_gpu = None
            
        # Build FAISS index as fallback
        if not self.use_gpu and FAISS_AVAILABLE:
            self._build_faiss_index()
        else:
            self.index = None
    
    def _build_faiss_index(self):
        """Build FAISS index for fast nearest neighbor search (CPU fallback)."""
        d = self.weighted_vectors.shape[1]
        self.index = faiss.IndexFlatL2(d)
        self.index.add(self.weighted_vectors)
        logger.debug(f"Built FAISS index with {self.index.ntotal} vectors")
    
    def find_similar(
        self,
        query_row: Dict[str, Any],
        k: int = 100,
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Find k most similar historical conditions.
        
        Returns:
            indices: Array of indices into conditions_df
            distances: Array of L2 distances (lower = more similar)
        """
        # Build query vector
        query_vec = build_single_condition_vector(query_row, self.cfg)
        query_weighted = apply_feature_weights(query_vec.reshape(1, -1), self.cfg)
        
        k = min(k, len(self.conditions_df))
        
        if self.use_gpu and self.vectors_gpu is not None:
            # PyTorch GPU - blazingly fast!
            query_gpu = torch.from_numpy(query_weighted).to(self.device)
            
            # Compute all distances in parallel on GPU
            # L2 distance: ||a - b||^2 = ||a||^2 + ||b||^2 - 2*a·b
            # For efficiency, we compute squared distances
            distances_gpu = torch.sum((self.vectors_gpu - query_gpu) ** 2, dim=1)
            
            # Get top-k (torch.topk returns smallest when largest=False)
            top_distances, top_indices = torch.topk(distances_gpu, k, largest=False)
            
            return top_indices.cpu().numpy(), top_distances.cpu().numpy()
            
        elif FAISS_AVAILABLE and self.index is not None:
            # FAISS CPU fallback
            distances, indices = self.index.search(query_weighted, k)
            return indices[0], distances[0]
        else:
            # Pure numpy fallback
            diffs = self.weighted_vectors - query_weighted
            distances = np.sum(diffs ** 2, axis=1)
            indices = np.argpartition(distances, k)[:k]
            return indices, distances[indices]
    
    def find_similar_batch(
        self,
        query_rows: List[Dict[str, Any]],
        k: int = 100,
    ) -> List[tuple[np.ndarray, np.ndarray]]:
        """
        Find k most similar for multiple queries at once (GPU batch processing).
        
        Much faster than calling find_similar() in a loop.
        """
        if not self.use_gpu or self.vectors_gpu is None:
            # Fallback to sequential
            return [self.find_similar(row, k) for row in query_rows]
        
        # Build all query vectors
        query_vecs = np.vstack([
            apply_feature_weights(
                build_single_condition_vector(row, self.cfg).reshape(1, -1),
                self.cfg
            )
            for row in query_rows
        ])
        
        k = min(k, len(self.conditions_df))
        
        # Move to GPU
        queries_gpu = torch.from_numpy(query_vecs).to(self.device)  # (n_queries, d)
        
        # Compute all pairwise distances: (n_queries, n_historical)
        # Using broadcasting: queries_gpu[:, None, :] - vectors_gpu[None, :, :]
        # This would be (n_queries, n_historical, d), then sum over d
        # More memory efficient: compute distances row by row but still on GPU
        
        results = []
        for i in range(len(query_rows)):
            distances_gpu = torch.sum((self.vectors_gpu - queries_gpu[i:i+1]) ** 2, dim=1)
            top_distances, top_indices = torch.topk(distances_gpu, k, largest=False)
            results.append((top_indices.cpu().numpy(), top_distances.cpu().numpy()))
        
        return results


def forecast_demand_fast(
    conditions_df: pd.DataFrame,
    future_row: pd.Series | Dict[str, Any],
    *,
    k_neighbors: int = 100,
    outcome_cols: Optional[Sequence[str]] = None,
    cfg: Optional[SimilarityConfig] = None,
    index: Optional[FastSimilarityIndex] = None,
) -> Dict[str, float]:
    """
    Fast demand forecast using vectorized operations and FAISS.
    
    Drop-in replacement for `forecast_demand_for_slot` with 10-50x speedup.
    
    Parameters
    ----------
    conditions_df:
        Historical conditions + outcomes table.
    future_row:
        Dict-like with dow, hour, temp, precip, calendar/vibe flags.
    k_neighbors:
        Number of similar past rows to aggregate.
    outcome_cols:
        Columns to aggregate as demand signals.
    cfg:
        Similarity configuration.
    index:
        Pre-built FastSimilarityIndex (optional, for repeated queries).
    
    Returns
    -------
    dict
        Forecast values and diagnostics.
    """
    if cfg is None:
        cfg = SimilarityConfig()
    
    if conditions_df.empty:
        raise ValueError("conditions_df is empty")
    
    if outcome_cols is None:
        outcome_cols = []
        for col in ("sales_units", "sales_revenue"):
            if col in conditions_df.columns:
                outcome_cols.append(col)
        if not outcome_cols:
            raise ValueError("No outcome columns found")
    
    # Build or use provided index
    if index is None:
        index = FastSimilarityIndex(conditions_df, cfg)
    
    # Convert to dict if needed
    if isinstance(future_row, pd.Series):
        future_row = future_row.to_dict()
    
    # Find similar days
    indices, distances = index.find_similar(future_row, k_neighbors)
    
    # Get neighbor data
    neighbor_df = conditions_df.iloc[indices].copy()
    
    # Convert distances to similarity scores (negative distance)
    # Lower distance = higher similarity
    similarity_scores = -distances
    
    # Aggregate outcomes (weighted by similarity)
    # Use softmax-style weighting
    max_score = similarity_scores.max()
    weights = np.exp(similarity_scores - max_score)  # Numerical stability
    weights = weights / weights.sum()
    
    result: Dict[str, float] = {}
    
    for col in outcome_cols:
        if col in neighbor_df.columns:
            values = neighbor_df[col].values
            # Weighted average
            result[f"expected_{col}"] = float(np.average(values, weights=weights))
            # Also compute simple mean for comparison
            result[f"mean_{col}"] = float(values.mean())
            result[f"std_{col}"] = float(values.std())
    
    # Diagnostics
    result["neighbors_used"] = len(indices)
    result["similarity_min"] = float(similarity_scores.min())
    result["similarity_max"] = float(similarity_scores.max())
    result["similarity_mean"] = float(similarity_scores.mean())
    
    return result


# Monkey-patch for easy swap
def enable_fast_forecasting():
    """
    Monkey-patch the predictor module to use fast implementation.
    
    Usage:
        from aoc_analytics.core.predictor_fast import enable_fast_forecasting
        enable_fast_forecasting()
    """
    from . import predictor
    predictor.forecast_demand_for_slot = forecast_demand_fast
    logger.info("Fast forecasting enabled (predictor.forecast_demand_for_slot patched)")
