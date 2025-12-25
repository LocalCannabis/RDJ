"""
AOC Adaptive Weights Engine

Continuously learns category preferences from sales + weather data.
Recalculates correlation weights on a schedule and stores them for
the decision router to use.

This is the "learning" part of AOC - it keeps the system fresh.

Architecture:
    1. AdaptiveWeightsEngine - calculates weights from data
    2. WeightsStore - persists weights with versioning
    3. WeightsScheduler - runs recalculation on schedule
    4. get_current_weights() - retrieves latest weights for decision router
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field, asdict
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import os

logger = logging.getLogger(__name__)


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class RegimeWeights:
    """Computed weights for a single regime."""
    regime_name: str
    category_boosts: Dict[str, float] = field(default_factory=dict)
    category_demotes: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    confidence: float = 0.0
    computed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "regime_name": self.regime_name,
            "category_boosts": self.category_boosts,
            "category_demotes": self.category_demotes,
            "sample_size": self.sample_size,
            "confidence": self.confidence,
            "computed_at": self.computed_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "RegimeWeights":
        return cls(
            regime_name=data["regime_name"],
            category_boosts=data.get("category_boosts", {}),
            category_demotes=data.get("category_demotes", {}),
            sample_size=data.get("sample_size", 0),
            confidence=data.get("confidence", 0.0),
            computed_at=datetime.fromisoformat(data["computed_at"]) if "computed_at" in data else datetime.now(),
        )


@dataclass 
class StoreWeights:
    """Computed weights for a single store."""
    store_id: str
    weather_sensitivity: Dict[str, Dict[str, float]] = field(default_factory=dict)
    time_patterns: Dict[str, Dict[str, float]] = field(default_factory=dict)
    weekend_boosts: Dict[str, float] = field(default_factory=dict)
    sample_size: int = 0
    computed_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "store_id": self.store_id,
            "weather_sensitivity": self.weather_sensitivity,
            "time_patterns": self.time_patterns,
            "weekend_boosts": self.weekend_boosts,
            "sample_size": self.sample_size,
            "computed_at": self.computed_at.isoformat(),
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "StoreWeights":
        return cls(
            store_id=data["store_id"],
            weather_sensitivity=data.get("weather_sensitivity", {}),
            time_patterns=data.get("time_patterns", {}),
            weekend_boosts=data.get("weekend_boosts", {}),
            sample_size=data.get("sample_size", 0),
            computed_at=datetime.fromisoformat(data["computed_at"]) if "computed_at" in data else datetime.now(),
        )


@dataclass
class WeightsSnapshot:
    """Complete snapshot of all computed weights."""
    version: int
    computed_at: datetime
    data_range_start: date
    data_range_end: date
    total_transactions: int
    regime_weights: Dict[str, RegimeWeights] = field(default_factory=dict)
    store_weights: Dict[str, StoreWeights] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "version": self.version,
            "computed_at": self.computed_at.isoformat(),
            "data_range_start": self.data_range_start.isoformat(),
            "data_range_end": self.data_range_end.isoformat(),
            "total_transactions": self.total_transactions,
            "regime_weights": {k: v.to_dict() for k, v in self.regime_weights.items()},
            "store_weights": {k: v.to_dict() for k, v in self.store_weights.items()},
        }


# =============================================================================
# WEIGHTS STORE - Database persistence
# =============================================================================

class WeightsStore:
    """
    Persists computed weights with versioning.
    
    Uses SQLite for local dev, can be swapped to PostgreSQL for production.
    """
    
    def __init__(self, db_path: Optional[str] = None):
        if db_path is None:
            # Default to aoc_analytics directory
            db_path = os.environ.get(
                "AOC_WEIGHTS_DB",
                str(Path(__file__).parent.parent.parent.parent / "aoc_weights.db")
            )
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        """Create weights storage tables."""
        conn = sqlite3.connect(self.db_path)
        conn.executescript("""
            CREATE TABLE IF NOT EXISTS weights_snapshots (
                version INTEGER PRIMARY KEY,
                computed_at TEXT NOT NULL,
                data_range_start TEXT NOT NULL,
                data_range_end TEXT NOT NULL,
                total_transactions INTEGER,
                weights_json TEXT NOT NULL,
                is_active INTEGER DEFAULT 0
            );
            
            CREATE INDEX IF NOT EXISTS idx_snapshots_active 
            ON weights_snapshots(is_active);
            
            CREATE TABLE IF NOT EXISTS weights_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                regime_or_store TEXT NOT NULL,
                category TEXT NOT NULL,
                weight_type TEXT NOT NULL,
                weight_value REAL NOT NULL,
                version INTEGER NOT NULL,
                computed_at TEXT NOT NULL
            );
            
            CREATE INDEX IF NOT EXISTS idx_history_regime 
            ON weights_history(regime_or_store, category);
        """)
        conn.commit()
        conn.close()
    
    def save_snapshot(self, snapshot: WeightsSnapshot, activate: bool = True) -> int:
        """Save a weights snapshot, optionally making it active."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Get next version
        cursor.execute("SELECT COALESCE(MAX(version), 0) + 1 FROM weights_snapshots")
        version = cursor.fetchone()[0]
        snapshot.version = version
        
        # Save snapshot
        cursor.execute("""
            INSERT INTO weights_snapshots 
            (version, computed_at, data_range_start, data_range_end, total_transactions, weights_json, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            version,
            snapshot.computed_at.isoformat(),
            snapshot.data_range_start.isoformat(),
            snapshot.data_range_end.isoformat(),
            snapshot.total_transactions,
            json.dumps(snapshot.to_dict()),
            1 if activate else 0,
        ))
        
        # If activating, deactivate others
        if activate:
            cursor.execute(
                "UPDATE weights_snapshots SET is_active = 0 WHERE version != ?",
                (version,)
            )
        
        # Save to history for trend tracking
        for regime_name, regime_weights in snapshot.regime_weights.items():
            for cat, weight in regime_weights.category_boosts.items():
                cursor.execute("""
                    INSERT INTO weights_history 
                    (regime_or_store, category, weight_type, weight_value, version, computed_at)
                    VALUES (?, ?, 'boost', ?, ?, ?)
                """, (regime_name, cat, weight, version, snapshot.computed_at.isoformat()))
            for cat, weight in regime_weights.category_demotes.items():
                cursor.execute("""
                    INSERT INTO weights_history 
                    (regime_or_store, category, weight_type, weight_value, version, computed_at)
                    VALUES (?, ?, 'demote', ?, ?, ?)
                """, (regime_name, cat, weight, version, snapshot.computed_at.isoformat()))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved weights snapshot v{version} (active={activate})")
        return version
    
    def get_active_snapshot(self) -> Optional[WeightsSnapshot]:
        """Get the currently active weights snapshot."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT weights_json FROM weights_snapshots 
            WHERE is_active = 1 
            ORDER BY version DESC LIMIT 1
        """)
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        data = json.loads(row[0])
        return self._parse_snapshot(data)
    
    def get_snapshot_by_version(self, version: int) -> Optional[WeightsSnapshot]:
        """Get a specific version of weights."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute(
            "SELECT weights_json FROM weights_snapshots WHERE version = ?",
            (version,)
        )
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        data = json.loads(row[0])
        return self._parse_snapshot(data)
    
    def _parse_snapshot(self, data: Dict) -> WeightsSnapshot:
        """Parse a snapshot from JSON data."""
        return WeightsSnapshot(
            version=data["version"],
            computed_at=datetime.fromisoformat(data["computed_at"]),
            data_range_start=date.fromisoformat(data["data_range_start"]),
            data_range_end=date.fromisoformat(data["data_range_end"]),
            total_transactions=data["total_transactions"],
            regime_weights={
                k: RegimeWeights.from_dict(v) 
                for k, v in data.get("regime_weights", {}).items()
            },
            store_weights={
                k: StoreWeights.from_dict(v) 
                for k, v in data.get("store_weights", {}).items()
            },
        )
    
    def list_versions(self, limit: int = 10) -> List[Dict[str, Any]]:
        """List recent weight versions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT version, computed_at, data_range_start, data_range_end, 
                   total_transactions, is_active
            FROM weights_snapshots
            ORDER BY version DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "version": row[0],
                "computed_at": row[1],
                "data_range_start": row[2],
                "data_range_end": row[3],
                "total_transactions": row[4],
                "is_active": bool(row[5]),
            }
            for row in rows
        ]
    
    def get_weight_trend(
        self, 
        regime_or_store: str, 
        category: str, 
        limit: int = 12
    ) -> List[Dict[str, Any]]:
        """Get historical trend of a specific weight."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT weight_type, weight_value, version, computed_at
            FROM weights_history
            WHERE regime_or_store = ? AND category = ?
            ORDER BY version DESC
            LIMIT ?
        """, (regime_or_store, category, limit))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                "weight_type": row[0],
                "weight_value": row[1],
                "version": row[2],
                "computed_at": row[3],
            }
            for row in rows
        ]


# =============================================================================
# ADAPTIVE WEIGHTS ENGINE - Computes weights from data
# =============================================================================

class AdaptiveWeightsEngine:
    """
    Computes category weights from sales + weather data.
    
    Uses the same methodology as our initial analysis but automated:
    1. Load sales data with weather context
    2. Calculate category share under different conditions
    3. Compute lift vs baseline
    4. Return as structured weights
    """
    
    # Minimum sample size for statistical significance
    MIN_SAMPLE_SIZE = 1000
    
    # Minimum lift (percentage points) to count as significant
    MIN_LIFT_PP = 0.2
    
    # Category mapping (from raw categories to simple categories)
    CATEGORY_MAP = {
        'Pre-Rolls': 'pre-rolls',
        'Pre-roll Indica': 'indica',
        'Pre-roll Sativa': 'sativa', 
        'Pre-roll Hybrid': 'hybrid',
        'Dried Flower': 'flower',
        'Flower Indica': 'indica',
        'Flower Sativa': 'sativa',
        'Beverages': 'beverages',
        'Distillate Gummies/Chews': 'edibles',
        'Rosin & Resin Gummies/Chews': 'edibles',
        'Extracts - Inhaled': 'extracts',
    }
    
    SIMPLE_CATEGORIES = ['flower', 'edibles', 'pre-rolls', 'indica', 'sativa', 'beverages', 'extracts']
    
    def __init__(self, db_path: Optional[str] = None):
        """Initialize with path to sales database."""
        if db_path is None:
            db_path = os.environ.get(
                "AOC_SALES_DB",
                str(Path(__file__).parent.parent.parent.parent / "aoc_sales.db")
            )
        self.db_path = db_path
    
    def compute_weights(
        self, 
        lookback_days: int = 365,
        min_date: Optional[date] = None,
        max_date: Optional[date] = None,
    ) -> WeightsSnapshot:
        """
        Compute all weights from recent data.
        
        Args:
            lookback_days: How many days of data to use (default 1 year)
            min_date: Override start date
            max_date: Override end date (default today)
        
        Returns:
            WeightsSnapshot with all computed weights
        """
        import pandas as pd
        
        # Calculate date range
        if max_date is None:
            max_date = date.today()
        if min_date is None:
            min_date = max_date - timedelta(days=lookback_days)
        
        logger.info(f"Computing weights for {min_date} to {max_date}")
        
        # Load data
        conn = sqlite3.connect(self.db_path)
        
        query = """
        SELECT s.date, s.time, s.location, s.category, s.quantity,
               wh.temp_c, wh.precip_mm
        FROM sales s
        LEFT JOIN weather_hourly wh ON s.location = wh.location AND s.date = wh.date 
            AND CAST(strftime('%H', s.time) AS INTEGER) = wh.hour
        WHERE s.date >= ? AND s.date <= ?
          AND wh.temp_c IS NOT NULL
        """
        
        df = pd.read_sql_query(query, conn, params=(min_date.isoformat(), max_date.isoformat()))
        conn.close()
        
        if len(df) < self.MIN_SAMPLE_SIZE:
            logger.warning(f"Insufficient data: {len(df)} transactions (need {self.MIN_SAMPLE_SIZE})")
            raise ValueError(f"Insufficient data for weight calculation: {len(df)} transactions")
        
        logger.info(f"Loaded {len(df):,} transactions")
        
        # Preprocess
        df['primary_cat'] = df['category'].str.split(' > ').str[0].str.strip().fillna('Unknown')
        df['simple_cat'] = df['primary_cat'].map(self.CATEGORY_MAP).fillna('other')
        df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
        df['hour'] = df['datetime'].dt.hour
        df['dow'] = df['datetime'].dt.dayofweek
        df['is_weekend'] = df['dow'] >= 5
        df['is_cold'] = df['temp_c'] < 10
        df['is_warm'] = df['temp_c'] > 18
        df['is_rainy'] = df['precip_mm'] > 0.5
        
        def get_period(h):
            if 6 <= h < 12: return 'morning'
            elif 12 <= h < 17: return 'afternoon'
            else: return 'evening'
        df['period'] = df['hour'].apply(get_period)
        
        # Compute regime weights
        regime_weights = self._compute_regime_weights(df)
        
        # Compute store-specific weights
        store_weights = self._compute_store_weights(df)
        
        # Create snapshot
        snapshot = WeightsSnapshot(
            version=0,  # Will be set by store
            computed_at=datetime.now(),
            data_range_start=min_date,
            data_range_end=max_date,
            total_transactions=len(df),
            regime_weights=regime_weights,
            store_weights=store_weights,
        )
        
        logger.info(f"Computed weights: {len(regime_weights)} regimes, {len(store_weights)} stores")
        return snapshot
    
    def _calc_lift(self, subset_df, full_df, cat: str) -> float:
        """Calculate percentage point lift for a category."""
        if len(subset_df) == 0 or len(full_df) == 0:
            return 0.0
        subset_share = len(subset_df[subset_df['simple_cat'] == cat]) / len(subset_df)
        full_share = len(full_df[full_df['simple_cat'] == cat]) / len(full_df)
        return (subset_share - full_share) * 100
    
    def _compute_regime_weights(self, df) -> Dict[str, RegimeWeights]:
        """Compute weights for each regime."""
        regimes = {}
        
        # COZY_INDOOR (cold weather)
        cold_df = df[df['is_cold']]
        if len(cold_df) >= self.MIN_SAMPLE_SIZE:
            boosts, demotes = {}, {}
            for cat in self.SIMPLE_CATEGORIES:
                lift = self._calc_lift(cold_df, df, cat)
                if lift > self.MIN_LIFT_PP:
                    boosts[cat] = round(lift, 2)
                elif lift < -self.MIN_LIFT_PP:
                    demotes[cat] = round(abs(lift), 2)
            
            regimes['cozy_indoor'] = RegimeWeights(
                regime_name='cozy_indoor',
                category_boosts=boosts,
                category_demotes=demotes,
                sample_size=len(cold_df),
                confidence=min(1.0, len(cold_df) / 100000),
            )
        
        # SUNNY_OUTDOOR (warm + dry)
        warm_df = df[df['is_warm'] & ~df['is_rainy']]
        if len(warm_df) >= self.MIN_SAMPLE_SIZE:
            boosts, demotes = {}, {}
            for cat in self.SIMPLE_CATEGORIES:
                lift = self._calc_lift(warm_df, df, cat)
                if lift > self.MIN_LIFT_PP:
                    boosts[cat] = round(lift, 2)
                elif lift < -self.MIN_LIFT_PP:
                    demotes[cat] = round(abs(lift), 2)
            
            regimes['sunny_outdoor'] = RegimeWeights(
                regime_name='sunny_outdoor',
                category_boosts=boosts,
                category_demotes=demotes,
                sample_size=len(warm_df),
                confidence=min(1.0, len(warm_df) / 100000),
            )
        
        # RAINY_DAY
        rain_df = df[df['is_rainy']]
        if len(rain_df) >= self.MIN_SAMPLE_SIZE:
            boosts, demotes = {}, {}
            for cat in self.SIMPLE_CATEGORIES:
                lift = self._calc_lift(rain_df, df, cat)
                if lift > self.MIN_LIFT_PP:
                    boosts[cat] = round(lift, 2)
                elif lift < -self.MIN_LIFT_PP:
                    demotes[cat] = round(abs(lift), 2)
            
            regimes['rainy_day'] = RegimeWeights(
                regime_name='rainy_day',
                category_boosts=boosts,
                category_demotes=demotes,
                sample_size=len(rain_df),
                confidence=min(1.0, len(rain_df) / 50000),
            )
        
        # EVENING_WIND_DOWN (weekday evenings)
        eve_df = df[(df['period'] == 'evening') & ~df['is_weekend']]
        if len(eve_df) >= self.MIN_SAMPLE_SIZE:
            boosts, demotes = {}, {}
            for cat in self.SIMPLE_CATEGORIES:
                lift = self._calc_lift(eve_df, df, cat)
                if lift > self.MIN_LIFT_PP:
                    boosts[cat] = round(lift, 2)
                elif lift < -self.MIN_LIFT_PP:
                    demotes[cat] = round(abs(lift), 2)
            
            regimes['evening_wind_down'] = RegimeWeights(
                regime_name='evening_wind_down',
                category_boosts=boosts,
                category_demotes=demotes,
                sample_size=len(eve_df),
                confidence=min(1.0, len(eve_df) / 100000),
            )
        
        # WEEKEND_SOCIAL (weekend afternoon+evening)
        wknd_df = df[df['is_weekend'] & df['period'].isin(['afternoon', 'evening'])]
        if len(wknd_df) >= self.MIN_SAMPLE_SIZE:
            boosts, demotes = {}, {}
            for cat in self.SIMPLE_CATEGORIES:
                lift = self._calc_lift(wknd_df, df, cat)
                if lift > self.MIN_LIFT_PP:
                    boosts[cat] = round(lift, 2)
                elif lift < -self.MIN_LIFT_PP:
                    demotes[cat] = round(abs(lift), 2)
            
            regimes['weekend_social'] = RegimeWeights(
                regime_name='weekend_social',
                category_boosts=boosts,
                category_demotes=demotes,
                sample_size=len(wknd_df),
                confidence=min(1.0, len(wknd_df) / 100000),
            )
        
        # MORNING_FUNCTIONAL (weekday mornings)
        morn_df = df[(df['period'] == 'morning') & ~df['is_weekend']]
        if len(morn_df) >= self.MIN_SAMPLE_SIZE:
            boosts, demotes = {}, {}
            for cat in self.SIMPLE_CATEGORIES:
                lift = self._calc_lift(morn_df, df, cat)
                if lift > self.MIN_LIFT_PP:
                    boosts[cat] = round(lift, 2)
                elif lift < -self.MIN_LIFT_PP:
                    demotes[cat] = round(abs(lift), 2)
            
            regimes['morning_functional'] = RegimeWeights(
                regime_name='morning_functional',
                category_boosts=boosts,
                category_demotes=demotes,
                sample_size=len(morn_df),
                confidence=min(1.0, len(morn_df) / 50000),
            )
        
        return regimes
    
    def _compute_store_weights(self, df) -> Dict[str, StoreWeights]:
        """Compute weights for each store."""
        stores = {}
        
        for location in df['location'].unique():
            store_df = df[df['location'] == location]
            
            if len(store_df) < self.MIN_SAMPLE_SIZE:
                continue
            
            store_id = location.lower().replace(" ", "_")
            
            # Weather sensitivity
            weather_sens = {}
            
            # Cold boosts/demotes
            cold_df = store_df[store_df['is_cold']]
            if len(cold_df) >= 500:
                cold_boosts, cold_demotes = {}, {}
                for cat in self.SIMPLE_CATEGORIES:
                    lift = self._calc_lift(cold_df, store_df, cat)
                    if lift > self.MIN_LIFT_PP:
                        cold_boosts[cat] = round(lift, 2)
                    elif lift < -self.MIN_LIFT_PP:
                        cold_demotes[cat] = round(abs(lift), 2)
                weather_sens['cold_boosts'] = cold_boosts
                weather_sens['cold_demotes'] = cold_demotes
            
            # Warm boosts/demotes
            warm_df = store_df[store_df['is_warm']]
            if len(warm_df) >= 500:
                warm_boosts, warm_demotes = {}, {}
                for cat in self.SIMPLE_CATEGORIES:
                    lift = self._calc_lift(warm_df, store_df, cat)
                    if lift > self.MIN_LIFT_PP:
                        warm_boosts[cat] = round(lift, 2)
                    elif lift < -self.MIN_LIFT_PP:
                        warm_demotes[cat] = round(abs(lift), 2)
                weather_sens['warm_boosts'] = warm_boosts
                weather_sens['warm_demotes'] = warm_demotes
            
            # Time patterns
            time_patterns = {}
            
            # Evening patterns
            eve_df = store_df[store_df['period'] == 'evening']
            if len(eve_df) >= 500:
                eve_boosts, eve_demotes = {}, {}
                for cat in self.SIMPLE_CATEGORIES:
                    lift = self._calc_lift(eve_df, store_df, cat)
                    if lift > self.MIN_LIFT_PP:
                        eve_boosts[cat] = round(lift, 2)
                    elif lift < -self.MIN_LIFT_PP:
                        eve_demotes[cat] = round(abs(lift), 2)
                time_patterns['evening_boosts'] = eve_boosts
                time_patterns['evening_demotes'] = eve_demotes
            
            # Weekend boosts
            wknd_df = store_df[store_df['is_weekend']]
            wknd_boosts = {}
            if len(wknd_df) >= 500:
                for cat in self.SIMPLE_CATEGORIES:
                    lift = self._calc_lift(wknd_df, store_df, cat)
                    if lift > self.MIN_LIFT_PP:
                        wknd_boosts[cat] = round(lift, 2)
            
            stores[store_id] = StoreWeights(
                store_id=store_id,
                weather_sensitivity=weather_sens,
                time_patterns=time_patterns,
                weekend_boosts=wknd_boosts,
                sample_size=len(store_df),
            )
        
        return stores


# =============================================================================
# GLOBAL WEIGHTS ACCESSOR
# =============================================================================

_weights_store: Optional[WeightsStore] = None
_cached_weights: Optional[WeightsSnapshot] = None
_cache_time: Optional[datetime] = None
CACHE_TTL_SECONDS = 300  # 5 minutes


def get_weights_store() -> WeightsStore:
    """Get or create the global weights store."""
    global _weights_store
    if _weights_store is None:
        _weights_store = WeightsStore()
    return _weights_store


def get_current_weights(force_refresh: bool = False) -> Optional[WeightsSnapshot]:
    """
    Get the currently active weights, with caching.
    
    This is the main entry point for the decision router.
    """
    global _cached_weights, _cache_time
    
    now = datetime.now()
    
    # Check cache
    if not force_refresh and _cached_weights is not None and _cache_time is not None:
        age = (now - _cache_time).total_seconds()
        if age < CACHE_TTL_SECONDS:
            return _cached_weights
    
    # Fetch from store
    store = get_weights_store()
    _cached_weights = store.get_active_snapshot()
    _cache_time = now
    
    return _cached_weights


def recalculate_and_save_weights(
    lookback_days: int = 365,
    activate: bool = True,
) -> int:
    """
    Recalculate weights from data and save to store.
    
    This is what the scheduled job calls.
    
    Returns:
        Version number of saved snapshot
    """
    engine = AdaptiveWeightsEngine()
    snapshot = engine.compute_weights(lookback_days=lookback_days)
    
    store = get_weights_store()
    version = store.save_snapshot(snapshot, activate=activate)
    
    # Clear cache
    global _cached_weights, _cache_time
    _cached_weights = None
    _cache_time = None
    
    return version


# =============================================================================
# CLI
# =============================================================================

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="AOC Adaptive Weights Engine")
    parser.add_argument("--recalculate", action="store_true", help="Recalculate weights from data")
    parser.add_argument("--lookback", type=int, default=365, help="Days of data to use")
    parser.add_argument("--list", action="store_true", help="List weight versions")
    parser.add_argument("--show", type=int, help="Show specific version")
    parser.add_argument("--trend", nargs=2, metavar=("REGIME", "CATEGORY"), help="Show weight trend")
    
    args = parser.parse_args()
    
    store = get_weights_store()
    
    if args.recalculate:
        print(f"Recalculating weights with {args.lookback} day lookback...")
        version = recalculate_and_save_weights(lookback_days=args.lookback)
        print(f"✓ Saved as version {version}")
        
        # Show summary
        snapshot = store.get_snapshot_by_version(version)
        if snapshot:
            print(f"\nData range: {snapshot.data_range_start} to {snapshot.data_range_end}")
            print(f"Transactions: {snapshot.total_transactions:,}")
            print(f"\nRegimes computed: {list(snapshot.regime_weights.keys())}")
            print(f"Stores computed: {list(snapshot.store_weights.keys())}")
            
            print("\n--- Regime Weights ---")
            for name, weights in snapshot.regime_weights.items():
                print(f"\n{name} (n={weights.sample_size:,}):")
                print(f"  Boosts: {weights.category_boosts}")
                print(f"  Demotes: {weights.category_demotes}")
    
    elif args.list:
        versions = store.list_versions()
        print("Weight Versions:")
        for v in versions:
            active = "✓ ACTIVE" if v["is_active"] else ""
            print(f"  v{v['version']}: {v['computed_at']} ({v['total_transactions']:,} txns) {active}")
    
    elif args.show:
        snapshot = store.get_snapshot_by_version(args.show)
        if snapshot:
            print(json.dumps(snapshot.to_dict(), indent=2))
        else:
            print(f"Version {args.show} not found")
    
    elif args.trend:
        regime, category = args.trend
        trend = store.get_weight_trend(regime, category)
        if trend:
            print(f"Weight trend for {regime}/{category}:")
            for t in trend:
                print(f"  v{t['version']}: {t['weight_type']} = {t['weight_value']:.2f} ({t['computed_at']})")
        else:
            print(f"No trend data for {regime}/{category}")
    
    else:
        # Show current weights
        weights = get_current_weights()
        if weights:
            print(f"Current active weights: v{weights.version}")
            print(f"Computed: {weights.computed_at}")
            print(f"Data range: {weights.data_range_start} to {weights.data_range_end}")
            print(f"Transactions: {weights.total_transactions:,}")
        else:
            print("No active weights. Run with --recalculate to compute initial weights.")
