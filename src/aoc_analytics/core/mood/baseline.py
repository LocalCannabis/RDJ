"""
Rolling Baseline and Z-Score Normalization

Computes rolling z-scores for mood signals against historical baselines.
This allows AOC to detect deviations from "normal" rather than using
absolute values.

Key features:
    - 30-day rolling window by default
    - Geo-specific baselines
    - Smoothing (3-7 day) to reduce noise
    - Handles missing data gracefully
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class BaselineStats:
    """Rolling baseline statistics for a signal."""
    signal_name: str
    geo: str
    window_start: str
    window_end: str
    sample_count: int
    mean: float
    std: float
    min_val: float
    max_val: float
    p25: float
    p75: float
    
    def z_score(self, value: float) -> float:
        """Compute z-score for a value against this baseline."""
        if self.std < 0.001:
            return 0.0
        z = (value - self.mean) / self.std
        # Clamp to prevent extreme outliers
        return max(-3.0, min(3.0, z))


@dataclass
class NormalizedMood:
    """Mood signals normalized against rolling baseline."""
    date: str
    geo: str
    
    # Raw values (0-1 scale)
    raw_anxiety: float
    raw_cozy: float
    raw_party: float
    raw_vibe: float
    
    # Z-scored values (relative to 30-day rolling baseline)
    anxiety_z: float
    cozy_z: float
    party_z: float
    vibe_z: float
    
    # Smoothed z-scores (3-day moving average)
    anxiety_z_smooth: Optional[float] = None
    cozy_z_smooth: Optional[float] = None
    party_z_smooth: Optional[float] = None
    vibe_z_smooth: Optional[float] = None
    
    # Baseline stats used
    baseline_days: int = 30
    baseline_coverage: float = 0.0  # What % of baseline window had data
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "date": self.date,
            "geo": self.geo,
            "raw": {
                "anxiety": self.raw_anxiety,
                "cozy": self.raw_cozy,
                "party": self.raw_party,
                "vibe": self.raw_vibe,
            },
            "z_scored": {
                "anxiety": self.anxiety_z,
                "cozy": self.cozy_z,
                "party": self.party_z,
                "vibe": self.vibe_z,
            },
            "z_smoothed": {
                "anxiety": self.anxiety_z_smooth,
                "cozy": self.cozy_z_smooth,
                "party": self.party_z_smooth,
                "vibe": self.vibe_z_smooth,
            },
            "baseline": {
                "days": self.baseline_days,
                "coverage": self.baseline_coverage,
            },
        }


class RollingBaseline:
    """
    Computes and maintains rolling baselines for mood signals.
    
    Uses historical data to establish "normal" ranges, then
    z-scores new observations against these baselines.
    """
    
    SIGNALS = ["anxiety", "cozy", "party", "vibe"]
    
    def __init__(
        self,
        db_path: str | Path,
        window_days: int = 30,
        smoothing_days: int = 3,
        min_samples: int = 7,
    ):
        """
        Initialize baseline calculator.
        
        Args:
            db_path: Path to SQLite database with mood_features_daily
            window_days: Rolling window size (default 30)
            smoothing_days: Smoothing window for z-scores (default 3)
            min_samples: Minimum samples needed for valid baseline
        """
        self.db_path = Path(db_path)
        self.window_days = window_days
        self.smoothing_days = smoothing_days
        self.min_samples = min_samples
    
    def _get_historical_values(
        self,
        signal: str,
        geo: str,
        end_date: date,
        days: int,
    ) -> List[Tuple[str, float]]:
        """
        Fetch historical values for a signal.
        
        Args:
            signal: Signal name (anxiety, cozy, party, vibe)
            geo: Geographic region
            end_date: End of window (exclusive)
            days: Number of days to look back
            
        Returns:
            List of (date, value) tuples
        """
        start_date = end_date - timedelta(days=days)
        
        # Map signal name to column
        column_map = {
            "anxiety": "local_vibe_anxiety",
            "cozy": "local_vibe_cozy",
            "party": "local_vibe_party",
            "vibe": "local_vibe_score",
        }
        
        column = column_map.get(signal)
        if not column:
            logger.warning(f"Unknown signal: {signal}")
            return []
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute(f"""
                SELECT date, {column}
                FROM mood_features_daily
                WHERE geo = ?
                  AND date >= ?
                  AND date < ?
                  AND {column} IS NOT NULL
                ORDER BY date
            """, (geo, start_date.isoformat(), end_date.isoformat()))
            
            return cursor.fetchall()
        except sqlite3.OperationalError as e:
            logger.warning(f"Database error fetching baseline: {e}")
            return []
        finally:
            conn.close()
    
    def compute_baseline(
        self,
        signal: str,
        geo: str,
        as_of_date: date,
    ) -> BaselineStats:
        """
        Compute baseline statistics for a signal.
        
        Args:
            signal: Signal name
            geo: Geographic region
            as_of_date: Compute baseline as of this date (exclusive)
            
        Returns:
            BaselineStats with mean, std, percentiles
        """
        values = self._get_historical_values(
            signal, geo, as_of_date, self.window_days
        )
        
        if len(values) < self.min_samples:
            # Not enough data - return neutral baseline
            logger.debug(
                f"Insufficient baseline data for {signal}/{geo}: "
                f"{len(values)} < {self.min_samples}"
            )
            return BaselineStats(
                signal_name=signal,
                geo=geo,
                window_start=(as_of_date - timedelta(days=self.window_days)).isoformat(),
                window_end=as_of_date.isoformat(),
                sample_count=len(values),
                mean=0.5,  # Neutral default
                std=0.15,  # Reasonable default variance
                min_val=0.0,
                max_val=1.0,
                p25=0.35,
                p75=0.65,
            )
        
        # Extract just values
        vals = [v[1] for v in values]
        
        # Compute statistics
        n = len(vals)
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        std = max(variance ** 0.5, 0.01)  # Minimum std to prevent div by zero
        
        sorted_vals = sorted(vals)
        min_val = sorted_vals[0]
        max_val = sorted_vals[-1]
        p25 = sorted_vals[int(n * 0.25)]
        p75 = sorted_vals[int(n * 0.75)]
        
        return BaselineStats(
            signal_name=signal,
            geo=geo,
            window_start=values[0][0],
            window_end=values[-1][0],
            sample_count=n,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            p25=p25,
            p75=p75,
        )
    
    def normalize_mood(
        self,
        raw_anxiety: float,
        raw_cozy: float,
        raw_party: float,
        raw_vibe: float,
        geo: str,
        target_date: Optional[date] = None,
    ) -> NormalizedMood:
        """
        Normalize raw mood values against rolling baseline.
        
        Args:
            raw_*: Raw mood values (0-1 scale)
            geo: Geographic region
            target_date: Date for normalization (default today)
            
        Returns:
            NormalizedMood with z-scored values
        """
        target_date = target_date or date.today()
        
        # Compute baselines
        baselines = {}
        for signal in self.SIGNALS:
            baselines[signal] = self.compute_baseline(signal, geo, target_date)
        
        # Compute z-scores
        anxiety_z = baselines["anxiety"].z_score(raw_anxiety)
        cozy_z = baselines["cozy"].z_score(raw_cozy)
        party_z = baselines["party"].z_score(raw_party)
        vibe_z = baselines["vibe"].z_score(raw_vibe)
        
        # Compute smoothed z-scores (need historical z-scores)
        smooth_anxiety = self._get_smoothed_z(
            "anxiety", geo, target_date, anxiety_z
        )
        smooth_cozy = self._get_smoothed_z(
            "cozy", geo, target_date, cozy_z
        )
        smooth_party = self._get_smoothed_z(
            "party", geo, target_date, party_z
        )
        smooth_vibe = self._get_smoothed_z(
            "vibe", geo, target_date, vibe_z
        )
        
        # Calculate baseline coverage
        total_expected = self.window_days
        total_samples = sum(b.sample_count for b in baselines.values()) / len(self.SIGNALS)
        coverage = total_samples / total_expected
        
        return NormalizedMood(
            date=target_date.isoformat(),
            geo=geo,
            raw_anxiety=raw_anxiety,
            raw_cozy=raw_cozy,
            raw_party=raw_party,
            raw_vibe=raw_vibe,
            anxiety_z=anxiety_z,
            cozy_z=cozy_z,
            party_z=party_z,
            vibe_z=vibe_z,
            anxiety_z_smooth=smooth_anxiety,
            cozy_z_smooth=smooth_cozy,
            party_z_smooth=smooth_party,
            vibe_z_smooth=smooth_vibe,
            baseline_days=self.window_days,
            baseline_coverage=coverage,
        )
    
    def _get_smoothed_z(
        self,
        signal: str,
        geo: str,
        target_date: date,
        current_z: float,
    ) -> Optional[float]:
        """
        Get smoothed z-score using recent values.
        
        Uses simple moving average over smoothing_days.
        """
        # Get recent z-scores from database
        column_map = {
            "anxiety": "anxiety_z",
            "cozy": "cozy_z",
            "party": "party_z",
            "vibe": "local_vibe_z",
        }
        
        column = column_map.get(signal)
        if not column:
            return current_z
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            start = target_date - timedelta(days=self.smoothing_days - 1)
            cursor.execute(f"""
                SELECT {column}
                FROM mood_features_daily
                WHERE geo = ?
                  AND date >= ?
                  AND date < ?
                  AND {column} IS NOT NULL
                ORDER BY date
            """, (geo, start.isoformat(), target_date.isoformat()))
            
            recent_z = [row[0] for row in cursor.fetchall()]
            
            if not recent_z:
                return current_z
            
            # Include current value
            all_z = recent_z + [current_z]
            return sum(all_z) / len(all_z)
            
        except sqlite3.OperationalError:
            return current_z
        finally:
            conn.close()
    
    def store_normalized(self, normalized: NormalizedMood) -> None:
        """
        Update mood_features_daily with z-scored values.
        
        Args:
            normalized: NormalizedMood to store
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute("""
                UPDATE mood_features_daily
                SET local_vibe_z = ?,
                    anxiety_z = ?,
                    cozy_z = ?,
                    party_z = ?
                WHERE date = ? AND geo = ?
            """, (
                normalized.vibe_z,
                normalized.anxiety_z,
                normalized.cozy_z,
                normalized.party_z,
                normalized.date,
                normalized.geo,
            ))
            
            if cursor.rowcount == 0:
                logger.warning(
                    f"No row updated for {normalized.date}/{normalized.geo}"
                )
            else:
                conn.commit()
                logger.info(
                    f"Updated z-scores for {normalized.date}/{normalized.geo}"
                )
                
        except Exception as e:
            logger.error(f"Failed to store normalized mood: {e}")
            conn.rollback()
        finally:
            conn.close()


def interpret_z_score(z: float) -> str:
    """
    Interpret a z-score for human-readable output.
    
    Args:
        z: Z-score value
        
    Returns:
        Human-readable interpretation
    """
    if z >= 2.0:
        return "very high (unusual)"
    elif z >= 1.0:
        return "elevated"
    elif z >= 0.5:
        return "slightly above normal"
    elif z > -0.5:
        return "normal"
    elif z > -1.0:
        return "slightly below normal"
    elif z > -2.0:
        return "depressed"
    else:
        return "very low (unusual)"


def run_baseline_test(db_path: str, geo: str = "CA"):
    """
    Test baseline computation with sample data.
    """
    print(f"=== Rolling Baseline Test ({geo}) ===\n")
    
    baseline = RollingBaseline(db_path=db_path)
    
    # Test with hypothetical current values
    raw_anxiety = 0.35
    raw_cozy = 0.45
    raw_party = 0.30
    raw_vibe = 0.50
    
    print(f"Raw values:")
    print(f"  Anxiety: {raw_anxiety:.2f}")
    print(f"  Cozy: {raw_cozy:.2f}")
    print(f"  Party: {raw_party:.2f}")
    print(f"  Vibe: {raw_vibe:.2f}")
    
    normalized = baseline.normalize_mood(
        raw_anxiety=raw_anxiety,
        raw_cozy=raw_cozy,
        raw_party=raw_party,
        raw_vibe=raw_vibe,
        geo=geo,
    )
    
    print(f"\nZ-scored (vs {normalized.baseline_days}-day baseline):")
    print(f"  Anxiety: {normalized.anxiety_z:+.2f} ({interpret_z_score(normalized.anxiety_z)})")
    print(f"  Cozy: {normalized.cozy_z:+.2f} ({interpret_z_score(normalized.cozy_z)})")
    print(f"  Party: {normalized.party_z:+.2f} ({interpret_z_score(normalized.party_z)})")
    print(f"  Vibe: {normalized.vibe_z:+.2f} ({interpret_z_score(normalized.vibe_z)})")
    
    print(f"\nBaseline coverage: {normalized.baseline_coverage:.0%}")
    
    return normalized


if __name__ == "__main__":
    import sys
    logging.basicConfig(level=logging.INFO)
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "aoc_sales.db"
    geo = sys.argv[2] if len(sys.argv) > 2 else "CA"
    
    run_baseline_test(db_path, geo)
