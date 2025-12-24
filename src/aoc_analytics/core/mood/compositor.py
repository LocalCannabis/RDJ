"""
Local Vibe Compositor

Combines Spotify and Google Trends signals into interpretable
mood indices for consumption by AOC decision engines.

Indices:
    - local_vibe_score: Overall mood (-1 to 1, higher = positive)
    - local_vibe_anxiety: Stress/anxiety level (-1 to 1)
    - local_vibe_cozy: At-home/comfort level (-1 to 1)
    - local_vibe_party: Going out/social level (-1 to 1)
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


@dataclass
class MoodFeatures:
    """Complete mood feature set for a day."""
    date: str
    geo: str
    
    # Raw Spotify values
    spotify_valence_mean: float = 0.5
    spotify_energy_mean: float = 0.5
    spotify_danceability_mean: float = 0.5
    spotify_tempo_mean: float = 120.0
    spotify_track_count: int = 0
    
    # Z-scored features
    spotify_valence_z: float = 0.0
    spotify_energy_z: float = 0.0
    google_stress_z: float = 0.0
    google_cozy_z: float = 0.0
    google_party_z: float = 0.0
    google_money_z: float = 0.0
    google_cannabis_z: float = 0.0
    
    # Derived Local Vibe indices
    local_vibe_score: float = 0.0
    local_vibe_anxiety: float = 0.0
    local_vibe_cozy: float = 0.0
    local_vibe_party: float = 0.0
    
    # Quality metadata
    data_quality_score: float = 0.0
    spotify_quality: float = 0.0
    google_quality: float = 0.0
    notes: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "date": self.date,
            "geo": self.geo,
            "spotify": {
                "valence_mean": self.spotify_valence_mean,
                "energy_mean": self.spotify_energy_mean,
                "danceability_mean": self.spotify_danceability_mean,
                "tempo_mean": self.spotify_tempo_mean,
                "track_count": self.spotify_track_count,
                "valence_z": self.spotify_valence_z,
                "energy_z": self.spotify_energy_z,
            },
            "google": {
                "stress_z": self.google_stress_z,
                "cozy_z": self.google_cozy_z,
                "party_z": self.google_party_z,
                "money_z": self.google_money_z,
                "cannabis_z": self.google_cannabis_z,
            },
            "vibe": {
                "score": self.local_vibe_score,
                "anxiety": self.local_vibe_anxiety,
                "cozy": self.local_vibe_cozy,
                "party": self.local_vibe_party,
            },
            "quality": {
                "overall": self.data_quality_score,
                "spotify": self.spotify_quality,
                "google": self.google_quality,
                "notes": self.notes,
            },
        }


def clamp(value: float, min_val: float = -1.0, max_val: float = 1.0) -> float:
    """Clamp value to safe display range."""
    return max(min_val, min(max_val, value))


class LocalVibeCompositor:
    """
    Combines raw mood signals into interpretable indices.
    
    Uses configurable weights and formulas to derive:
    - Anxiety index (stress + money pressure)
    - Cozy index (at-home - going out)
    - Party index (going out + music energy)
    - Overall vibe score
    """
    
    # Default weights for vibe composition
    DEFAULT_WEIGHTS = {
        # Anxiety composition
        "anxiety_stress_weight": 0.6,
        "anxiety_money_weight": 0.4,
        
        # Cozy composition
        "cozy_base_weight": 1.0,
        "cozy_party_subtract": 0.5,
        
        # Party composition
        "party_base_weight": 1.0,
        "party_energy_add": 0.3,
        
        # Overall vibe composition
        "vibe_valence_weight": 0.5,
        "vibe_energy_weight": 0.3,
        "vibe_anxiety_subtract": 0.4,
        "vibe_party_add": 0.2,
    }
    
    def __init__(
        self,
        db_path: str | Path,
        weights: Optional[Dict[str, float]] = None,
        baseline_days: int = 30,
    ):
        """
        Initialize compositor.
        
        Args:
            db_path: Path to SQLite database
            weights: Custom weights (merged with defaults)
            baseline_days: Days to use for rolling baseline
        """
        self.db_path = Path(db_path)
        self.weights = {**self.DEFAULT_WEIGHTS, **(weights or {})}
        self.baseline_days = baseline_days
    
    def _get_rolling_stats(
        self,
        values: List[float],
    ) -> Tuple[float, float]:
        """Calculate mean and std for z-score computation."""
        if not values:
            return 0.5, 0.1  # Default safe values
        
        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values)
        std = max(variance ** 0.5, 0.01)  # Prevent division by zero
        
        return mean, std
    
    def _compute_z_score(
        self,
        value: float,
        mean: float,
        std: float,
    ) -> float:
        """Compute z-score and clamp to reasonable range."""
        z = (value - mean) / std if std > 0 else 0.0
        return clamp(z, -3.0, 3.0)  # Prevent extreme outliers
    
    def compute_vibe_indices(
        self,
        spotify_valence_z: float,
        spotify_energy_z: float,
        google_stress_z: float,
        google_cozy_z: float,
        google_party_z: float,
        google_money_z: float,
    ) -> Tuple[float, float, float, float]:
        """
        Compute derived vibe indices from z-scored inputs.
        
        Returns:
            Tuple of (vibe_score, anxiety, cozy, party)
        """
        w = self.weights
        
        # Anxiety: stress + money pressure
        anxiety = clamp(
            w["anxiety_stress_weight"] * google_stress_z +
            w["anxiety_money_weight"] * google_money_z
        )
        
        # Cozy: at-home minus going-out
        cozy = clamp(
            w["cozy_base_weight"] * google_cozy_z -
            w["cozy_party_subtract"] * google_party_z
        )
        
        # Party: going-out plus music energy
        party = clamp(
            w["party_base_weight"] * google_party_z +
            w["party_energy_add"] * spotify_energy_z
        )
        
        # Overall vibe score
        vibe_score = clamp(
            w["vibe_valence_weight"] * spotify_valence_z +
            w["vibe_energy_weight"] * spotify_energy_z -
            w["vibe_anxiety_subtract"] * anxiety +
            w["vibe_party_add"] * party
        )
        
        return vibe_score, anxiety, cozy, party
    
    def compute_data_quality(
        self,
        spotify_track_count: int,
        google_term_coverage: float,
        hours_since_spotify: float,
        hours_since_google: float,
    ) -> Tuple[float, float, float]:
        """
        Compute data quality scores.
        
        Returns:
            Tuple of (overall_quality, spotify_quality, google_quality)
        """
        # Spotify quality
        spotify_count_score = min(spotify_track_count / 40, 1.0)
        spotify_freshness = max(0, 1 - hours_since_spotify / 48)
        spotify_quality = 0.6 * spotify_count_score + 0.4 * spotify_freshness
        
        # Google quality
        google_coverage_score = google_term_coverage
        google_freshness = max(0, 1 - hours_since_google / 48)
        google_quality = 0.6 * google_coverage_score + 0.4 * google_freshness
        
        # Overall quality
        overall = 0.5 * spotify_quality + 0.5 * google_quality
        
        return overall, spotify_quality, google_quality
    
    def compute_daily_features(
        self,
        target_date: str,
        geo: str = "CA",
    ) -> MoodFeatures:
        """
        Compute complete mood features for a date.
        
        Pulls raw data from database, normalizes, and derives indices.
        
        Args:
            target_date: Date to compute features for (YYYY-MM-DD)
            geo: Geographic region
            
        Returns:
            MoodFeatures with all computed values
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        features = MoodFeatures(date=target_date, geo=geo)
        notes = []
        
        # =========================================
        # Get Spotify data
        # =========================================
        
        # Get today's Spotify aggregates from playlist snapshot + audio features
        cursor.execute("""
            SELECT 
                AVG(af.valence) as valence_mean,
                AVG(af.energy) as energy_mean,
                AVG(af.danceability) as danceability_mean,
                AVG(af.tempo) as tempo_mean,
                COUNT(DISTINCT ps.track_id) as track_count,
                MAX(ps.fetched_at) as last_fetch
            FROM mood_spotify_playlist_snapshot ps
            JOIN mood_spotify_audio_features af ON ps.track_id = af.track_id
            WHERE ps.snapshot_date = ?
        """, (target_date,))
        
        spotify_row = cursor.fetchone()
        
        if spotify_row and spotify_row["track_count"]:
            features.spotify_valence_mean = spotify_row["valence_mean"] or 0.5
            features.spotify_energy_mean = spotify_row["energy_mean"] or 0.5
            features.spotify_danceability_mean = spotify_row["danceability_mean"] or 0.5
            features.spotify_tempo_mean = spotify_row["tempo_mean"] or 120.0
            features.spotify_track_count = spotify_row["track_count"] or 0
        else:
            notes.append("No Spotify data for date")
        
        # Get rolling baseline for Spotify (past N days)
        cursor.execute("""
            SELECT 
                AVG(af.valence) as valence_mean,
                AVG(af.energy) as energy_mean
            FROM mood_spotify_playlist_snapshot ps
            JOIN mood_spotify_audio_features af ON ps.track_id = af.track_id
            WHERE ps.snapshot_date >= date(?, '-' || ? || ' days')
            AND ps.snapshot_date < ?
        """, (target_date, self.baseline_days, target_date))
        
        baseline_row = cursor.fetchone()
        
        # Compute Spotify z-scores
        if baseline_row and baseline_row["valence_mean"]:
            # Simplified z-score (using 0.15 as typical std for Spotify features)
            valence_baseline = baseline_row["valence_mean"]
            energy_baseline = baseline_row["energy_mean"]
            
            features.spotify_valence_z = clamp(
                (features.spotify_valence_mean - valence_baseline) / 0.15, -2, 2
            )
            features.spotify_energy_z = clamp(
                (features.spotify_energy_mean - energy_baseline) / 0.15, -2, 2
            )
        
        # =========================================
        # Get Google data
        # =========================================
        
        cursor.execute("""
            SELECT 
                stress_score, cozy_score, party_score,
                money_pressure_score, cannabis_interest_score,
                term_coverage, fetched_at
            FROM mood_google_buckets_daily
            WHERE date = ? AND geo = ?
        """, (target_date, geo))
        
        google_row = cursor.fetchone()
        
        if google_row:
            # Google bucket scores are already z-scored
            features.google_stress_z = clamp(google_row["stress_score"] or 0, -2, 2)
            features.google_cozy_z = clamp(google_row["cozy_score"] or 0, -2, 2)
            features.google_party_z = clamp(google_row["party_score"] or 0, -2, 2)
            features.google_money_z = clamp(google_row["money_pressure_score"] or 0, -2, 2)
            features.google_cannabis_z = clamp(google_row["cannabis_interest_score"] or 0, -2, 2)
            term_coverage = google_row["term_coverage"] or 0
        else:
            notes.append("No Google data for date")
            term_coverage = 0
        
        conn.close()
        
        # =========================================
        # Compute derived indices
        # =========================================
        
        (
            features.local_vibe_score,
            features.local_vibe_anxiety,
            features.local_vibe_cozy,
            features.local_vibe_party,
        ) = self.compute_vibe_indices(
            features.spotify_valence_z,
            features.spotify_energy_z,
            features.google_stress_z,
            features.google_cozy_z,
            features.google_party_z,
            features.google_money_z,
        )
        
        # =========================================
        # Compute quality scores
        # =========================================
        
        # Simplified freshness (assume recent if data exists)
        hours_spotify = 0 if features.spotify_track_count > 0 else 72
        hours_google = 0 if term_coverage > 0 else 72
        
        (
            features.data_quality_score,
            features.spotify_quality,
            features.google_quality,
        ) = self.compute_data_quality(
            features.spotify_track_count,
            term_coverage,
            hours_spotify,
            hours_google,
        )
        
        features.notes = "; ".join(notes) if notes else ""
        
        return features
    
    def save_features_to_db(
        self,
        features: MoodFeatures,
    ) -> None:
        """Save computed features to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO mood_features_daily (
                date, geo,
                spotify_valence_mean, spotify_energy_mean,
                spotify_danceability_mean, spotify_tempo_mean, spotify_track_count,
                spotify_valence_z, spotify_energy_z,
                google_stress_z, google_cozy_z, google_party_z,
                google_money_z, google_cannabis_z,
                local_vibe_score, local_vibe_anxiety, local_vibe_cozy, local_vibe_party,
                data_quality_score, spotify_quality, google_quality, notes
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            features.date,
            features.geo,
            features.spotify_valence_mean,
            features.spotify_energy_mean,
            features.spotify_danceability_mean,
            features.spotify_tempo_mean,
            features.spotify_track_count,
            features.spotify_valence_z,
            features.spotify_energy_z,
            features.google_stress_z,
            features.google_cozy_z,
            features.google_party_z,
            features.google_money_z,
            features.google_cannabis_z,
            features.local_vibe_score,
            features.local_vibe_anxiety,
            features.local_vibe_cozy,
            features.local_vibe_party,
            features.data_quality_score,
            features.spotify_quality,
            features.google_quality,
            features.notes,
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved mood features for {features.date} ({features.geo})")
    
    def get_features_for_date(
        self,
        target_date: str,
        geo: str = "CA",
    ) -> Optional[MoodFeatures]:
        """
        Get pre-computed features from database.
        
        Returns None if not found.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM mood_features_daily
            WHERE date = ? AND geo = ?
        """, (target_date, geo))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return MoodFeatures(
            date=row["date"],
            geo=row["geo"],
            spotify_valence_mean=row["spotify_valence_mean"] or 0.5,
            spotify_energy_mean=row["spotify_energy_mean"] or 0.5,
            spotify_danceability_mean=row["spotify_danceability_mean"] or 0.5,
            spotify_tempo_mean=row["spotify_tempo_mean"] or 120.0,
            spotify_track_count=row["spotify_track_count"] or 0,
            spotify_valence_z=row["spotify_valence_z"] or 0.0,
            spotify_energy_z=row["spotify_energy_z"] or 0.0,
            google_stress_z=row["google_stress_z"] or 0.0,
            google_cozy_z=row["google_cozy_z"] or 0.0,
            google_party_z=row["google_party_z"] or 0.0,
            google_money_z=row["google_money_z"] or 0.0,
            google_cannabis_z=row["google_cannabis_z"] or 0.0,
            local_vibe_score=row["local_vibe_score"] or 0.0,
            local_vibe_anxiety=row["local_vibe_anxiety"] or 0.0,
            local_vibe_cozy=row["local_vibe_cozy"] or 0.0,
            local_vibe_party=row["local_vibe_party"] or 0.0,
            data_quality_score=row["data_quality_score"] or 0.0,
            spotify_quality=row["spotify_quality"] or 0.0,
            google_quality=row["google_quality"] or 0.0,
            notes=row["notes"] or "",
        )
    
    def get_latest_features(
        self,
        geo: str = "CA",
        max_age_days: int = 7,
    ) -> Optional[MoodFeatures]:
        """
        Get most recent features within age limit.
        
        Used for AOC to get "current" mood context.
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cutoff = (date.today() - timedelta(days=max_age_days)).isoformat()
        
        cursor.execute("""
            SELECT date FROM mood_features_daily
            WHERE geo = ? AND date >= ?
            ORDER BY date DESC
            LIMIT 1
        """, (geo, cutoff))
        
        row = cursor.fetchone()
        conn.close()
        
        if not row:
            return None
        
        return self.get_features_for_date(row["date"], geo)


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) < 2:
        print("Usage: python compositor.py <db_path> [date]")
        sys.exit(1)
    
    db_path = sys.argv[1]
    target_date = sys.argv[2] if len(sys.argv) > 2 else date.today().isoformat()
    
    compositor = LocalVibeCompositor(db_path)
    features = compositor.compute_daily_features(target_date)
    
    print(f"\nMood Features for {target_date}:")
    print(f"  Spotify: valence={features.spotify_valence_mean:.3f}, energy={features.spotify_energy_mean:.3f}")
    print(f"  Z-scores: valence_z={features.spotify_valence_z:.2f}, energy_z={features.spotify_energy_z:.2f}")
    print(f"\n  Google Z-scores:")
    print(f"    stress={features.google_stress_z:.2f}, cozy={features.google_cozy_z:.2f}")
    print(f"    party={features.google_party_z:.2f}, money={features.google_money_z:.2f}")
    print(f"\n  Local Vibe Indices:")
    print(f"    Overall Score: {features.local_vibe_score:.3f}")
    print(f"    Anxiety: {features.local_vibe_anxiety:.3f}")
    print(f"    Cozy: {features.local_vibe_cozy:.3f}")
    print(f"    Party: {features.local_vibe_party:.3f}")
    print(f"\n  Data Quality: {features.data_quality_score:.2f}")
    if features.notes:
        print(f"  Notes: {features.notes}")
