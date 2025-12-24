"""
Mood Orchestrator

Runtime provider selection and aggregation for the mood pipeline.
Selects the best available providers and combines their signals
into final mood scores for AOC consumption.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from .providers import (
    DailyFeatures,
    GoogleTrendsProvider,
    MoodProvider,
    ProviderHealth,
    ProviderStatus,
    RawRecord,
    SpotifyProviderMetadataOnly,
    SpotifyProviderPreviewAudio,
    get_available_providers,
)
from .baseline import RollingBaseline, NormalizedMood, interpret_z_score

logger = logging.getLogger(__name__)


@dataclass
class AggregatedMood:
    """Final aggregated mood signals from all providers."""
    date: str
    geo: str
    
    # Primary mood indices (0-1 scale)
    local_vibe_score: float = 0.5  # Overall mood score
    local_vibe_anxiety: float = 0.5  # Stress/anxiety level
    local_vibe_cozy: float = 0.5  # Comfort/relaxation
    local_vibe_party: float = 0.5  # Social/energy
    
    # Music-specific (from Spotify)
    music_valence: Optional[float] = None
    music_energy: Optional[float] = None
    
    # Z-scored versions for AOC (relative to rolling baseline)
    local_vibe_z: Optional[float] = None
    anxiety_z: Optional[float] = None
    cozy_z: Optional[float] = None
    party_z: Optional[float] = None
    
    # Quality metadata
    spotify_quality: float = 0.0
    google_quality: float = 0.0
    overall_quality: float = 0.0
    
    # Source tracking
    spotify_provider: Optional[str] = None
    providers_used: List[str] = field(default_factory=list)
    
    # Raw contributions
    spotify_features: Optional[DailyFeatures] = None
    google_features: Optional[DailyFeatures] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage."""
        return {
            "date": self.date,
            "geo": self.geo,
            "local_vibe_score": self.local_vibe_score,
            "local_vibe_anxiety": self.local_vibe_anxiety,
            "local_vibe_cozy": self.local_vibe_cozy,
            "local_vibe_party": self.local_vibe_party,
            "music_valence": self.music_valence,
            "music_energy": self.music_energy,
            "local_vibe_z": self.local_vibe_z,
            "anxiety_z": self.anxiety_z,
            "cozy_z": self.cozy_z,
            "party_z": self.party_z,
            "spotify_quality": self.spotify_quality,
            "google_quality": self.google_quality,
            "overall_quality": self.overall_quality,
            "spotify_provider": self.spotify_provider,
            "providers_used": ",".join(self.providers_used),
        }


class MoodOrchestrator:
    """
    Orchestrates mood data collection from multiple providers.
    
    Responsibilities:
        - Check provider availability at runtime
        - Select best available Spotify provider
        - Fetch data from all available providers
        - Aggregate into final mood signals
        - Store raw and derived data
    """
    
    # Provider priority for Spotify (best first)
    SPOTIFY_PRIORITY = [
        "spotify_preview_audio",  # Best: local extraction
        "spotify_metadata",  # Fallback: metadata only
    ]
    
    # Weights for combining signals
    WEIGHTS = {
        "spotify": 0.35,  # Music mood
        "google": 0.65,  # Search trends (main signal)
    }
    
    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        geo: str = "CA",
    ):
        """
        Initialize orchestrator.
        
        Args:
            db_path: Path to SQLite database for storage.
            geo: Default geographic region.
        """
        self.db_path = Path(db_path) if db_path else None
        self.default_geo = geo
        
        self._spotify_provider: Optional[MoodProvider] = None
        self._google_provider: Optional[MoodProvider] = None
        self._provider_health: Dict[str, ProviderHealth] = {}
    
    def check_providers(self) -> Dict[str, ProviderHealth]:
        """
        Check availability of all providers.
        
        Returns:
            Dict of provider_name -> ProviderHealth
        """
        self._provider_health = get_available_providers()
        return self._provider_health
    
    def select_spotify_provider(self) -> Optional[MoodProvider]:
        """
        Select the best available Spotify provider.
        
        Returns:
            Best available SpotifyProvider or None.
        """
        if not self._provider_health:
            self.check_providers()
        
        for provider_name in self.SPOTIFY_PRIORITY:
            health = self._provider_health.get(provider_name)
            
            if health and health.status in (
                ProviderStatus.AVAILABLE,
                ProviderStatus.DEGRADED,
            ):
                logger.info(f"Selected Spotify provider: {provider_name}")
                
                if provider_name == "spotify_preview_audio":
                    self._spotify_provider = SpotifyProviderPreviewAudio()
                elif provider_name == "spotify_metadata":
                    self._spotify_provider = SpotifyProviderMetadataOnly()
                
                return self._spotify_provider
        
        logger.warning("No Spotify provider available")
        return None
    
    def select_google_provider(self) -> Optional[MoodProvider]:
        """
        Select Google Trends provider if available.
        
        Returns:
            GoogleTrendsProvider or None.
        """
        if not self._provider_health:
            self.check_providers()
        
        health = self._provider_health.get("google_trends")
        
        if health and health.status == ProviderStatus.AVAILABLE:
            logger.info("Google Trends provider available")
            self._google_provider = GoogleTrendsProvider(geo=self.default_geo)
            return self._google_provider
        
        logger.warning("Google Trends provider not available")
        return None
    
    def fetch_daily_mood(
        self,
        target_date: Optional[date] = None,
        geo: Optional[str] = None,
    ) -> AggregatedMood:
        """
        Fetch and aggregate mood data for a day.
        
        Args:
            target_date: Date to fetch (defaults to today).
            geo: Geographic region (defaults to instance default).
            
        Returns:
            AggregatedMood with combined signals from all providers.
        """
        target_date = target_date or date.today()
        geo = geo or self.default_geo
        
        logger.info(f"Fetching mood data for {target_date} ({geo})")
        
        # Ensure providers are selected
        if self._spotify_provider is None:
            self.select_spotify_provider()
        if self._google_provider is None:
            self.select_google_provider()
        
        # Collect from each provider
        spotify_features = None
        google_features = None
        providers_used = []
        
        # Spotify (with fallback to metadata if preview yields no data)
        if self._spotify_provider:
            try:
                raw_records = self._spotify_provider.fetch_raw(target_date, geo)
                spotify_features = self._spotify_provider.compute_features(raw_records)
                
                # Fallback: if preview provider got zero samples, try metadata
                if (
                    spotify_features.sample_size == 0
                    and self._spotify_provider.name == "spotify_preview_audio"
                ):
                    logger.info("Preview provider yielded no samples, falling back to metadata")
                    metadata_provider = SpotifyProviderMetadataOnly()
                    raw_records = metadata_provider.fetch_raw(target_date, geo)
                    spotify_features = metadata_provider.compute_features(raw_records)
                    self._spotify_provider = metadata_provider  # Update for future calls
                
                providers_used.append(self._spotify_provider.name)
                logger.info(
                    f"Spotify: {spotify_features.sample_size} samples, "
                    f"quality={spotify_features.quality_score:.2f}"
                )
            except Exception as e:
                logger.error(f"Spotify fetch failed: {e}")
        
        # Google Trends
        if self._google_provider:
            try:
                raw_records = self._google_provider.fetch_raw(target_date, geo)
                google_features = self._google_provider.compute_features(raw_records)
                providers_used.append(self._google_provider.name)
                logger.info(
                    f"Google: {google_features.sample_size} samples, "
                    f"quality={google_features.quality_score:.2f}"
                )
            except Exception as e:
                logger.error(f"Google Trends fetch failed: {e}")
        
        # Aggregate
        return self._aggregate_features(
            target_date=target_date,
            geo=geo,
            spotify_features=spotify_features,
            google_features=google_features,
            providers_used=providers_used,
        )
    
    def _aggregate_features(
        self,
        target_date: date,
        geo: str,
        spotify_features: Optional[DailyFeatures],
        google_features: Optional[DailyFeatures],
        providers_used: List[str],
    ) -> AggregatedMood:
        """
        Combine features from multiple providers into final mood.
        
        Weighting:
            - Google Trends is anchor signal (65%)
            - Spotify is supplementary (35%)
            - Weights redistribute if one source missing
        """
        result = AggregatedMood(
            date=target_date.isoformat(),
            geo=geo,
            providers_used=providers_used,
        )
        
        # Default neutral values
        spotify_anxiety = 0.5
        spotify_cozy = 0.5
        spotify_party = 0.5
        google_anxiety = 0.5
        google_cozy = 0.5
        google_party = 0.5
        
        # Extract Spotify signals
        if spotify_features:
            result.spotify_features = spotify_features
            result.spotify_provider = spotify_features.source
            result.spotify_quality = spotify_features.quality_score
            
            result.music_valence = spotify_features.valence
            result.music_energy = spotify_features.energy
            
            # Map energy/valence to mood dimensions
            # Low valence + low energy → anxiety
            # High valence + low energy → cozy
            # High valence + high energy → party
            v = spotify_features.valence or 0.5
            e = spotify_features.energy or 0.5
            
            spotify_anxiety = max(0, min(1, 0.7 - 0.4 * v - 0.3 * e))
            spotify_cozy = max(0, min(1, 0.3 + 0.4 * v - 0.3 * e))
            spotify_party = max(0, min(1, 0.3 * v + 0.7 * e))
        
        # Extract Google signals
        if google_features:
            result.google_features = google_features
            result.google_quality = google_features.quality_score
            
            google_anxiety = google_features.anxiety or 0.5
            google_cozy = google_features.cozy or 0.5
            google_party = google_features.party or 0.5
        
        # Compute weights based on availability
        spotify_weight = self.WEIGHTS["spotify"] if spotify_features else 0
        google_weight = self.WEIGHTS["google"] if google_features else 0
        
        total_weight = spotify_weight + google_weight
        if total_weight > 0:
            spotify_weight /= total_weight
            google_weight /= total_weight
        else:
            # No data - use defaults
            spotify_weight = 0
            google_weight = 0
        
        # Weighted combination
        result.local_vibe_anxiety = (
            spotify_weight * spotify_anxiety +
            google_weight * google_anxiety
        )
        result.local_vibe_cozy = (
            spotify_weight * spotify_cozy +
            google_weight * google_cozy
        )
        result.local_vibe_party = (
            spotify_weight * spotify_party +
            google_weight * google_party
        )
        
        # Overall vibe score
        # Higher cozy + party - anxiety = positive vibe
        result.local_vibe_score = (
            0.3 * result.local_vibe_cozy +
            0.4 * result.local_vibe_party +
            0.3 * (1 - result.local_vibe_anxiety)
        )
        
        # Overall quality
        result.overall_quality = (
            spotify_weight * result.spotify_quality +
            google_weight * result.google_quality
        )
        
        return result
    
    def store_mood(self, mood: AggregatedMood) -> None:
        """
        Store aggregated mood to database.
        
        Stores both:
        - Intraday sample (mood_samples_intraday) - each collection
        - Daily aggregate (mood_features_daily) - upserts, keeps latest
        
        Args:
            mood: AggregatedMood to store.
        """
        if not self.db_path:
            logger.warning("No database path configured, skipping storage")
            return
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now()
        
        try:
            # Ensure intraday table exists
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS mood_samples_intraday (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    date TEXT NOT NULL,
                    hour INTEGER NOT NULL,
                    geo TEXT NOT NULL,
                    vibe_score REAL,
                    anxiety REAL,
                    cozy REAL,
                    party REAL,
                    spotify_energy REAL,
                    spotify_popularity REAL,
                    spotify_sample_count INTEGER,
                    google_stress REAL,
                    google_cozy REAL,
                    google_party REAL,
                    google_available INTEGER DEFAULT 0,
                    quality_score REAL,
                    providers TEXT,
                    UNIQUE(timestamp, geo)
                )
            """)
            
            # Store intraday sample
            spotify_popularity = None
            spotify_sample_count = None
            if mood.spotify_features and mood.spotify_features.extra:
                spotify_popularity = mood.spotify_features.extra.get("popularity_mean")
                spotify_sample_count = mood.spotify_features.sample_size
            
            google_stress = None
            google_cozy = None
            google_party = None
            google_available = 0
            if mood.google_features:
                google_stress = mood.google_features.anxiety
                google_cozy = mood.google_features.cozy
                google_party = mood.google_features.party
                google_available = 1
            
            cursor.execute("""
                INSERT OR REPLACE INTO mood_samples_intraday (
                    timestamp, date, hour, geo,
                    vibe_score, anxiety, cozy, party,
                    spotify_energy, spotify_popularity, spotify_sample_count,
                    google_stress, google_cozy, google_party, google_available,
                    quality_score, providers
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                now.isoformat(),
                mood.date,
                now.hour,
                mood.geo,
                mood.local_vibe_score,
                mood.local_vibe_anxiety,
                mood.local_vibe_cozy,
                mood.local_vibe_party,
                mood.music_energy,
                spotify_popularity,
                spotify_sample_count,
                google_stress,
                google_cozy,
                google_party,
                google_available,
                mood.overall_quality,
                ",".join(mood.providers_used),
            ))
            
            logger.info(f"Stored intraday sample for {mood.date} {now.hour:02d}:00")
            
            # Upsert into mood_features_daily (keeps latest of the day)
            cursor.execute("""
                INSERT OR REPLACE INTO mood_features_daily (
                    date, geo,
                    spotify_valence_mean, spotify_energy_mean,
                    spotify_valence_z, spotify_energy_z,
                    google_stress_z, google_cozy_z, google_party_z,
                    local_vibe_score, local_vibe_anxiety,
                    local_vibe_cozy, local_vibe_party,
                    local_vibe_z, anxiety_z, cozy_z, party_z,
                    data_quality_score, spotify_quality, google_quality,
                    notes, computed_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mood.date,
                mood.geo,
                mood.music_valence,
                mood.music_energy,
                None,  # spotify_valence_z (legacy)
                None,  # spotify_energy_z (legacy)
                None,  # google_stress_z (legacy)
                None,  # google_cozy_z (legacy)
                None,  # google_party_z (legacy)
                mood.local_vibe_score,
                mood.local_vibe_anxiety,
                mood.local_vibe_cozy,
                mood.local_vibe_party,
                mood.local_vibe_z,
                mood.anxiety_z,
                mood.cozy_z,
                mood.party_z,
                mood.overall_quality,
                mood.spotify_quality,
                mood.google_quality,
                f"providers: {','.join(mood.providers_used)}",
                now.isoformat(),
            ))
            
            conn.commit()
            logger.info(f"Updated daily aggregate for {mood.date}")
            
        except Exception as e:
            logger.error(f"Failed to store mood: {e}")
            conn.rollback()
            raise
        finally:
            conn.close()
    
    def get_status(self) -> Dict[str, Any]:
        """
        Get current orchestrator status for API/UI.
        
        Returns:
            Status dict with provider info and capabilities.
        """
        if not self._provider_health:
            self.check_providers()
        
        spotify_status = "not_configured"
        spotify_provider = None
        
        for name in self.SPOTIFY_PRIORITY:
            health = self._provider_health.get(name)
            if health:
                if health.status == ProviderStatus.AVAILABLE:
                    spotify_status = "connected"
                    spotify_provider = name
                    break
                elif health.status == ProviderStatus.DEGRADED:
                    spotify_status = "degraded"
                    spotify_provider = name
                    break
        
        google_health = self._provider_health.get("google_trends")
        google_status = "not_configured"
        if google_health:
            if google_health.status == ProviderStatus.AVAILABLE:
                google_status = "connected"
            elif google_health.status == ProviderStatus.DEGRADED:
                google_status = "degraded"
        
        return {
            "spotify": {
                "status": spotify_status,
                "provider": spotify_provider,
                "capabilities": (
                    self._provider_health.get(spotify_provider, {}).capabilities
                    if spotify_provider else []
                ),
            },
            "google_trends": {
                "status": google_status,
            },
            "overall_status": (
                "operational" if spotify_status == "connected" or google_status == "connected"
                else "degraded" if spotify_status == "degraded" or google_status == "degraded"
                else "offline"
            ),
        }
    
    def normalize_mood(
        self,
        mood: AggregatedMood,
    ) -> NormalizedMood:
        """
        Normalize raw mood against rolling baseline.
        
        Computes z-scores relative to 30-day historical baseline.
        
        Args:
            mood: AggregatedMood to normalize.
            
        Returns:
            NormalizedMood with z-scored values.
        """
        if not self.db_path:
            logger.warning("No database path, cannot compute baseline")
            return NormalizedMood(
                date=mood.date,
                geo=mood.geo,
                raw_anxiety=mood.local_vibe_anxiety,
                raw_cozy=mood.local_vibe_cozy,
                raw_party=mood.local_vibe_party,
                raw_vibe=mood.local_vibe_score,
                anxiety_z=0.0,
                cozy_z=0.0,
                party_z=0.0,
                vibe_z=0.0,
            )
        
        baseline = RollingBaseline(db_path=self.db_path)
        
        normalized = baseline.normalize_mood(
            raw_anxiety=mood.local_vibe_anxiety,
            raw_cozy=mood.local_vibe_cozy,
            raw_party=mood.local_vibe_party,
            raw_vibe=mood.local_vibe_score,
            geo=mood.geo,
            target_date=date.fromisoformat(mood.date),
        )
        
        # Update the AggregatedMood with z-scores
        mood.anxiety_z = normalized.anxiety_z
        mood.cozy_z = normalized.cozy_z
        mood.party_z = normalized.party_z
        mood.local_vibe_z = normalized.vibe_z
        
        return normalized


def run_daily_collection(
    db_path: str,
    geo: str = "CA",
    normalize: bool = True,
) -> AggregatedMood:
    """
    Convenience function to run daily mood collection.
    
    Args:
        db_path: Path to SQLite database.
        geo: Geographic region.
        normalize: Whether to compute z-scores against baseline.
        
    Returns:
        Collected and stored AggregatedMood.
    """
    orchestrator = MoodOrchestrator(db_path=db_path, geo=geo)
    
    # Check providers
    print("Checking providers...")
    health = orchestrator.check_providers()
    for name, h in health.items():
        status_icon = "✓" if h.status == ProviderStatus.AVAILABLE else "○"
        print(f"  {status_icon} {name}: {h.status.value}")
    
    # Fetch mood
    print(f"\nFetching mood for {date.today()} ({geo})...")
    mood = orchestrator.fetch_daily_mood()
    
    print(f"\nRaw Results:")
    print(f"  Local Vibe Score: {mood.local_vibe_score:.2f}")
    print(f"  Anxiety: {mood.local_vibe_anxiety:.2f}")
    print(f"  Cozy: {mood.local_vibe_cozy:.2f}")
    print(f"  Party: {mood.local_vibe_party:.2f}")
    print(f"  Quality: {mood.overall_quality:.2f}")
    print(f"  Providers: {', '.join(mood.providers_used)}")
    
    # Normalize against baseline
    if normalize and db_path:
        print(f"\nNormalizing against 30-day baseline...")
        normalized = orchestrator.normalize_mood(mood)
        
        print(f"\nZ-Scored Results (vs baseline):")
        print(f"  Vibe Z: {mood.local_vibe_z:+.2f} ({interpret_z_score(mood.local_vibe_z)})")
        print(f"  Anxiety Z: {mood.anxiety_z:+.2f} ({interpret_z_score(mood.anxiety_z)})")
        print(f"  Cozy Z: {mood.cozy_z:+.2f} ({interpret_z_score(mood.cozy_z)})")
        print(f"  Party Z: {mood.party_z:+.2f} ({interpret_z_score(mood.party_z)})")
        print(f"  Baseline coverage: {normalized.baseline_coverage:.0%}")
    
    # Store
    if db_path:
        orchestrator.store_mood(mood)
        print(f"\nStored to {db_path}")
    
    return mood
    
    return mood


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    db_path = sys.argv[1] if len(sys.argv) > 1 else "aoc_sales.db"
    geo = sys.argv[2] if len(sys.argv) > 2 else "CA"
    
    run_daily_collection(db_path, geo)
