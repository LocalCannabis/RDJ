"""
Mood Provider Abstraction

Clean interface for mood data providers that allows runtime selection
based on API availability and data quality.

Providers:
    - SpotifyProviderQuota: Full playlist + audio features (if approved)
    - SpotifyProviderPreviewAudio: Preview URL + local feature extraction
    - SpotifyProviderMetadataOnly: Popularity/genres/recency fallback
    - GoogleTrendsProvider: Bucket-based mood signals
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Protocol

logger = logging.getLogger(__name__)


class ProviderStatus(Enum):
    """Provider availability status."""
    AVAILABLE = "available"
    DEGRADED = "degraded"  # Working but limited
    UNAVAILABLE = "unavailable"
    NOT_CONFIGURED = "not_configured"


@dataclass
class RawRecord:
    """Generic raw record from a mood provider."""
    source: str  # e.g., "spotify", "google_trends"
    record_type: str  # e.g., "track", "term", "preview"
    record_id: str
    timestamp: datetime
    data: Dict[str, Any]
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DailyFeatures:
    """Standardized daily mood features from a provider."""
    date: str  # YYYY-MM-DD
    geo: str  # e.g., "CA", "BC", "CA-BC"
    source: str  # Provider name
    
    # Core mood signals (all normalized 0-1 or z-scored)
    valence: Optional[float] = None  # Happiness/positivity
    energy: Optional[float] = None  # Intensity/arousal
    anxiety: Optional[float] = None  # Stress level
    cozy: Optional[float] = None  # Comfort/relaxation
    party: Optional[float] = None  # Social/celebratory
    
    # Additional signals
    extra: Dict[str, float] = field(default_factory=dict)
    
    # Quality metadata
    sample_size: int = 0
    coverage: float = 0.0  # 0-1, what % of expected data was available
    quality_score: float = 0.0  # 0-1, overall quality


@dataclass
class ProviderHealth:
    """Health check result for a provider."""
    provider_name: str
    status: ProviderStatus
    message: str
    last_check: datetime
    capabilities: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)


class MoodProvider(ABC):
    """
    Abstract base class for mood data providers.
    
    All providers must implement:
        - fetch_raw(): Get raw data records for a date
        - compute_features(): Transform raw records into standardized features
        - quality_score(): Assess data quality for a set of records
        - health_check(): Verify provider is working
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Provider name identifier."""
        pass
    
    @property
    @abstractmethod
    def source_type(self) -> str:
        """Source type (e.g., 'spotify', 'google', 'weather')."""
        pass
    
    @abstractmethod
    def health_check(self) -> ProviderHealth:
        """
        Check if provider is available and working.
        
        Returns:
            ProviderHealth with status and capabilities.
        """
        pass
    
    @abstractmethod
    def fetch_raw(
        self,
        target_date: date,
        geo: str = "CA",
    ) -> List[RawRecord]:
        """
        Fetch raw data for a specific date.
        
        Args:
            target_date: Date to fetch data for.
            geo: Geographic region.
            
        Returns:
            List of raw records from this provider.
        """
        pass
    
    @abstractmethod
    def compute_features(
        self,
        raw_records: List[RawRecord],
    ) -> DailyFeatures:
        """
        Transform raw records into standardized daily features.
        
        Args:
            raw_records: Raw records from fetch_raw().
            
        Returns:
            DailyFeatures with normalized mood signals.
        """
        pass
    
    def quality_score(self, raw_records: List[RawRecord]) -> float:
        """
        Compute quality score for a set of raw records.
        
        Default implementation based on coverage.
        Override for provider-specific logic.
        
        Args:
            raw_records: Raw records to assess.
            
        Returns:
            Quality score between 0 and 1.
        """
        if not raw_records:
            return 0.0
        
        # Default: quality based on record count
        # Providers should override with specific logic
        expected_min = 10
        expected_ideal = 50
        
        count = len(raw_records)
        if count >= expected_ideal:
            return 1.0
        elif count >= expected_min:
            return 0.5 + 0.5 * (count - expected_min) / (expected_ideal - expected_min)
        else:
            return 0.5 * count / expected_min


class SpotifyProviderBase(MoodProvider):
    """Base class for Spotify-based providers with shared utilities."""
    
    source_type = "spotify"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        from .spotify import SpotifyMoodClient
        self._client = SpotifyMoodClient(
            client_id=client_id,
            client_secret=client_secret,
        )
    
    def _test_auth(self) -> bool:
        """Test if authentication works."""
        try:
            self._client._ensure_token()
            return True
        except Exception as e:
            logger.warning(f"Spotify auth failed: {e}")
            return False
    
    def _test_track_access(self) -> bool:
        """Test if basic track endpoint works."""
        try:
            # Use a known stable track ID
            self._client._api_request("tracks/4cOdK2wGLETKBW3PvgPWqT")
            return True
        except Exception as e:
            logger.warning(f"Spotify track access failed: {e}")
            return False


class SpotifyProviderMetadataOnly(SpotifyProviderBase):
    """
    Fallback Spotify provider using only metadata (no audio features).
    
    Derives mood signals from:
        - popularity scores
        - explicit content rate
        - release recency
        - genre distribution
    
    This is the lowest-fidelity option but always works with basic API access.
    """
    
    name = "spotify_metadata"
    
    # Search queries to sample tracks (simulates playlist access)
    SAMPLE_QUERIES = [
        "top hits canada 2024",
        "viral hits",
        "new music friday canada",
        "chill vibes",
        "party hits",
        "workout music",
    ]
    
    def health_check(self) -> ProviderHealth:
        """Check if metadata-only access works."""
        if not self._test_auth():
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.NOT_CONFIGURED,
                message="Spotify credentials not configured or invalid",
                last_check=datetime.now(),
            )
        
        if not self._test_track_access():
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.UNAVAILABLE,
                message="Cannot access Spotify track endpoint",
                last_check=datetime.now(),
            )
        
        return ProviderHealth(
            provider_name=self.name,
            status=ProviderStatus.AVAILABLE,
            message="Metadata access working",
            last_check=datetime.now(),
            capabilities=["track_metadata", "search", "popularity"],
            limitations=["no_audio_features", "no_playlists"],
        )
    
    def fetch_raw(
        self,
        target_date: date,
        geo: str = "CA",
    ) -> List[RawRecord]:
        """
        Fetch track metadata via search queries.
        
        Note: Search results may vary day-to-day even for same query,
        providing some natural variance for mood sampling.
        """
        records = []
        seen_tracks = set()
        
        for query in self.SAMPLE_QUERIES:
            try:
                # Append geo hint to queries
                search_query = f"{query} {geo}" if geo else query
                
                data = self._client._api_request(
                    "search",
                    params={
                        "q": search_query,
                        "type": "track",
                        "market": geo if len(geo) == 2 else "CA",
                        "limit": 20,
                    }
                )
                
                items = data.get("tracks", {}).get("items", [])
                for item in items:
                    if not item:
                        continue
                    
                    track_id = item.get("id")
                    if not track_id or track_id in seen_tracks:
                        continue
                    
                    seen_tracks.add(track_id)
                    
                    # Extract metadata
                    artists = item.get("artists", [])
                    album = item.get("album", {})
                    
                    records.append(RawRecord(
                        source="spotify",
                        record_type="track_metadata",
                        record_id=track_id,
                        timestamp=datetime.now(),
                        data={
                            "track_name": item.get("name"),
                            "artist": artists[0]["name"] if artists else None,
                            "artist_ids": [a["id"] for a in artists],
                            "popularity": item.get("popularity", 0),
                            "explicit": item.get("explicit", False),
                            "duration_ms": item.get("duration_ms", 0),
                            "preview_url": item.get("preview_url"),
                            "release_date": album.get("release_date"),
                            "album_type": album.get("album_type"),
                        },
                        metadata={
                            "query": query,
                            "geo": geo,
                        }
                    ))
                    
            except Exception as e:
                logger.warning(f"Search query '{query}' failed: {e}")
                continue
        
        logger.info(f"Fetched {len(records)} track metadata records")
        return records
    
    def compute_features(
        self,
        raw_records: List[RawRecord],
    ) -> DailyFeatures:
        """
        Compute mood features from track metadata.
        
        Signals derived:
            - popularity_mean: Overall mainstream-ness (higher = more popular)
            - explicit_rate: Proportion of explicit content (proxy for intensity)
            - release_recency: How new the tracks are (proxy for "buzz")
        """
        if not raw_records:
            return DailyFeatures(
                date=date.today().isoformat(),
                geo="CA",
                source=self.name,
                quality_score=0.0,
            )
        
        # Extract values
        popularities = []
        explicit_count = 0
        recency_scores = []
        
        today = date.today()
        
        for r in raw_records:
            data = r.data
            
            pop = data.get("popularity", 0)
            if pop:
                popularities.append(pop / 100.0)  # Normalize to 0-1
            
            if data.get("explicit"):
                explicit_count += 1
            
            # Recency: days since release, capped and inverted
            release_str = data.get("release_date")
            if release_str:
                try:
                    if len(release_str) == 4:  # Year only
                        release_date = date(int(release_str), 6, 1)
                    elif len(release_str) == 7:  # YYYY-MM
                        parts = release_str.split("-")
                        release_date = date(int(parts[0]), int(parts[1]), 15)
                    else:
                        release_date = date.fromisoformat(release_str)
                    
                    days_old = (today - release_date).days
                    # Recency score: 1.0 for today, decays over ~180 days
                    recency = max(0, 1.0 - (days_old / 180))
                    recency_scores.append(recency)
                except (ValueError, TypeError):
                    pass
        
        # Compute aggregates
        n = len(raw_records)
        popularity_mean = sum(popularities) / len(popularities) if popularities else 0.5
        explicit_rate = explicit_count / n if n > 0 else 0.0
        recency_mean = sum(recency_scores) / len(recency_scores) if recency_scores else 0.5
        
        # Map to mood signals (weak proxies)
        # Higher popularity + recency = more "buzzy" = slightly more energy
        # Higher explicit rate = slightly more intensity
        energy_proxy = 0.4 + 0.3 * popularity_mean + 0.3 * explicit_rate
        valence_proxy = 0.5  # Can't really infer happiness from metadata
        
        return DailyFeatures(
            date=today.isoformat(),
            geo=raw_records[0].metadata.get("geo", "CA") if raw_records else "CA",
            source=self.name,
            valence=valence_proxy,
            energy=energy_proxy,
            extra={
                "popularity_mean": popularity_mean,
                "explicit_rate": explicit_rate,
                "recency_mean": recency_mean,
            },
            sample_size=n,
            coverage=min(1.0, n / 50),  # Expect ~50 tracks
            quality_score=self.quality_score(raw_records),
        )
    
    def quality_score(self, raw_records: List[RawRecord]) -> float:
        """Metadata-only has inherently lower quality ceiling."""
        base_score = super().quality_score(raw_records)
        # Cap at 0.6 since metadata is weak proxy
        return min(0.6, base_score * 0.6)


class SpotifyProviderPreviewAudio(SpotifyProviderBase):
    """
    Spotify provider using preview URLs + local audio feature extraction.
    
    Downloads ~30s preview clips and extracts:
        - Energy proxy: RMS loudness, spectral centroid, onset rate
        - Valence proxy: Major/minor estimation, brightness
    
    This is the "strong pivot" - independent of deprecated Spotify endpoints.
    """
    
    name = "spotify_preview_audio"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        super().__init__(client_id, client_secret)
        self.cache_dir = cache_dir
        self._audio_extractor: Optional["AudioFeatureExtractor"] = None
    
    def _get_extractor(self) -> "AudioFeatureExtractor":
        """Lazy-load audio extractor."""
        if self._audio_extractor is None:
            from .audio_extraction import AudioFeatureExtractor
            self._audio_extractor = AudioFeatureExtractor(cache_dir=self.cache_dir)
        return self._audio_extractor
    
    def health_check(self) -> ProviderHealth:
        """Check if preview audio extraction is available."""
        # First check Spotify access
        if not self._test_auth():
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.NOT_CONFIGURED,
                message="Spotify credentials not configured",
                last_check=datetime.now(),
            )
        
        if not self._test_track_access():
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.UNAVAILABLE,
                message="Cannot access Spotify API",
                last_check=datetime.now(),
            )
        
        # Check if audio libraries are available
        try:
            from .audio_extraction import AudioFeatureExtractor
            extractor = AudioFeatureExtractor()
            if not extractor.is_available():
                return ProviderHealth(
                    provider_name=self.name,
                    status=ProviderStatus.DEGRADED,
                    message="Audio extraction libraries not installed (librosa/essentia)",
                    last_check=datetime.now(),
                    capabilities=["track_metadata", "preview_urls"],
                    limitations=["no_local_extraction"],
                )
        except ImportError:
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.DEGRADED,
                message="Audio extraction module not available",
                last_check=datetime.now(),
                capabilities=["track_metadata", "preview_urls"],
                limitations=["no_local_extraction"],
            )
        
        # Check if we can get tracks with preview URLs
        try:
            data = self._client._api_request(
                "search",
                params={"q": "top hits", "type": "track", "limit": 5}
            )
            items = data.get("tracks", {}).get("items", [])
            has_previews = any(i.get("preview_url") for i in items if i)
            
            if not has_previews:
                return ProviderHealth(
                    provider_name=self.name,
                    status=ProviderStatus.DEGRADED,
                    message="No preview URLs available in search results",
                    last_check=datetime.now(),
                    capabilities=["track_metadata"],
                    limitations=["no_previews"],
                )
        except Exception as e:
            logger.warning(f"Preview check failed: {e}")
        
        return ProviderHealth(
            provider_name=self.name,
            status=ProviderStatus.AVAILABLE,
            message="Preview audio extraction available",
            last_check=datetime.now(),
            capabilities=["track_metadata", "preview_urls", "local_audio_features"],
            limitations=["preview_only_30s"],
        )
    
    def fetch_raw(
        self,
        target_date: date,
        geo: str = "CA",
    ) -> List[RawRecord]:
        """
        Fetch tracks with preview URLs and extract audio features.
        """
        # Use metadata provider to get tracks
        metadata_provider = SpotifyProviderMetadataOnly(
            client_id=self._client.client_id,
            client_secret=self._client.client_secret,
        )
        metadata_records = metadata_provider.fetch_raw(target_date, geo)
        
        # Filter to tracks with preview URLs
        records_with_preview = []
        
        for record in metadata_records:
            preview_url = record.data.get("preview_url")
            if preview_url:
                # Add audio features from local extraction
                try:
                    extractor = self._get_extractor()
                    audio_features = extractor.extract_from_url(preview_url)
                    
                    record.data["local_audio_features"] = audio_features
                    record.record_type = "track_with_audio"
                    records_with_preview.append(record)
                    
                except Exception as e:
                    logger.warning(f"Audio extraction failed for {record.record_id}: {e}")
                    # Still include the record, just without audio features
                    records_with_preview.append(record)
        
        logger.info(
            f"Fetched {len(records_with_preview)} tracks with previews "
            f"(out of {len(metadata_records)} total)"
        )
        return records_with_preview
    
    def compute_features(
        self,
        raw_records: List[RawRecord],
    ) -> DailyFeatures:
        """
        Compute mood features from local audio analysis.
        """
        if not raw_records:
            return DailyFeatures(
                date=date.today().isoformat(),
                geo="CA",
                source=self.name,
                quality_score=0.0,
            )
        
        # Aggregate local audio features
        energies = []
        valences = []
        
        for r in raw_records:
            audio = r.data.get("local_audio_features", {})
            
            if "energy_proxy" in audio:
                energies.append(audio["energy_proxy"])
            if "valence_proxy" in audio:
                valences.append(audio["valence_proxy"])
        
        energy_mean = sum(energies) / len(energies) if energies else None
        valence_mean = sum(valences) / len(valences) if valences else None
        
        # Fall back to metadata if no audio features
        if energy_mean is None:
            metadata_features = SpotifyProviderMetadataOnly(
                client_id=self._client.client_id,
                client_secret=self._client.client_secret,
            ).compute_features(raw_records)
            energy_mean = metadata_features.energy
            valence_mean = metadata_features.valence
        
        n = len(raw_records)
        audio_count = len(energies)
        
        return DailyFeatures(
            date=date.today().isoformat(),
            geo=raw_records[0].metadata.get("geo", "CA") if raw_records else "CA",
            source=self.name,
            valence=valence_mean,
            energy=energy_mean,
            extra={
                "audio_sample_count": audio_count,
                "metadata_sample_count": n,
                "audio_coverage": audio_count / n if n > 0 else 0,
            },
            sample_size=n,
            coverage=min(1.0, audio_count / 30),  # Want ~30 audio samples
            quality_score=self.quality_score(raw_records),
        )
    
    def quality_score(self, raw_records: List[RawRecord]) -> float:
        """Quality based on audio extraction success rate."""
        if not raw_records:
            return 0.0
        
        audio_count = sum(
            1 for r in raw_records 
            if r.data.get("local_audio_features")
        )
        
        total = len(raw_records)
        extraction_rate = audio_count / total if total > 0 else 0
        
        # Base score from record count, multiplied by extraction success
        base = super().quality_score(raw_records)
        return base * (0.5 + 0.5 * extraction_rate)


class GoogleTrendsProvider(MoodProvider):
    """
    Google Trends provider for mood bucket signals.
    
    Queries term buckets and aggregates into mood indices:
        - stress/anxiety
        - cozy/at-home
        - party/social
        - money pressure
        - cannabis interest (optional)
    """
    
    name = "google_trends"
    source_type = "google"
    
    # Term buckets for mood signals
    TERM_BUCKETS = {
        "stress": [
            "anxiety", "stress relief", "cant sleep", "headache",
            "meditation", "overwhelmed",
        ],
        "cozy": [
            "netflix", "baking", "comfort food", "stay home",
            "cozy", "blanket", "soup recipe",
        ],
        "party": [
            "party", "nightclub", "bars near me", "cocktail recipe",
            "going out", "festival",
        ],
        "money": [
            "payday", "budget", "cheap", "sale", "discount",
            "bills", "debt",
        ],
        "cannabis": [
            "dispensary", "weed", "cannabis", "edibles",
            "thc", "cbd",
        ],
    }
    
    def __init__(self, geo: str = "CA"):
        self.default_geo = geo
        self._client = None
    
    def _get_client(self):
        """Lazy-load Google Trends client."""
        if self._client is None:
            from .google_trends import GoogleTrendsMoodClient
            self._client = GoogleTrendsMoodClient()
        return self._client
    
    def health_check(self) -> ProviderHealth:
        """Check if Google Trends is accessible."""
        try:
            from pytrends.request import TrendReq
            
            pytrends = TrendReq(hl='en-US', tz=360)
            pytrends.build_payload(['test'], timeframe='now 1-d', geo='CA')
            df = pytrends.interest_over_time()
            
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.AVAILABLE,
                message="Google Trends accessible",
                last_check=datetime.now(),
                capabilities=["term_trends", "geo_filtering", "time_series"],
                limitations=["rate_limited", "relative_values_only"],
            )
        except ImportError:
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.NOT_CONFIGURED,
                message="pytrends not installed",
                last_check=datetime.now(),
            )
        except Exception as e:
            return ProviderHealth(
                provider_name=self.name,
                status=ProviderStatus.UNAVAILABLE,
                message=f"Google Trends error: {str(e)[:100]}",
                last_check=datetime.now(),
            )
    
    def fetch_raw(
        self,
        target_date: date,
        geo: str = "CA",
    ) -> List[RawRecord]:
        """
        Fetch term trends for all buckets.
        """
        from pytrends.request import TrendReq
        import time
        
        records = []
        pytrends = TrendReq(hl='en-US', tz=360)
        
        for bucket_name, terms in self.TERM_BUCKETS.items():
            # Batch terms (pytrends limit is 5 per request)
            for i in range(0, len(terms), 5):
                batch = terms[i:i + 5]
                
                try:
                    pytrends.build_payload(
                        batch,
                        timeframe='today 1-m',  # Last month
                        geo=geo,
                    )
                    
                    df = pytrends.interest_over_time()
                    
                    if df.empty:
                        continue
                    
                    # Get most recent values
                    latest = df.iloc[-1]
                    
                    for term in batch:
                        if term in df.columns:
                            records.append(RawRecord(
                                source="google_trends",
                                record_type="term_trend",
                                record_id=f"{bucket_name}:{term}",
                                timestamp=datetime.now(),
                                data={
                                    "term": term,
                                    "bucket": bucket_name,
                                    "value": int(latest[term]),
                                    "date": str(df.index[-1].date()),
                                },
                                metadata={"geo": geo},
                            ))
                    
                    # Rate limit protection
                    time.sleep(1)
                    
                except Exception as e:
                    logger.warning(f"Google Trends fetch failed for {batch}: {e}")
                    continue
        
        logger.info(f"Fetched {len(records)} Google Trends records")
        return records
    
    def compute_features(
        self,
        raw_records: List[RawRecord],
    ) -> DailyFeatures:
        """
        Aggregate term trends into mood bucket scores.
        """
        if not raw_records:
            return DailyFeatures(
                date=date.today().isoformat(),
                geo=self.default_geo,
                source=self.name,
                quality_score=0.0,
            )
        
        # Aggregate by bucket
        bucket_values: Dict[str, List[float]] = {
            bucket: [] for bucket in self.TERM_BUCKETS.keys()
        }
        
        for r in raw_records:
            bucket = r.data.get("bucket")
            value = r.data.get("value", 0)
            if bucket in bucket_values:
                bucket_values[bucket].append(value / 100.0)  # Normalize 0-1
        
        # Compute bucket means
        bucket_scores = {}
        for bucket, values in bucket_values.items():
            if values:
                bucket_scores[bucket] = sum(values) / len(values)
            else:
                bucket_scores[bucket] = 0.5  # Neutral default
        
        # Map to mood features
        geo = raw_records[0].metadata.get("geo", self.default_geo)
        
        return DailyFeatures(
            date=date.today().isoformat(),
            geo=geo,
            source=self.name,
            anxiety=bucket_scores.get("stress", 0.5),
            cozy=bucket_scores.get("cozy", 0.5),
            party=bucket_scores.get("party", 0.5),
            extra={
                "money_pressure": bucket_scores.get("money", 0.5),
                "cannabis_interest": bucket_scores.get("cannabis", 0.5),
            },
            sample_size=len(raw_records),
            coverage=len(bucket_scores) / len(self.TERM_BUCKETS),
            quality_score=self.quality_score(raw_records),
        )
    
    def quality_score(self, raw_records: List[RawRecord]) -> float:
        """Quality based on bucket coverage."""
        if not raw_records:
            return 0.0
        
        # Check how many buckets have data
        buckets_with_data = set()
        for r in raw_records:
            bucket = r.data.get("bucket")
            if bucket:
                buckets_with_data.add(bucket)
        
        coverage = len(buckets_with_data) / len(self.TERM_BUCKETS)
        
        # Also factor in number of terms per bucket
        terms_per_bucket = len(raw_records) / max(1, len(buckets_with_data))
        term_coverage = min(1.0, terms_per_bucket / 4)  # Want ~4 terms per bucket
        
        return coverage * 0.6 + term_coverage * 0.4


# Provider registry for runtime selection
PROVIDERS = {
    "spotify_metadata": SpotifyProviderMetadataOnly,
    "spotify_preview_audio": SpotifyProviderPreviewAudio,
    "google_trends": GoogleTrendsProvider,
}


def get_available_providers() -> Dict[str, ProviderHealth]:
    """
    Check all providers and return their health status.
    
    Returns:
        Dict of provider_name -> ProviderHealth
    """
    results = {}
    
    for name, provider_class in PROVIDERS.items():
        try:
            provider = provider_class()
            results[name] = provider.health_check()
        except Exception as e:
            results[name] = ProviderHealth(
                provider_name=name,
                status=ProviderStatus.UNAVAILABLE,
                message=f"Failed to instantiate: {str(e)[:100]}",
                last_check=datetime.now(),
            )
    
    return results


if __name__ == "__main__":
    # Quick test
    import sys
    logging.basicConfig(level=logging.INFO)
    
    print("Checking provider availability...\n")
    
    for name, health in get_available_providers().items():
        status_icon = {
            ProviderStatus.AVAILABLE: "✓",
            ProviderStatus.DEGRADED: "~",
            ProviderStatus.UNAVAILABLE: "✗",
            ProviderStatus.NOT_CONFIGURED: "○",
        }.get(health.status, "?")
        
        print(f"{status_icon} {name}: {health.status.value}")
        print(f"  {health.message}")
        if health.capabilities:
            print(f"  Capabilities: {', '.join(health.capabilities)}")
        if health.limitations:
            print(f"  Limitations: {', '.join(health.limitations)}")
        print()
