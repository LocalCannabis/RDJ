"""
AOC Mood Data Pipeline

Privacy-safe, store-agnostic "Local Vibe" signal pipeline that collects
daily mood proxies from Spotify + Google, normalizes them, and generates
interpretable indices for AOC decision engines.

Modules:
    - schema: Database table definitions and initialization
    - providers: Pluggable mood data providers (Spotify, Google, etc.)
    - audio_extraction: Local audio feature extraction from previews
    - orchestrator: Runtime provider selection and aggregation
    - spotify: Spotify Web API client for playlist/audio features
    - google_trends: Google Trends client for mood bucket signals
    - compositor: Local Vibe index computation
"""

from .schema import init_mood_tables, MOOD_TABLES
from .spotify import SpotifyMoodClient
from .google_trends import GoogleTrendsMoodClient
from .compositor import LocalVibeCompositor, MoodFeatures
from .providers import (
    MoodProvider,
    ProviderStatus,
    ProviderHealth,
    RawRecord,
    DailyFeatures,
    SpotifyProviderMetadataOnly,
    SpotifyProviderPreviewAudio,
    GoogleTrendsProvider,
    get_available_providers,
)
from .orchestrator import (
    MoodOrchestrator,
    AggregatedMood,
    run_daily_collection,
)
from .baseline import (
    RollingBaseline,
    BaselineStats,
    NormalizedMood,
    interpret_z_score,
)

__all__ = [
    # Schema
    "init_mood_tables",
    "MOOD_TABLES",
    # Legacy clients
    "SpotifyMoodClient",
    "GoogleTrendsMoodClient",
    # Compositor
    "LocalVibeCompositor",
    "MoodFeatures",
    # Provider abstraction
    "MoodProvider",
    "ProviderStatus",
    "ProviderHealth",
    "RawRecord",
    "DailyFeatures",
    "SpotifyProviderMetadataOnly",
    "SpotifyProviderPreviewAudio",
    "GoogleTrendsProvider",
    "get_available_providers",
    # Orchestrator
    "MoodOrchestrator",
    "AggregatedMood",
    "run_daily_collection",
    # Baseline normalization
    "RollingBaseline",
    "BaselineStats",
    "NormalizedMood",
    "interpret_z_score",
]
