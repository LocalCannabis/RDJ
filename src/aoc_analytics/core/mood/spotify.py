"""
Spotify Mood Client

Fetches playlist tracks and audio features from Spotify Web API
to derive mood signals (valence, energy, danceability).

Uses Client Credentials Flow (no user auth required).

Environment Variables:
    SPOTIFY_CLIENT_ID: Spotify application client ID
    SPOTIFY_CLIENT_SECRET: Spotify application client secret
"""

from __future__ import annotations

import base64
import logging
import os
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import httpx

logger = logging.getLogger(__name__)


# Canada-specific playlists
CANADA_PLAYLISTS = {
    "top_50_canada": "37i9dQZEVXbMDoHDwVN2tF",
    "viral_50_canada": "37i9dQZEVXbKfIuOAZrk7G",
}

# Fallback playlists if Canada-specific fail
FALLBACK_PLAYLISTS = {
    "top_50_global": "37i9dQZEVXbMDoHDwVN2tF",
}

# Audio feature fields we care about
AUDIO_FEATURE_FIELDS = [
    "valence", "energy", "danceability", "tempo",
    "speechiness", "acousticness", "instrumentalness", "liveness",
    "duration_ms", "time_signature", "key", "mode"
]


@dataclass
class TrackInfo:
    """Basic track information from a playlist."""
    track_id: str
    track_name: str
    artist: str
    rank: int
    
    
@dataclass
class AudioFeatures:
    """Spotify audio features for a track."""
    track_id: str
    valence: float = 0.0
    energy: float = 0.0
    danceability: float = 0.0
    tempo: float = 0.0
    speechiness: float = 0.0
    acousticness: float = 0.0
    instrumentalness: float = 0.0
    liveness: float = 0.0
    duration_ms: int = 0
    time_signature: int = 4
    key: int = 0
    mode: int = 1
    
    @classmethod
    def from_api_response(cls, data: dict) -> Optional["AudioFeatures"]:
        """Create from Spotify API response."""
        if not data or not data.get("id"):
            return None
        return cls(
            track_id=data["id"],
            valence=data.get("valence", 0.0),
            energy=data.get("energy", 0.0),
            danceability=data.get("danceability", 0.0),
            tempo=data.get("tempo", 0.0),
            speechiness=data.get("speechiness", 0.0),
            acousticness=data.get("acousticness", 0.0),
            instrumentalness=data.get("instrumentalness", 0.0),
            liveness=data.get("liveness", 0.0),
            duration_ms=data.get("duration_ms", 0),
            time_signature=data.get("time_signature", 4),
            key=data.get("key", 0),
            mode=data.get("mode", 1),
        )


@dataclass
class DailyMoodSnapshot:
    """Aggregated mood metrics for a day."""
    date: str
    playlist_count: int
    track_count: int
    valence_mean: float
    energy_mean: float
    danceability_mean: float
    tempo_mean: float
    valence_p25: float = 0.0
    valence_p75: float = 0.0
    energy_p25: float = 0.0
    energy_p75: float = 0.0


class SpotifyMoodClient:
    """
    Client for fetching mood-related data from Spotify.
    
    Uses Client Credentials Flow for authentication.
    Fetches playlists and audio features to derive mood signals.
    """
    
    TOKEN_URL = "https://accounts.spotify.com/api/token"
    API_BASE = "https://api.spotify.com/v1"
    
    def __init__(
        self,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
        db_path: Optional[str | Path] = None,
    ):
        """
        Initialize Spotify client.
        
        Args:
            client_id: Spotify client ID (or from SPOTIFY_CLIENT_ID env)
            client_secret: Spotify client secret (or from SPOTIFY_CLIENT_SECRET env)
            db_path: Path to SQLite database for caching/storing
        """
        self.client_id = client_id or os.getenv("SPOTIFY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIFY_CLIENT_SECRET")
        self.db_path = Path(db_path) if db_path else None
        
        self._access_token: Optional[str] = None
        self._token_expires_at: float = 0
        
        self._http_client = httpx.Client(timeout=30.0)
        
    def __del__(self):
        if hasattr(self, "_http_client"):
            self._http_client.close()
    
    def _get_auth_header(self) -> str:
        """Get base64 encoded auth header."""
        if not self.client_id or not self.client_secret:
            raise ValueError(
                "Spotify credentials not configured. "
                "Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables."
            )
        auth_str = f"{self.client_id}:{self.client_secret}"
        return base64.b64encode(auth_str.encode()).decode()
    
    def _ensure_token(self) -> str:
        """Ensure we have a valid access token."""
        if self._access_token and time.time() < self._token_expires_at:
            return self._access_token
        
        logger.debug("Fetching new Spotify access token")
        
        response = self._http_client.post(
            self.TOKEN_URL,
            headers={
                "Authorization": f"Basic {self._get_auth_header()}",
                "Content-Type": "application/x-www-form-urlencoded",
            },
            data={"grant_type": "client_credentials"},
        )
        response.raise_for_status()
        
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires_at = time.time() + data.get("expires_in", 3600) - 60
        
        logger.debug("Spotify access token refreshed")
        return self._access_token
    
    def _api_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make authenticated API request."""
        token = self._ensure_token()
        
        url = f"{self.API_BASE}/{endpoint}"
        response = self._http_client.get(
            url,
            headers={"Authorization": f"Bearer {token}"},
            params=params,
        )
        response.raise_for_status()
        return response.json()
    
    def fetch_playlist_tracks(
        self,
        playlist_id: str,
        limit: int = 50,
    ) -> Tuple[str, List[TrackInfo]]:
        """
        Fetch tracks from a playlist.
        
        Args:
            playlist_id: Spotify playlist ID
            limit: Maximum number of tracks to fetch
            
        Returns:
            Tuple of (playlist_name, list of TrackInfo)
        """
        logger.info(f"Fetching playlist {playlist_id}")
        
        # Get playlist info
        data = self._api_request(f"playlists/{playlist_id}")
        playlist_name = data.get("name", "Unknown")
        
        # Get tracks
        tracks_data = data.get("tracks", {}).get("items", [])
        
        tracks = []
        for i, item in enumerate(tracks_data[:limit]):
            track = item.get("track")
            if not track or not track.get("id"):
                continue
            
            # Get primary artist
            artists = track.get("artists", [])
            artist_name = artists[0]["name"] if artists else "Unknown"
            
            tracks.append(TrackInfo(
                track_id=track["id"],
                track_name=track.get("name", "Unknown"),
                artist=artist_name,
                rank=i + 1,
            ))
        
        logger.info(f"Fetched {len(tracks)} tracks from '{playlist_name}'")
        return playlist_name, tracks
    
    def fetch_audio_features(
        self,
        track_ids: List[str],
    ) -> List[AudioFeatures]:
        """
        Fetch audio features for tracks (batch up to 100).
        
        Args:
            track_ids: List of Spotify track IDs
            
        Returns:
            List of AudioFeatures
        """
        if not track_ids:
            return []
        
        features = []
        
        # Batch in groups of 100 (Spotify limit)
        for i in range(0, len(track_ids), 100):
            batch = track_ids[i:i + 100]
            ids_str = ",".join(batch)
            
            logger.debug(f"Fetching audio features for {len(batch)} tracks")
            
            try:
                data = self._api_request("audio-features", params={"ids": ids_str})
                
                for item in data.get("audio_features", []):
                    if item:
                        af = AudioFeatures.from_api_response(item)
                        if af:
                            features.append(af)
            except Exception as e:
                logger.error(f"Error fetching audio features: {e}")
        
        logger.info(f"Fetched audio features for {len(features)} tracks")
        return features
    
    def fetch_daily_snapshot(
        self,
        playlists: Optional[Dict[str, str]] = None,
        snapshot_date: Optional[str] = None,
    ) -> DailyMoodSnapshot:
        """
        Fetch and aggregate mood data for a day.
        
        Args:
            playlists: Dict of name -> playlist_id (defaults to Canada playlists)
            snapshot_date: Date string (defaults to today)
            
        Returns:
            DailyMoodSnapshot with aggregated metrics
        """
        playlists = playlists or CANADA_PLAYLISTS
        snapshot_date = snapshot_date or date.today().isoformat()
        
        all_tracks: List[TrackInfo] = []
        all_features: List[AudioFeatures] = []
        playlist_count = 0
        
        # Fetch from each playlist
        for name, playlist_id in playlists.items():
            try:
                playlist_name, tracks = self.fetch_playlist_tracks(playlist_id)
                all_tracks.extend(tracks)
                playlist_count += 1
                
                # Fetch audio features
                track_ids = [t.track_id for t in tracks]
                features = self.fetch_audio_features(track_ids)
                all_features.extend(features)
                
            except Exception as e:
                logger.warning(f"Failed to fetch playlist {name}: {e}")
        
        # Aggregate metrics
        if not all_features:
            logger.warning("No audio features collected")
            return DailyMoodSnapshot(
                date=snapshot_date,
                playlist_count=playlist_count,
                track_count=0,
                valence_mean=0.5,
                energy_mean=0.5,
                danceability_mean=0.5,
                tempo_mean=120.0,
            )
        
        # Calculate means
        valences = [f.valence for f in all_features]
        energies = [f.energy for f in all_features]
        danceabilities = [f.danceability for f in all_features]
        tempos = [f.tempo for f in all_features]
        
        def mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0
        
        def percentile(vals: List[float], p: float) -> float:
            if not vals:
                return 0.0
            sorted_vals = sorted(vals)
            idx = int(len(sorted_vals) * p)
            return sorted_vals[min(idx, len(sorted_vals) - 1)]
        
        snapshot = DailyMoodSnapshot(
            date=snapshot_date,
            playlist_count=playlist_count,
            track_count=len(all_features),
            valence_mean=mean(valences),
            energy_mean=mean(energies),
            danceability_mean=mean(danceabilities),
            tempo_mean=mean(tempos),
            valence_p25=percentile(valences, 0.25),
            valence_p75=percentile(valences, 0.75),
            energy_p25=percentile(energies, 0.25),
            energy_p75=percentile(energies, 0.75),
        )
        
        logger.info(
            f"Daily snapshot: {snapshot.track_count} tracks, "
            f"valence={snapshot.valence_mean:.3f}, energy={snapshot.energy_mean:.3f}"
        )
        
        return snapshot
    
    def save_snapshot_to_db(
        self,
        db_path: str | Path,
        snapshot_date: Optional[str] = None,
        playlists: Optional[Dict[str, str]] = None,
    ) -> DailyMoodSnapshot:
        """
        Fetch and save daily snapshot to database.
        
        Args:
            db_path: Path to SQLite database
            snapshot_date: Date string (defaults to today)
            playlists: Dict of name -> playlist_id
            
        Returns:
            DailyMoodSnapshot
        """
        playlists = playlists or CANADA_PLAYLISTS
        snapshot_date = snapshot_date or date.today().isoformat()
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        all_tracks: List[Tuple[TrackInfo, str, str]] = []  # (track, playlist_id, playlist_name)
        all_features: List[AudioFeatures] = []
        
        # Fetch from each playlist
        for name, playlist_id in playlists.items():
            try:
                playlist_name, tracks = self.fetch_playlist_tracks(playlist_id)
                
                for track in tracks:
                    all_tracks.append((track, playlist_id, playlist_name))
                
                # Fetch audio features
                track_ids = [t.track_id for t in tracks]
                features = self.fetch_audio_features(track_ids)
                all_features.extend(features)
                
            except Exception as e:
                logger.warning(f"Failed to fetch playlist {name}: {e}")
        
        # Save playlist snapshots
        for track, playlist_id, playlist_name in all_tracks:
            cursor.execute("""
                INSERT OR REPLACE INTO mood_spotify_playlist_snapshot
                (snapshot_date, playlist_id, playlist_name, track_id, artist, track_name, rank)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (
                snapshot_date,
                playlist_id,
                playlist_name,
                track.track_id,
                track.artist,
                track.track_name,
                track.rank,
            ))
        
        # Save audio features (dedupe by track_id)
        for af in all_features:
            cursor.execute("""
                INSERT OR REPLACE INTO mood_spotify_audio_features
                (track_id, valence, energy, danceability, tempo, speechiness,
                 acousticness, instrumentalness, liveness, duration_ms, time_signature, key, mode)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                af.track_id,
                af.valence,
                af.energy,
                af.danceability,
                af.tempo,
                af.speechiness,
                af.acousticness,
                af.instrumentalness,
                af.liveness,
                af.duration_ms,
                af.time_signature,
                af.key,
                af.mode,
            ))
        
        conn.commit()
        conn.close()
        
        # Calculate aggregated snapshot
        if not all_features:
            return DailyMoodSnapshot(
                date=snapshot_date,
                playlist_count=len(playlists),
                track_count=0,
                valence_mean=0.5,
                energy_mean=0.5,
                danceability_mean=0.5,
                tempo_mean=120.0,
            )
        
        valences = [f.valence for f in all_features]
        energies = [f.energy for f in all_features]
        danceabilities = [f.danceability for f in all_features]
        tempos = [f.tempo for f in all_features]
        
        def mean(vals: List[float]) -> float:
            return sum(vals) / len(vals) if vals else 0.0
        
        return DailyMoodSnapshot(
            date=snapshot_date,
            playlist_count=len(playlists),
            track_count=len(all_features),
            valence_mean=mean(valences),
            energy_mean=mean(energies),
            danceability_mean=mean(danceabilities),
            tempo_mean=mean(tempos),
        )


def check_spotify_credentials() -> bool:
    """Check if Spotify credentials are configured."""
    client_id = os.getenv("SPOTIFY_CLIENT_ID")
    client_secret = os.getenv("SPOTIFY_CLIENT_SECRET")
    return bool(client_id and client_secret)


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if not check_spotify_credentials():
        print("ERROR: Spotify credentials not configured.")
        print("Set SPOTIFY_CLIENT_ID and SPOTIFY_CLIENT_SECRET environment variables.")
        sys.exit(1)
    
    client = SpotifyMoodClient()
    
    # Test fetching playlists
    print("\nFetching Canada playlists...")
    snapshot = client.fetch_daily_snapshot()
    
    print(f"\nDaily Mood Snapshot:")
    print(f"  Date: {snapshot.date}")
    print(f"  Playlists: {snapshot.playlist_count}")
    print(f"  Tracks: {snapshot.track_count}")
    print(f"  Valence (musical positiveness): {snapshot.valence_mean:.3f}")
    print(f"  Energy: {snapshot.energy_mean:.3f}")
    print(f"  Danceability: {snapshot.danceability_mean:.3f}")
    print(f"  Tempo: {snapshot.tempo_mean:.1f} BPM")
