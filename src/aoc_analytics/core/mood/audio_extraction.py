"""
Local Audio Feature Extraction

Extracts mood-relevant audio features from preview MP3s locally,
independent of Spotify's deprecated audio-features endpoint.

Features extracted:
    - Energy proxy: RMS loudness, spectral centroid, onset rate
    - Valence proxy: Major/minor estimation, brightness, harmonic features

Uses librosa as primary library with optional Essentia enhancement.
"""

from __future__ import annotations

import hashlib
import io
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import httpx

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFeatures:
    """Audio features extracted from a preview clip."""
    
    # Energy-related (0-1 normalized)
    rms_mean: float = 0.0  # Root mean square loudness
    rms_std: float = 0.0
    spectral_centroid_mean: float = 0.0  # Brightness
    onset_rate: float = 0.0  # Tempo/activity proxy
    
    # Valence-related
    mode_probability: float = 0.5  # 1.0 = definitely major, 0.0 = definitely minor
    spectral_brightness: float = 0.0
    spectral_flatness: float = 0.0  # Noise-like vs tonal
    
    # Rhythm
    tempo_estimate: float = 120.0
    beat_strength: float = 0.0
    
    # Derived proxies (0-1)
    energy_proxy: float = 0.0
    valence_proxy: float = 0.0
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary."""
        return {
            "rms_mean": self.rms_mean,
            "rms_std": self.rms_std,
            "spectral_centroid_mean": self.spectral_centroid_mean,
            "onset_rate": self.onset_rate,
            "mode_probability": self.mode_probability,
            "spectral_brightness": self.spectral_brightness,
            "spectral_flatness": self.spectral_flatness,
            "tempo_estimate": self.tempo_estimate,
            "beat_strength": self.beat_strength,
            "energy_proxy": self.energy_proxy,
            "valence_proxy": self.valence_proxy,
        }


class AudioFeatureExtractor:
    """
    Extracts mood features from audio files/URLs.
    
    Uses librosa for feature extraction, with optional Essentia
    for more sophisticated analysis.
    """
    
    def __init__(
        self,
        cache_dir: Optional[str | Path] = None,
        use_cache: bool = True,
    ):
        """
        Initialize extractor.
        
        Args:
            cache_dir: Directory to cache downloaded previews.
            use_cache: Whether to cache downloaded files.
        """
        self.cache_dir = Path(cache_dir) if cache_dir else None
        self.use_cache = use_cache
        
        if self.cache_dir and self.use_cache:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._librosa_available = None
        self._essentia_available = None
    
    def is_available(self) -> bool:
        """Check if extraction libraries are available."""
        return self._check_librosa()
    
    def _check_librosa(self) -> bool:
        """Check if librosa is importable."""
        if self._librosa_available is None:
            try:
                import librosa
                self._librosa_available = True
            except ImportError:
                self._librosa_available = False
        return self._librosa_available
    
    def _check_essentia(self) -> bool:
        """Check if essentia is importable."""
        if self._essentia_available is None:
            try:
                import essentia.standard
                self._essentia_available = True
            except ImportError:
                self._essentia_available = False
        return self._essentia_available
    
    def _get_cache_path(self, url: str) -> Optional[Path]:
        """Get cache path for a URL."""
        if not self.cache_dir or not self.use_cache:
            return None
        
        url_hash = hashlib.md5(url.encode()).hexdigest()
        return self.cache_dir / f"preview_{url_hash}.mp3"
    
    def _download_preview(self, url: str) -> bytes:
        """Download preview audio from URL."""
        cache_path = self._get_cache_path(url)
        
        # Check cache first
        if cache_path and cache_path.exists():
            logger.debug(f"Cache hit for {url[:50]}...")
            return cache_path.read_bytes()
        
        # Download
        logger.debug(f"Downloading preview from {url[:50]}...")
        
        with httpx.Client(timeout=30.0) as client:
            response = client.get(url, follow_redirects=True)
            response.raise_for_status()
            audio_data = response.content
        
        # Cache if enabled
        if cache_path:
            cache_path.write_bytes(audio_data)
            logger.debug(f"Cached preview ({len(audio_data)} bytes)")
        
        return audio_data
    
    def extract_from_url(self, url: str) -> Dict[str, Any]:
        """
        Extract features from a preview URL.
        
        Args:
            url: URL to the audio preview (MP3).
            
        Returns:
            Dictionary of extracted features.
        """
        audio_data = self._download_preview(url)
        return self.extract_from_bytes(audio_data)
    
    def extract_from_bytes(self, audio_data: bytes) -> Dict[str, Any]:
        """
        Extract features from audio bytes.
        
        Args:
            audio_data: Raw audio file bytes (MP3).
            
        Returns:
            Dictionary of extracted features.
        """
        if not self._check_librosa():
            raise ImportError(
                "librosa is required for audio feature extraction. "
                "Install with: pip install librosa"
            )
        
        import librosa
        import numpy as np
        
        # Load audio from bytes
        # librosa needs a file-like object or path
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=True) as f:
            f.write(audio_data)
            f.flush()
            
            try:
                y, sr = librosa.load(f.name, sr=22050, mono=True)
            except Exception as e:
                logger.error(f"Failed to load audio: {e}")
                return ExtractedFeatures().to_dict()
        
        return self._extract_features_librosa(y, sr)
    
    def extract_from_file(self, file_path: str | Path) -> Dict[str, Any]:
        """
        Extract features from an audio file.
        
        Args:
            file_path: Path to audio file.
            
        Returns:
            Dictionary of extracted features.
        """
        if not self._check_librosa():
            raise ImportError("librosa required for audio extraction")
        
        import librosa
        
        y, sr = librosa.load(str(file_path), sr=22050, mono=True)
        return self._extract_features_librosa(y, sr)
    
    def _extract_features_librosa(
        self,
        y,  # numpy array
        sr: int,
    ) -> Dict[str, Any]:
        """
        Extract all features using librosa.
        
        Args:
            y: Audio time series
            sr: Sample rate
            
        Returns:
            Dictionary of features
        """
        import librosa
        import numpy as np
        
        features = ExtractedFeatures()
        
        # Skip if audio too short
        if len(y) < sr * 2:  # Less than 2 seconds
            logger.warning("Audio too short for feature extraction")
            return features.to_dict()
        
        try:
            # === Energy features ===
            
            # RMS loudness
            rms = librosa.feature.rms(y=y)[0]
            features.rms_mean = float(np.mean(rms))
            features.rms_std = float(np.std(rms))
            
            # Spectral centroid (brightness/sharpness)
            centroid = librosa.feature.spectral_centroid(y=y, sr=sr)[0]
            # Normalize by Nyquist frequency
            features.spectral_centroid_mean = float(np.mean(centroid) / (sr / 2))
            
            # Onset detection (activity/intensity)
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            onsets = librosa.onset.onset_detect(
                onset_envelope=onset_env,
                sr=sr,
                units='time'
            )
            duration = len(y) / sr
            features.onset_rate = len(onsets) / duration if duration > 0 else 0
            # Normalize: typical range 0-4 onsets/sec -> 0-1
            features.onset_rate = min(1.0, features.onset_rate / 4.0)
            
            # === Valence features ===
            
            # Chroma features for key/mode estimation
            chroma = librosa.feature.chroma_cqt(y=y, sr=sr)
            
            # Simple major/minor estimation
            # Major keys tend to have stronger 3rd scale degree
            # This is a rough approximation
            chroma_mean = np.mean(chroma, axis=1)
            # Normalize
            chroma_mean = chroma_mean / (np.sum(chroma_mean) + 1e-8)
            
            # Find likely root note
            root = np.argmax(chroma_mean)
            
            # Check major 3rd (4 semitones) vs minor 3rd (3 semitones)
            major_third = (root + 4) % 12
            minor_third = (root + 3) % 12
            
            major_strength = chroma_mean[major_third]
            minor_strength = chroma_mean[minor_third]
            
            # Mode probability: higher = more major
            total = major_strength + minor_strength + 1e-8
            features.mode_probability = float(major_strength / total)
            
            # Spectral brightness (high frequency content)
            # Brighter often correlates with happier
            features.spectral_brightness = features.spectral_centroid_mean
            
            # Spectral flatness (noise-like vs tonal)
            flatness = librosa.feature.spectral_flatness(y=y)[0]
            features.spectral_flatness = float(np.mean(flatness))
            
            # === Rhythm features ===
            
            # Tempo
            tempo, beat_frames = librosa.beat.beat_track(y=y, sr=sr)
            # Handle both scalar and array return values
            if hasattr(tempo, '__iter__'):
                tempo = tempo[0] if len(tempo) > 0 else 120.0
            features.tempo_estimate = float(tempo)
            
            # Beat strength
            if len(beat_frames) > 0:
                beat_times = librosa.frames_to_time(beat_frames, sr=sr)
                # Regularity of beats
                if len(beat_times) > 1:
                    intervals = np.diff(beat_times)
                    interval_std = np.std(intervals) if len(intervals) > 0 else 1.0
                    # Lower std = more regular beats = stronger
                    features.beat_strength = float(max(0, 1.0 - interval_std / 0.5))
            
            # === Compute proxy scores ===
            
            # Energy proxy: combination of loudness, onset rate, tempo
            # Higher values = more energetic
            tempo_norm = min(1.0, max(0, (features.tempo_estimate - 60) / 140))  # 60-200 BPM range
            features.energy_proxy = (
                0.35 * min(1.0, features.rms_mean * 5)  # RMS typically 0-0.2
                + 0.35 * features.onset_rate
                + 0.30 * tempo_norm
            )
            
            # Valence proxy: major/minor + brightness + beat regularity
            # Higher = happier
            features.valence_proxy = (
                0.45 * features.mode_probability
                + 0.30 * features.spectral_brightness
                + 0.25 * features.beat_strength
            )
            
            # Clamp to 0-1
            features.energy_proxy = max(0.0, min(1.0, features.energy_proxy))
            features.valence_proxy = max(0.0, min(1.0, features.valence_proxy))
            
        except Exception as e:
            logger.error(f"Feature extraction error: {e}")
        
        return features.to_dict()


def test_extraction():
    """Quick test of audio extraction."""
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    extractor = AudioFeatureExtractor()
    
    if not extractor.is_available():
        print("librosa not available. Install with: pip install librosa")
        sys.exit(1)
    
    print("Audio extraction libraries available!")
    
    # Test with a sample URL if provided
    if len(sys.argv) > 1:
        url = sys.argv[1]
        print(f"\nExtracting features from: {url[:50]}...")
        
        features = extractor.extract_from_url(url)
        
        print("\nExtracted features:")
        for key, value in features.items():
            print(f"  {key}: {value:.3f}" if isinstance(value, float) else f"  {key}: {value}")


if __name__ == "__main__":
    test_extraction()
