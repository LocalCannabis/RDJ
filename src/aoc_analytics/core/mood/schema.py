"""
Mood Data Schema

Database table definitions for the mood signal pipeline.
Tables are designed to store both raw data (for debugging/reprocessing)
and derived features (for consumption by AOC).
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# Table definitions
MOOD_TABLES = {
    "mood_spotify_playlist_snapshot": """
        CREATE TABLE IF NOT EXISTS mood_spotify_playlist_snapshot (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            snapshot_date TEXT NOT NULL,
            playlist_id TEXT NOT NULL,
            playlist_name TEXT NOT NULL,
            track_id TEXT NOT NULL,
            artist TEXT,
            track_name TEXT,
            rank INTEGER,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(snapshot_date, playlist_id, track_id)
        )
    """,
    
    "mood_spotify_audio_features": """
        CREATE TABLE IF NOT EXISTS mood_spotify_audio_features (
            track_id TEXT PRIMARY KEY,
            valence REAL,
            energy REAL,
            danceability REAL,
            tempo REAL,
            speechiness REAL,
            acousticness REAL,
            instrumentalness REAL,
            liveness REAL,
            duration_ms INTEGER,
            time_signature INTEGER,
            key INTEGER,
            mode INTEGER,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """,
    
    "mood_google_terms_raw": """
        CREATE TABLE IF NOT EXISTS mood_google_terms_raw (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            geo TEXT NOT NULL,
            term TEXT NOT NULL,
            value_0_100 INTEGER,
            window_start TEXT,
            window_end TEXT,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            source TEXT DEFAULT 'pytrends',
            UNIQUE(date, geo, term)
        )
    """,
    
    "mood_google_buckets_daily": """
        CREATE TABLE IF NOT EXISTS mood_google_buckets_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            geo TEXT NOT NULL,
            stress_score REAL,
            cozy_score REAL,
            party_score REAL,
            money_pressure_score REAL,
            cannabis_interest_score REAL,
            term_coverage REAL,
            fetched_at TEXT DEFAULT CURRENT_TIMESTAMP,
            UNIQUE(date, geo)
        )
    """,
    
    "mood_features_daily": """
        CREATE TABLE IF NOT EXISTS mood_features_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            geo TEXT NOT NULL,
            
            -- Raw Spotify aggregates
            spotify_valence_mean REAL,
            spotify_energy_mean REAL,
            spotify_danceability_mean REAL,
            spotify_tempo_mean REAL,
            spotify_track_count INTEGER,
            
            -- Z-scored features (rolling 30-day baseline)
            spotify_valence_z REAL,
            spotify_energy_z REAL,
            google_stress_z REAL,
            google_cozy_z REAL,
            google_party_z REAL,
            google_money_z REAL,
            google_cannabis_z REAL,
            
            -- Derived Local Vibe indices [-1, 1]
            local_vibe_score REAL,
            local_vibe_anxiety REAL,
            local_vibe_cozy REAL,
            local_vibe_party REAL,
            
            -- Quality & metadata
            data_quality_score REAL,
            spotify_quality REAL,
            google_quality REAL,
            notes TEXT,
            computed_at TEXT DEFAULT CURRENT_TIMESTAMP,
            
            UNIQUE(date, geo)
        )
    """,
    
    # Intraday samples (3x/day) - raw samples before daily aggregation
    "mood_samples_intraday": """
        CREATE TABLE IF NOT EXISTS mood_samples_intraday (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT NOT NULL,
            date TEXT NOT NULL,
            hour INTEGER NOT NULL,
            geo TEXT NOT NULL,
            
            -- Raw mood values (0-1)
            vibe_score REAL,
            anxiety REAL,
            cozy REAL,
            party REAL,
            
            -- Spotify metadata
            spotify_energy REAL,
            spotify_popularity REAL,
            spotify_sample_count INTEGER,
            
            -- Google trends (if available)
            google_stress REAL,
            google_cozy REAL,
            google_party REAL,
            google_available INTEGER DEFAULT 0,
            
            -- Quality
            quality_score REAL,
            providers TEXT,
            
            UNIQUE(timestamp, geo)
        )
    """,
}

# Indexes for performance
MOOD_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_spotify_snapshot_date ON mood_spotify_playlist_snapshot(snapshot_date)",
    "CREATE INDEX IF NOT EXISTS idx_spotify_snapshot_playlist ON mood_spotify_playlist_snapshot(playlist_id)",
    "CREATE INDEX IF NOT EXISTS idx_google_terms_date ON mood_google_terms_raw(date)",
    "CREATE INDEX IF NOT EXISTS idx_google_terms_geo ON mood_google_terms_raw(geo, date)",
    "CREATE INDEX IF NOT EXISTS idx_google_buckets_date ON mood_google_buckets_daily(date)",
    "CREATE INDEX IF NOT EXISTS idx_mood_features_date ON mood_features_daily(date)",
    "CREATE INDEX IF NOT EXISTS idx_mood_features_geo_date ON mood_features_daily(geo, date)",
    "CREATE INDEX IF NOT EXISTS idx_mood_intraday_date ON mood_samples_intraday(date, geo)",
    "CREATE INDEX IF NOT EXISTS idx_mood_intraday_hour ON mood_samples_intraday(date, hour, geo)",
]


def init_mood_tables(db_path: str | Path) -> None:
    """
    Initialize all mood-related tables in the database.
    
    Args:
        db_path: Path to the SQLite database file.
    """
    db_path = Path(db_path)
    logger.info(f"Initializing mood tables in {db_path}")
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        # Create tables
        for table_name, create_sql in MOOD_TABLES.items():
            logger.debug(f"Creating table: {table_name}")
            cursor.execute(create_sql)
        
        # Create indexes
        for index_sql in MOOD_INDEXES:
            cursor.execute(index_sql)
        
        conn.commit()
        logger.info(f"Mood tables initialized successfully ({len(MOOD_TABLES)} tables)")
        
    except Exception as e:
        logger.error(f"Error initializing mood tables: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def get_mood_table_stats(db_path: str | Path) -> dict:
    """
    Get row counts for all mood tables.
    
    Args:
        db_path: Path to the SQLite database file.
        
    Returns:
        Dictionary of table_name -> row_count
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    stats = {}
    for table_name in MOOD_TABLES.keys():
        try:
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            stats[table_name] = cursor.fetchone()[0]
        except sqlite3.OperationalError:
            stats[table_name] = None  # Table doesn't exist
    
    conn.close()
    return stats


def get_latest_mood_date(db_path: str | Path, geo: str = "CA") -> Optional[str]:
    """
    Get the most recent date with mood features.
    
    Args:
        db_path: Path to the SQLite database file.
        geo: Geographic region to check.
        
    Returns:
        Date string (YYYY-MM-DD) or None if no data.
    """
    db_path = Path(db_path)
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    try:
        cursor.execute(
            "SELECT MAX(date) FROM mood_features_daily WHERE geo = ?",
            (geo,)
        )
        result = cursor.fetchone()
        return result[0] if result else None
    except sqlite3.OperationalError:
        return None
    finally:
        conn.close()


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if len(sys.argv) > 1:
        db_path = sys.argv[1]
    else:
        db_path = "aoc_sales.db"
    
    init_mood_tables(db_path)
    stats = get_mood_table_stats(db_path)
    print("\nMood table stats:")
    for table, count in stats.items():
        print(f"  {table}: {count} rows")
