"""
Payday Index Computation

Calculates payday proximity scores based on calendar events.
Payday effects decay over a window of days.

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
All Rights Reserved. Proprietary and Confidential.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable, Any, Union
import sqlite3

from ..db_adapter import DBAdapter, wrap_connection


@dataclass
class IndexRow:
    """Single payday index record."""
    date: str
    hour: int
    location: str
    payday: float
    metadata: dict[str, Any] | None


# Configuration
PAYDAY_WINDOW_DAYS = 4  # Days around payday with elevated spending
DECAY_PER_DAY = 0.15    # Score decay per day from payday


def build_payday_index(
    conn: Union[sqlite3.Connection, DBAdapter, Any],
    *,
    location: str,
    start_date: str,
    end_date: str,
    timezone: str = "America/Vancouver",
) -> Iterable[IndexRow]:
    """
    Build payday proximity index for a date range.
    
    Paydays create a halo effect on surrounding days with
    decaying influence.
    
    Args:
        conn: Database connection (SQLite, PostgreSQL, or DBAdapter)
        location: Store location identifier
        start_date: Start of range (YYYY-MM-DD)
        end_date: End of range (YYYY-MM-DD)
        timezone: Timezone for date calculations
    
    Yields:
        IndexRow records with payday scores
    """
    # Wrap in adapter if not already
    db = conn if isinstance(conn, DBAdapter) else wrap_connection(conn)
    
    rows = db.execute(
        """
        SELECT
            date(start_ts) AS event_date,
            location,
            importance
        FROM calendar_events
        WHERE event_type = 'payday'
          AND location = ?
          AND date(start_ts) BETWEEN ? AND ?
        ORDER BY start_ts
        """,
        (location, start_date, end_date),
    ).fetchall()
    
    if not rows:
        return []

    scores: dict[tuple[str, int, str], float] = {}
    for row in rows:
        event_date = datetime.fromisoformat(f"{row[0]}T00:00:00")
        importance = row[2] or 1.0
        for offset in range(-1, PAYDAY_WINDOW_DAYS):
            current = event_date + timedelta(days=offset)
            date_key = current.date().isoformat()
            if date_key < start_date or date_key > end_date:
                continue
            decay = max(0.0, 1.0 - (abs(offset) * DECAY_PER_DAY))
            key = (date_key, 0, location)
            scores[key] = scores.get(key, 0.0) + float(importance) * decay

    result_rows: list[IndexRow] = []
    for (date_key, hour, loc), value in scores.items():
        result_rows.append(
            IndexRow(
                date=date_key,
                hour=hour,
                location=loc,
                payday=min(value, 1.0),
                metadata={"payday_raw": value},
            )
        )
    return result_rows
