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
from typing import Iterable, Any
import sqlite3


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
    conn: sqlite3.Connection,
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
        conn: SQLite connection
        location: Store location identifier
        start_date: Start of range (YYYY-MM-DD)
        end_date: End of range (YYYY-MM-DD)
        timezone: Timezone for date calculations
    
    Yields:
        IndexRow records with payday scores
    """
    cursor = conn.execute(
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
    )
    events = cursor.fetchall()
    if not events:
        return []

    scores: dict[tuple[str, int, str], float] = {}
    for event in events:
        event_date = datetime.fromisoformat(f"{event['event_date']}T00:00:00")
        for offset in range(-1, PAYDAY_WINDOW_DAYS):
            current = event_date + timedelta(days=offset)
            date_key = current.date().isoformat()
            if date_key < start_date or date_key > end_date:
                continue
            decay = max(0.0, 1.0 - (abs(offset) * DECAY_PER_DAY))
            key = (date_key, 0, location)
            scores[key] = scores.get(key, 0.0) + float(event["importance"]) * decay

    rows: list[IndexRow] = []
    for (date_key, hour, loc), value in scores.items():
        rows.append(
            IndexRow(
                date=date_key,
                hour=hour,
                location=loc,
                payday=min(value, 1.0),
                metadata={"payday_raw": value},
            )
        )
    return rows
