"""
Anomaly Registry for flagging periods that distort normal forecasting patterns.

This module provides a flexible system for tracking known disruptions that affect
sales patterns in ways that would poison matched-conditions forecasting if not
accounted for. Examples include:

- Supply disruptions (BCLDB strikes, LP shortages)
- Demand shocks (COVID lockdowns, extreme weather events)
- Competitive shifts (nearby store opens/closes)
- Operational issues (POS outages, renovations)
- Regulatory changes (new product category launches)
- Data quality issues (import gaps, duplicates)

The registry allows operators to:
1. Flag historical periods that shouldn't be trusted at full weight
2. Specify which categories/SKUs are affected
3. Set severity (0.0 = ignore completely, 1.0 = trust fully)
4. Add notes for future reference

When building forecasts, anomalous periods are down-weighted in similarity
matching so they don't dominate predictions for normal operating conditions.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Callable, Optional, Protocol

import pandas as pd

logger = logging.getLogger(__name__)


class ConnectionProvider(Protocol):
    """Protocol for database connection providers."""
    
    def __call__(self) -> sqlite3.Connection:
        """Return a database connection."""
        ...


class AnomalyType(str, Enum):
    """Categories of anomalies that can affect forecast accuracy."""
    
    SUPPLY_DISRUPTION = "supply_disruption"
    """External supply chain issues (strikes, LP problems, shipping delays)."""
    
    DEMAND_SHOCK = "demand_shock"
    """Unusual demand patterns (COVID, extreme weather, emergency events)."""
    
    COMPETITIVE_SHIFT = "competitive_shift"
    """Market changes (new competitor, competitor closure)."""
    
    OPERATIONAL = "operational"
    """Internal issues (POS outage, renovation, reduced hours)."""
    
    REGULATORY = "regulatory"
    """Rule changes (new product categories, packaging requirements)."""
    
    DATA_QUALITY = "data_quality"
    """Technical issues (import gaps, duplicates, system migration)."""
    
    PROMOTIONAL = "promotional"
    """Major sales events that create artificial spikes (420, Black Friday)."""
    
    OTHER = "other"
    """Catch-all for anomalies that don't fit other categories."""


@dataclass
class Anomaly:
    """Represents a known anomalous period that affects forecast accuracy.
    
    Attributes:
        id: Database primary key (None for unsaved records).
        anomaly_type: Category of disruption.
        name: Human-readable identifier (e.g., "BCLDB Strike 2025").
        start_date: First day of the anomaly (inclusive).
        end_date: Last day of the anomaly (inclusive), or None if ongoing.
        severity: How much to trust data from this period (0.0-1.0).
            - 0.0 = Ignore completely (data is useless)
            - 0.3 = Low trust (major disruption)
            - 0.5 = Medium trust (noticeable but partial impact)
            - 0.7 = High trust (minor disruption)
            - 1.0 = Full trust (normal operations)
        affected_categories: List of category paths affected, or None for all.
        affected_skus: List of SKU IDs affected, or None for all in categories.
        supply_channel: Which supply channel was affected ('bcldb', 'direct', 'both', None).
        notes: Free-form description of the anomaly and its impact.
        created_at: When this record was created.
    """
    
    id: Optional[int] = None
    anomaly_type: AnomalyType = AnomalyType.OTHER
    name: str = ""
    start_date: date = field(default_factory=date.today)
    end_date: Optional[date] = None
    severity: float = 0.5
    affected_categories: Optional[list[str]] = None
    affected_skus: Optional[list[str]] = None
    supply_channel: Optional[str] = None
    notes: Optional[str] = None
    created_at: Optional[datetime] = None
    
    def __post_init__(self):
        """Validate and normalize fields."""
        # Ensure severity is in valid range
        self.severity = max(0.0, min(1.0, self.severity))
        
        # Convert string dates if needed
        if isinstance(self.start_date, str):
            self.start_date = date.fromisoformat(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = date.fromisoformat(self.end_date)
        
        # Normalize anomaly_type
        if isinstance(self.anomaly_type, str):
            self.anomaly_type = AnomalyType(self.anomaly_type)
    
    def is_active_on(self, check_date: date) -> bool:
        """Check if this anomaly was active on a given date."""
        if check_date < self.start_date:
            return False
        if self.end_date is not None and check_date > self.end_date:
            return False
        return True
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "id": self.id,
            "anomaly_type": self.anomaly_type.value,
            "name": self.name,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "severity": self.severity,
            "affected_categories": self.affected_categories,
            "affected_skus": self.affected_skus,
            "supply_channel": self.supply_channel,
            "notes": self.notes,
            "created_at": self.created_at.isoformat() if self.created_at else None,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Anomaly":
        """Create from dictionary."""
        return cls(
            id=data.get("id"),
            anomaly_type=AnomalyType(data.get("anomaly_type", "other")),
            name=data.get("name", ""),
            start_date=date.fromisoformat(data["start_date"]) if data.get("start_date") else date.today(),
            end_date=date.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            severity=float(data.get("severity", 0.5)),
            affected_categories=data.get("affected_categories"),
            affected_skus=data.get("affected_skus"),
            supply_channel=data.get("supply_channel"),
            notes=data.get("notes"),
            created_at=datetime.fromisoformat(data["created_at"]) if data.get("created_at") else None,
        )


# =============================================================================
# Seed Data: Example anomalies to demonstrate the system
# =============================================================================

SEED_ANOMALIES: list[Anomaly] = [
    Anomaly(
        anomaly_type=AnomalyType.SUPPLY_DISRUPTION,
        name="BCLDB Strike 2025",
        start_date=date(2025, 9, 1),
        end_date=date(2025, 10, 26),
        severity=0.3,
        affected_categories=None,  # All categories affected
        supply_channel="bcldb",
        notes=(
            "BCGEU strike shut down BCLDB warehouse operations for ~8 weeks. "
            "Direct-delivery LPs saw artificial demand spikes as retailers scrambled "
            "for alternative supply. BCLDB-dependent SKUs experienced stockouts. "
            "Sales patterns during this period are not representative of normal operations."
        ),
    ),
]


# =============================================================================
# Database Operations
# =============================================================================

def init_anomaly_table(conn: sqlite3.Connection) -> None:
    """Create the anomaly_registry table if it doesn't exist.
    
    This is safe to call multiple times (idempotent).
    """
    sql = """
    CREATE TABLE IF NOT EXISTS anomaly_registry (
        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
        anomaly_type        TEXT NOT NULL,
        name                TEXT NOT NULL,
        start_date          DATE NOT NULL,
        end_date            DATE,
        severity            REAL NOT NULL DEFAULT 0.5,
        affected_categories TEXT,
        affected_skus       TEXT,
        supply_channel      TEXT,
        notes               TEXT,
        created_at          TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        
        UNIQUE(name, start_date)
    );
    
    CREATE INDEX IF NOT EXISTS idx_anomaly_dates 
    ON anomaly_registry(start_date, end_date);
    
    CREATE INDEX IF NOT EXISTS idx_anomaly_type
    ON anomaly_registry(anomaly_type);
    """
    
    conn.executescript(sql)
    conn.commit()
    logger.debug("Anomaly registry table initialized")


def seed_anomalies(conn: sqlite3.Connection, force: bool = False) -> int:
    """Insert seed anomalies if they don't already exist.
    
    Args:
        conn: Database connection.
        force: If True, update existing records with seed values.
    
    Returns:
        Number of records inserted or updated.
    """
    count = 0
    cur = conn.cursor()
    for anomaly in SEED_ANOMALIES:
        # Check if already exists
        cur.execute(
            "SELECT id FROM anomaly_registry WHERE name = ? AND start_date = ?",
            (anomaly.name, anomaly.start_date.isoformat()),
        )
        existing = cur.fetchone()
        
        if existing is None:
            # Insert new
            cur.execute(
                """
                INSERT INTO anomaly_registry 
                (anomaly_type, name, start_date, end_date, severity, 
                 affected_categories, affected_skus, supply_channel, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    anomaly.anomaly_type.value,
                    anomaly.name,
                    anomaly.start_date.isoformat(),
                    anomaly.end_date.isoformat() if anomaly.end_date else None,
                    anomaly.severity,
                    json.dumps(anomaly.affected_categories) if anomaly.affected_categories else None,
                    json.dumps(anomaly.affected_skus) if anomaly.affected_skus else None,
                    anomaly.supply_channel,
                    anomaly.notes,
                ),
            )
            count += 1
            logger.info(f"Seeded anomaly: {anomaly.name}")
        elif force:
            # Update existing
            cur.execute(
                """
                UPDATE anomaly_registry SET
                    anomaly_type = ?,
                    end_date = ?,
                    severity = ?,
                    affected_categories = ?,
                    affected_skus = ?,
                    supply_channel = ?,
                    notes = ?
                WHERE id = ?
                """,
                (
                    anomaly.anomaly_type.value,
                    anomaly.end_date.isoformat() if anomaly.end_date else None,
                    anomaly.severity,
                    json.dumps(anomaly.affected_categories) if anomaly.affected_categories else None,
                    json.dumps(anomaly.affected_skus) if anomaly.affected_skus else None,
                    anomaly.supply_channel,
                    anomaly.notes,
                    existing[0],
                ),
            )
            count += 1
            logger.info(f"Updated anomaly: {anomaly.name}")
    
    conn.commit()
    return count


def create_anomaly(anomaly: Anomaly, conn: sqlite3.Connection) -> Anomaly:
    """Insert a new anomaly into the registry.
    
    Returns the anomaly with its assigned ID.
    """
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO anomaly_registry 
        (anomaly_type, name, start_date, end_date, severity, 
         affected_categories, affected_skus, supply_channel, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            anomaly.anomaly_type.value,
            anomaly.name,
            anomaly.start_date.isoformat(),
            anomaly.end_date.isoformat() if anomaly.end_date else None,
            anomaly.severity,
            json.dumps(anomaly.affected_categories) if anomaly.affected_categories else None,
            json.dumps(anomaly.affected_skus) if anomaly.affected_skus else None,
            anomaly.supply_channel,
            anomaly.notes,
        ),
    )
    conn.commit()
    anomaly.id = cur.lastrowid
    logger.info(f"Created anomaly: {anomaly.name} (id={anomaly.id})")
    return anomaly


def update_anomaly(anomaly: Anomaly, conn: sqlite3.Connection) -> Anomaly:
    """Update an existing anomaly in the registry.
    
    The anomaly must have a valid ID.
    """
    if anomaly.id is None:
        raise ValueError("Cannot update anomaly without ID")
    
    cur = conn.cursor()
    cur.execute(
        """
        UPDATE anomaly_registry SET
            anomaly_type = ?,
            name = ?,
            start_date = ?,
            end_date = ?,
            severity = ?,
            affected_categories = ?,
            affected_skus = ?,
            supply_channel = ?,
            notes = ?
        WHERE id = ?
        """,
        (
            anomaly.anomaly_type.value,
            anomaly.name,
            anomaly.start_date.isoformat(),
            anomaly.end_date.isoformat() if anomaly.end_date else None,
            anomaly.severity,
            json.dumps(anomaly.affected_categories) if anomaly.affected_categories else None,
            json.dumps(anomaly.affected_skus) if anomaly.affected_skus else None,
            anomaly.supply_channel,
            anomaly.notes,
            anomaly.id,
        ),
    )
    conn.commit()
    logger.info(f"Updated anomaly: {anomaly.name} (id={anomaly.id})")
    return anomaly


def delete_anomaly(anomaly_id: int, conn: sqlite3.Connection) -> bool:
    """Delete an anomaly from the registry.
    
    Returns True if a record was deleted.
    """
    cur = conn.cursor()
    cur.execute("DELETE FROM anomaly_registry WHERE id = ?", (anomaly_id,))
    conn.commit()
    deleted = cur.rowcount > 0
    if deleted:
        logger.info(f"Deleted anomaly id={anomaly_id}")
    return deleted


def get_anomaly(anomaly_id: int, conn: sqlite3.Connection) -> Optional[Anomaly]:
    """Fetch a single anomaly by ID."""
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, anomaly_type, name, start_date, end_date, severity,
               affected_categories, affected_skus, supply_channel, notes, created_at
        FROM anomaly_registry
        WHERE id = ?
        """,
        (anomaly_id,),
    )
    row = cur.fetchone()
    if row is None:
        return None
    return _row_to_anomaly(row)


def list_anomalies(
    conn: sqlite3.Connection,
    include_expired: bool = True,
    anomaly_type: Optional[AnomalyType] = None,
) -> list[Anomaly]:
    """List all anomalies in the registry.
    
    Args:
        conn: Database connection.
        include_expired: If False, only return anomalies that are ongoing or future.
        anomaly_type: Filter by anomaly type.
    
    Returns:
        List of Anomaly objects ordered by start_date descending.
    """
    where_clauses = []
    params: list[Any] = []
    
    if not include_expired:
        where_clauses.append("(end_date IS NULL OR end_date >= ?)")
        params.append(date.today().isoformat())
    
    if anomaly_type is not None:
        where_clauses.append("anomaly_type = ?")
        params.append(anomaly_type.value)
    
    where_sql = ""
    if where_clauses:
        where_sql = "WHERE " + " AND ".join(where_clauses)
    
    cur = conn.cursor()
    cur.execute(
        f"""
        SELECT id, anomaly_type, name, start_date, end_date, severity,
               affected_categories, affected_skus, supply_channel, notes, created_at
        FROM anomaly_registry
        {where_sql}
        ORDER BY start_date DESC
        """,
        params,
    )
    
    return [_row_to_anomaly(row) for row in cur.fetchall()]


def get_anomalies_for_date_range(
    start_date: date,
    end_date: date,
    conn: sqlite3.Connection,
) -> list[Anomaly]:
    """Get all anomalies that overlap with the given date range.
    
    An anomaly overlaps if:
    - Its start_date <= end_date AND
    - Its end_date is NULL OR end_date >= start_date
    """
    cur = conn.cursor()
    cur.execute(
        """
        SELECT id, anomaly_type, name, start_date, end_date, severity,
               affected_categories, affected_skus, supply_channel, notes, created_at
        FROM anomaly_registry
        WHERE start_date <= ? 
          AND (end_date IS NULL OR end_date >= ?)
        ORDER BY start_date
        """,
        (end_date.isoformat(), start_date.isoformat()),
    )
    
    return [_row_to_anomaly(row) for row in cur.fetchall()]


def _row_to_anomaly(row) -> Anomaly:
    """Convert a database row to an Anomaly object."""
    return Anomaly(
        id=row[0],
        anomaly_type=AnomalyType(row[1]),
        name=row[2],
        start_date=date.fromisoformat(row[3]),
        end_date=date.fromisoformat(row[4]) if row[4] else None,
        severity=row[5],
        affected_categories=json.loads(row[6]) if row[6] else None,
        affected_skus=json.loads(row[7]) if row[7] else None,
        supply_channel=row[8],
        notes=row[9],
        created_at=datetime.fromisoformat(row[10]) if row[10] else None,
    )


# =============================================================================
# DataFrame Integration
# =============================================================================

def add_anomaly_flags_to_df(
    df: pd.DataFrame,
    conn: sqlite3.Connection,
    date_col: str = "date",
) -> pd.DataFrame:
    """Add anomaly flags to a conditions DataFrame.
    
    This function looks up all anomalies that overlap with the date range in
    the DataFrame and adds the following columns:
    
    - is_anomaly_period: bool - True if any anomaly is active on this date
    - anomaly_severity: float - Minimum severity across active anomalies (lower = less trust)
    - anomaly_names: str - Comma-separated list of active anomaly names
    - anomaly_types: str - Comma-separated list of active anomaly types
    - supply_channel_affected: str or None - Supply channel affected by anomaly
    
    Args:
        df: DataFrame with a date column.
        conn: Database connection.
        date_col: Name of the date column.
    
    Returns:
        DataFrame with anomaly columns added.
    """
    if df.empty:
        df = df.copy()
        df["is_anomaly_period"] = False
        df["anomaly_severity"] = 1.0
        df["anomaly_names"] = ""
        df["anomaly_types"] = ""
        df["supply_channel_affected"] = None
        return df
    
    # Ensure we have proper date objects
    dates = pd.to_datetime(df[date_col])
    min_date = dates.min().date() if hasattr(dates.min(), 'date') else dates.min()
    max_date = dates.max().date() if hasattr(dates.max(), 'date') else dates.max()
    
    # Handle date objects that are already date type
    if isinstance(min_date, datetime):
        min_date = min_date.date()
    if isinstance(max_date, datetime):
        max_date = max_date.date()
    
    # Fetch anomalies for this range
    anomalies = get_anomalies_for_date_range(min_date, max_date, conn)
    
    if not anomalies:
        df = df.copy()
        df["is_anomaly_period"] = False
        df["anomaly_severity"] = 1.0
        df["anomaly_names"] = ""
        df["anomaly_types"] = ""
        df["supply_channel_affected"] = None
        return df
    
    # Build lookup for each date
    def get_anomaly_info(row_date):
        # Normalize date
        if isinstance(row_date, str):
            row_date = date.fromisoformat(row_date)
        elif isinstance(row_date, datetime):
            row_date = row_date.date()
        elif hasattr(row_date, 'date'):
            row_date = row_date.date()
        
        # Find active anomalies
        active = [a for a in anomalies if a.is_active_on(row_date)]
        
        if not active:
            return False, 1.0, "", "", None
        
        # Aggregate anomaly info
        min_severity = min(a.severity for a in active)
        names = ", ".join(a.name for a in active)
        types = ", ".join(a.anomaly_type.value for a in active)
        
        # Get supply channel (first non-None)
        channels = [a.supply_channel for a in active if a.supply_channel]
        channel = channels[0] if channels else None
        
        return True, min_severity, names, types, channel
    
    df = df.copy()
    results = df[date_col].apply(get_anomaly_info)
    
    df["is_anomaly_period"] = results.apply(lambda x: x[0])
    df["anomaly_severity"] = results.apply(lambda x: x[1])
    df["anomaly_names"] = results.apply(lambda x: x[2])
    df["anomaly_types"] = results.apply(lambda x: x[3])
    df["supply_channel_affected"] = results.apply(lambda x: x[4])
    
    return df


def compute_anomaly_adjusted_weight(
    base_weight: float,
    anomaly_severity: float,
    is_anomaly_period: bool,
) -> float:
    """Compute the adjusted weight for a historical observation.
    
    When building forecasts, observations from anomalous periods should
    contribute less to the prediction than normal periods.
    
    Args:
        base_weight: The original weight (e.g., similarity score).
        anomaly_severity: The severity from the anomaly registry (0.0-1.0).
        is_anomaly_period: Whether this observation is in an anomaly period.
    
    Returns:
        Adjusted weight. If not an anomaly period, returns base_weight unchanged.
    """
    if not is_anomaly_period:
        return base_weight
    
    # Scale the weight by severity
    # severity=0.3 means this observation gets 30% of its normal influence
    return base_weight * anomaly_severity
