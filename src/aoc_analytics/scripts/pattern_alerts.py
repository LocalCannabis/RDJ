"""
Regular Customer Pattern Alerting

Detects when SKUs start showing new regular customer signatures.
Useful for:
- Identifying emerging loyalty patterns
- Catching when "hero" products are becoming regular-driven
- Inventory planning for predictable demand
- Early warning when patterns change

Alert Types:
- NEW_REGULAR: SKU starts showing repeat customer behavior
- REGULAR_LOST: Previously regular SKU loses its pattern
- PATTERN_SHIFT: Regular changes their behavior (timing, quantity, etc.)
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional
from collections import defaultdict
import json


@dataclass
class PatternAlert:
    """An alert about regular customer pattern changes."""
    
    alert_type: str  # NEW_REGULAR, REGULAR_LOST, PATTERN_SHIFT
    severity: str    # INFO, WARNING, ATTENTION
    sku: str
    product_name: str
    category: str
    
    # Pattern details
    old_pattern: Optional[Dict] = None
    new_pattern: Optional[Dict] = None
    
    # Impact
    estimated_daily_qty: float = 0.0
    estimated_monthly_revenue: float = 0.0
    
    # Detection details
    confidence: float = 0.0
    detected_at: str = ""
    days_since_pattern_start: int = 0
    
    def __str__(self):
        severity_emoji = {"INFO": "ℹ️", "WARNING": "⚠️", "ATTENTION": "🚨"}.get(self.severity, "•")
        
        lines = [
            f"\n{severity_emoji} {self.alert_type}: {self.product_name[:40]}",
            f"   SKU: {self.sku} | Category: {self.category}",
            f"   Confidence: {self.confidence:.0%} | Days active: {self.days_since_pattern_start}",
            f"   Estimated impact: {self.estimated_daily_qty:.1f} units/day, ${self.estimated_monthly_revenue:.0f}/month",
        ]
        
        if self.new_pattern:
            lines.append(f"   Pattern: {self.new_pattern}")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            "alert_type": self.alert_type,
            "severity": self.severity,
            "sku": self.sku,
            "product_name": self.product_name,
            "category": self.category,
            "estimated_daily_qty": self.estimated_daily_qty,
            "estimated_monthly_revenue": self.estimated_monthly_revenue,
            "confidence": self.confidence,
            "detected_at": self.detected_at,
            "days_since_pattern_start": self.days_since_pattern_start,
            "old_pattern": self.old_pattern,
            "new_pattern": self.new_pattern,
        }


class PatternAlertStore:
    """Persistent storage for known patterns and alerts."""
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self._init_tables()
    
    def _init_tables(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Store known regular patterns
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS known_regular_patterns (
                sku TEXT PRIMARY KEY,
                product_name TEXT,
                category TEXT,
                pattern_type TEXT,
                confidence REAL,
                first_detected TEXT,
                last_confirmed TEXT,
                pattern_data TEXT,
                is_active INTEGER DEFAULT 1
            )
        """)
        
        # Store alerts
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS pattern_alerts (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                alert_type TEXT,
                severity TEXT,
                sku TEXT,
                product_name TEXT,
                category TEXT,
                confidence REAL,
                estimated_daily_qty REAL,
                estimated_monthly_revenue REAL,
                pattern_data TEXT,
                created_at TEXT DEFAULT CURRENT_TIMESTAMP,
                acknowledged INTEGER DEFAULT 0,
                acknowledged_by TEXT,
                acknowledged_at TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def get_known_patterns(self) -> Dict[str, Dict]:
        """Get all known regular patterns."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("SELECT * FROM known_regular_patterns WHERE is_active = 1")
        rows = cursor.fetchall()
        conn.close()
        
        return {
            row["sku"]: {
                "product_name": row["product_name"],
                "category": row["category"],
                "pattern_type": row["pattern_type"],
                "confidence": row["confidence"],
                "first_detected": row["first_detected"],
                "last_confirmed": row["last_confirmed"],
                "pattern_data": json.loads(row["pattern_data"]) if row["pattern_data"] else {},
            }
            for row in rows
        }
    
    def save_pattern(self, sku: str, product_name: str, category: str, 
                    pattern_type: str, confidence: float, pattern_data: Dict):
        """Save or update a known pattern."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        now = datetime.now().isoformat()
        
        cursor.execute("""
            INSERT INTO known_regular_patterns 
            (sku, product_name, category, pattern_type, confidence, first_detected, last_confirmed, pattern_data, is_active)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(sku) DO UPDATE SET
                confidence = ?,
                last_confirmed = ?,
                pattern_data = ?,
                is_active = 1
        """, (
            sku, product_name, category, pattern_type, confidence, now, now, json.dumps(pattern_data),
            confidence, now, json.dumps(pattern_data)
        ))
        
        conn.commit()
        conn.close()
    
    def mark_pattern_inactive(self, sku: str):
        """Mark a pattern as no longer active."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute("UPDATE known_regular_patterns SET is_active = 0 WHERE sku = ?", (sku,))
        conn.commit()
        conn.close()
    
    def save_alert(self, alert: PatternAlert):
        """Save an alert."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO pattern_alerts 
            (alert_type, severity, sku, product_name, category, confidence,
             estimated_daily_qty, estimated_monthly_revenue, pattern_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            alert.alert_type, alert.severity, alert.sku, alert.product_name, alert.category,
            alert.confidence, alert.estimated_daily_qty, alert.estimated_monthly_revenue,
            json.dumps(alert.new_pattern)
        ))
        
        conn.commit()
        conn.close()
    
    def get_recent_alerts(self, days: int = 7) -> List[PatternAlert]:
        """Get alerts from the last N days."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT * FROM pattern_alerts
            WHERE created_at >= datetime('now', ?)
            ORDER BY created_at DESC
        """, (f'-{days} days',))
        
        alerts = []
        for row in cursor.fetchall():
            alerts.append(PatternAlert(
                alert_type=row["alert_type"],
                severity=row["severity"],
                sku=row["sku"],
                product_name=row["product_name"],
                category=row["category"],
                confidence=row["confidence"],
                estimated_daily_qty=row["estimated_daily_qty"],
                estimated_monthly_revenue=row["estimated_monthly_revenue"],
                new_pattern=json.loads(row["pattern_data"]) if row["pattern_data"] else None,
                detected_at=row["created_at"],
            ))
        
        conn.close()
        return alerts


class PatternMonitor:
    """Monitor for new and changing regular customer patterns."""
    
    def __init__(self, sales_db_path: str = None, alert_db_path: str = None):
        if sales_db_path is None:
            candidates = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent / "aoc_sales.db",
            ]
            for p in candidates:
                if p.exists():
                    sales_db_path = str(p)
                    break
        
        if alert_db_path is None:
            alert_db_path = str(Path(sales_db_path).parent / "pattern_alerts.db")
        
        self.sales_db_path = sales_db_path
        self.alert_store = PatternAlertStore(alert_db_path)
    
    def get_sales_connection(self):
        conn = sqlite3.connect(self.sales_db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def scan_for_new_patterns(self, lookback_days: int = 30, 
                              min_confidence: float = 0.5) -> List[PatternAlert]:
        """
        Scan recent sales for new regular customer patterns.
        
        Compares recent patterns against known patterns to detect:
        - NEW_REGULAR: SKU shows repeat behavior for first time
        - PATTERN_SHIFT: Known regular changes behavior
        """
        
        known_patterns = self.alert_store.get_known_patterns()
        alerts = []
        
        conn = self.get_sales_connection()
        cursor = conn.cursor()
        
        # Find SKUs with same-day repeat patterns in recent period
        cursor.execute("""
            WITH daily_purchases AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    date,
                    COUNT(*) as txns_that_day,
                    SUM(quantity) as qty_that_day,
                    SUM(subtotal) as spend_that_day,
                    AVG(CAST(strftime('%H', time) AS INTEGER)) as avg_hour
                FROM sales
                WHERE date >= date('now', ?)
                GROUP BY sku, date
                HAVING txns_that_day > 1
            )
            SELECT 
                sku,
                product_name,
                category,
                COUNT(*) as repeat_days,
                AVG(txns_that_day) as avg_txns_per_day,
                AVG(qty_that_day) as avg_qty_per_day,
                SUM(spend_that_day) as total_spend,
                AVG(avg_hour) as typical_hour,
                MIN(date) as first_seen,
                MAX(date) as last_seen
            FROM daily_purchases
            GROUP BY sku
            HAVING repeat_days >= 5  -- At least 5 days with repeats
            ORDER BY repeat_days DESC
        """, (f'-{lookback_days} days',))
        
        detected_patterns = cursor.fetchall()
        conn.close()
        
        now = datetime.now()
        
        for row in detected_patterns:
            sku = row["sku"]
            
            # Calculate confidence
            # More repeat days = higher confidence
            confidence = min(1.0, row["repeat_days"] / 20)
            
            if confidence < min_confidence:
                continue
            
            pattern_data = {
                "repeat_days": row["repeat_days"],
                "avg_txns_per_day": round(row["avg_txns_per_day"], 1),
                "avg_qty_per_day": round(row["avg_qty_per_day"], 1),
                "typical_hour": int(row["typical_hour"]) if row["typical_hour"] else None,
                "first_seen": row["first_seen"],
                "last_seen": row["last_seen"],
            }
            
            # Calculate impact
            daily_qty = row["avg_qty_per_day"]
            monthly_revenue = row["total_spend"] / lookback_days * 30
            
            # Is this a new pattern or known?
            if sku not in known_patterns:
                # NEW_REGULAR: First time seeing this pattern
                alert = PatternAlert(
                    alert_type="NEW_REGULAR",
                    severity="INFO" if confidence < 0.7 else "ATTENTION",
                    sku=sku,
                    product_name=row["product_name"],
                    category=row["category"] or "Unknown",
                    confidence=confidence,
                    estimated_daily_qty=daily_qty,
                    estimated_monthly_revenue=monthly_revenue,
                    new_pattern=pattern_data,
                    detected_at=now.isoformat(),
                    days_since_pattern_start=row["repeat_days"],
                )
                alerts.append(alert)
                
                # Save as known pattern
                self.alert_store.save_pattern(
                    sku, row["product_name"], row["category"] or "Unknown",
                    "DAILY_REPEAT", confidence, pattern_data
                )
            else:
                # Check for pattern shift
                old_pattern = known_patterns[sku]["pattern_data"]
                
                # Detect significant changes
                if old_pattern.get("avg_qty_per_day"):
                    qty_change = abs(daily_qty - old_pattern["avg_qty_per_day"]) / old_pattern["avg_qty_per_day"]
                    
                    if qty_change > 0.3:  # >30% change
                        alert = PatternAlert(
                            alert_type="PATTERN_SHIFT",
                            severity="WARNING",
                            sku=sku,
                            product_name=row["product_name"],
                            category=row["category"] or "Unknown",
                            confidence=confidence,
                            estimated_daily_qty=daily_qty,
                            estimated_monthly_revenue=monthly_revenue,
                            old_pattern=old_pattern,
                            new_pattern=pattern_data,
                            detected_at=now.isoformat(),
                        )
                        alerts.append(alert)
                
                # Update the known pattern
                self.alert_store.save_pattern(
                    sku, row["product_name"], row["category"] or "Unknown",
                    "DAILY_REPEAT", confidence, pattern_data
                )
        
        # Check for lost patterns
        for sku, known in known_patterns.items():
            if not any(row["sku"] == sku for row in detected_patterns):
                # Pattern no longer detected
                last_confirmed = datetime.fromisoformat(known["last_confirmed"])
                days_since = (now - last_confirmed).days
                
                if days_since > 14:  # Haven't seen pattern in 2 weeks
                    alert = PatternAlert(
                        alert_type="REGULAR_LOST",
                        severity="INFO",
                        sku=sku,
                        product_name=known["product_name"],
                        category=known["category"],
                        old_pattern=known["pattern_data"],
                        detected_at=now.isoformat(),
                        days_since_pattern_start=days_since,
                    )
                    alerts.append(alert)
                    self.alert_store.mark_pattern_inactive(sku)
        
        # Save new alerts
        for alert in alerts:
            self.alert_store.save_alert(alert)
        
        return alerts
    
    def get_alert_summary(self) -> Dict:
        """Get summary of current pattern state."""
        
        known = self.alert_store.get_known_patterns()
        recent_alerts = self.alert_store.get_recent_alerts(days=7)
        
        return {
            "known_patterns": len(known),
            "recent_alerts": len(recent_alerts),
            "alerts_by_type": {
                "NEW_REGULAR": len([a for a in recent_alerts if a.alert_type == "NEW_REGULAR"]),
                "REGULAR_LOST": len([a for a in recent_alerts if a.alert_type == "REGULAR_LOST"]),
                "PATTERN_SHIFT": len([a for a in recent_alerts if a.alert_type == "PATTERN_SHIFT"]),
            },
            "top_categories": list(set(p["category"] for p in known.values()))[:5],
        }


def main():
    """Run pattern monitoring scan."""
    
    print("=" * 70)
    print("🔔 REGULAR CUSTOMER PATTERN MONITOR")
    print("   Detecting new and changing regular patterns")
    print("=" * 70)
    
    monitor = PatternMonitor()
    
    print("\nScanning last 30 days for regular patterns...")
    alerts = monitor.scan_for_new_patterns(lookback_days=30, min_confidence=0.4)
    
    # Group by type
    new_regulars = [a for a in alerts if a.alert_type == "NEW_REGULAR"]
    lost_regulars = [a for a in alerts if a.alert_type == "REGULAR_LOST"]
    shifts = [a for a in alerts if a.alert_type == "PATTERN_SHIFT"]
    
    print(f"\n📊 SCAN RESULTS")
    print(f"   New regulars detected: {len(new_regulars)}")
    print(f"   Lost regulars: {len(lost_regulars)}")
    print(f"   Pattern shifts: {len(shifts)}")
    
    if new_regulars:
        print(f"\n\n🆕 NEW REGULAR PATTERNS ({len(new_regulars)})")
        print("-" * 60)
        for alert in sorted(new_regulars, key=lambda a: -a.confidence)[:10]:
            print(alert)
    
    if shifts:
        print(f"\n\n⚠️ PATTERN SHIFTS ({len(shifts)})")
        print("-" * 60)
        for alert in shifts[:5]:
            print(alert)
    
    if lost_regulars:
        print(f"\n\n👋 LOST REGULARS ({len(lost_regulars)})")
        print("-" * 60)
        for alert in lost_regulars[:5]:
            print(alert)
    
    # Summary
    summary = monitor.get_alert_summary()
    print(f"\n\n📈 OVERALL STATUS")
    print(f"   Known active patterns: {summary['known_patterns']}")
    print(f"   Alerts this week: {summary['recent_alerts']}")


if __name__ == "__main__":
    main()
