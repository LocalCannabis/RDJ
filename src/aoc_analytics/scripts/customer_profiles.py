#!/usr/bin/env python3
"""
Regular Customer Profile Detection (Without PII)

Identifies distinct "customer archetypes" based on transaction patterns,
not personal data. These profiles can be used for:

1. Forecasting: Model regulars as predictable baseline
2. Inventory: Know which SKUs depend on specific habits
3. Marketing: Understand your customer segments without tracking them

Archetypes detected:
- DAILY_RITUAL: Same product(s), multiple times per day
- WEEKLY_STOCK_UP: Larger orders, predictable weekly pattern
- PAYDAY_BUYER: Activity spikes around 1st/15th
- TIME_LOCKED: Consistent purchase time (e.g., "lunch break buyer")
- CATEGORY_LOYALIST: Sticks to one category exclusively
- DEAL_SEEKER: Activity correlated with promotions/sales
"""

import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from collections import defaultdict
from typing import List, Dict, Tuple, Optional
import numpy as np
from pathlib import Path


@dataclass
class CustomerArchetype:
    """A detected customer behavior pattern."""
    
    archetype: str  # DAILY_RITUAL, WEEKLY_STOCK_UP, etc.
    description: str
    
    # Pattern strength (0-1)
    confidence: float
    
    # Associated SKUs/categories
    primary_skus: List[str]
    primary_categories: List[str]
    
    # Temporal patterns
    typical_day_of_week: Optional[int] = None  # 0=Mon, 6=Sun
    typical_hour: Optional[int] = None
    typical_quantity: int = 1
    
    # Frequency
    estimated_visits_per_month: float = 0.0
    estimated_monthly_spend: float = 0.0
    
    # Transaction fingerprint
    avg_basket_size: float = 1.0
    same_day_repeat_rate: float = 0.0
    
    def __str__(self):
        dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
        day_str = dow_names[self.typical_day_of_week] if self.typical_day_of_week is not None else "Any"
        hour_str = f"{self.typical_hour}:00" if self.typical_hour is not None else "Any"
        
        return (
            f"\n{'='*60}\n"
            f"👤 {self.archetype} ({self.confidence:.0%} confidence)\n"
            f"{'='*60}\n"
            f"   {self.description}\n\n"
            f"   📦 Products: {', '.join(self.primary_skus[:3])}\n"
            f"   🏷️  Categories: {', '.join(self.primary_categories[:3])}\n"
            f"   📅 Typical day: {day_str} @ {hour_str}\n"
            f"   🛒 Typical qty: {self.typical_quantity}\n"
            f"   💰 ~${self.estimated_monthly_spend:.0f}/month, {self.estimated_visits_per_month:.1f} visits\n"
        )


@dataclass
class RegularProfile:
    """A cluster of transactions that appear to be one regular customer."""
    
    profile_id: str
    archetype: CustomerArchetype
    
    # Transaction fingerprint
    transaction_dates: List[str]
    transaction_count: int
    total_quantity: int
    total_spend: float
    
    # Predictability score (how consistent is this pattern?)
    predictability: float  # 0-1
    
    # Time since last activity
    days_since_last: int
    
    # Is this customer still active?
    is_active: bool
    
    def __str__(self):
        status = "🟢 Active" if self.is_active else "🔴 Inactive"
        return (
            f"\n{'-'*50}\n"
            f"Profile {self.profile_id}: {self.archetype.archetype}\n"
            f"   {self.transaction_count} transactions, ${self.total_spend:.0f} total\n"
            f"   Predictability: {self.predictability:.0%}\n"
            f"   Last seen: {self.days_since_last} days ago ({status})\n"
        )


class CustomerProfiler:
    """Detect and cluster regular customer patterns."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            candidates = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent.parent / "aoc_sales.db",
                Path.home() / "Projects" / "aoc-analytics" / "aoc_sales.db",
            ]
            for p in candidates:
                if p.exists():
                    db_path = str(p)
                    break
            else:
                raise FileNotFoundError("Could not find aoc_sales.db")
        self.db_path = str(db_path)
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn
    
    def detect_daily_rituals(self, min_days: int = 30) -> List[CustomerArchetype]:
        """
        Detect DAILY_RITUAL patterns: same product, multiple times per day.
        
        This is your "Vortex Afghan Black guy" pattern.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find SKUs with significant same-day repeat purchases
        cursor.execute("""
            WITH daily_purchases AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    date,
                    COUNT(*) as purchases_that_day,
                    SUM(quantity) as qty_that_day,
                    SUM(subtotal) as spend_that_day,
                    GROUP_CONCAT(time) as times
                FROM sales
                WHERE date >= date('now', '-180 days')
                GROUP BY sku, date
                HAVING purchases_that_day > 1
            )
            SELECT 
                sku,
                product_name,
                category,
                COUNT(*) as multi_purchase_days,
                AVG(purchases_that_day) as avg_purchases_per_day,
                AVG(qty_that_day) as avg_qty_per_day,
                SUM(spend_that_day) as total_spend
            FROM daily_purchases
            GROUP BY sku
            HAVING multi_purchase_days >= ?
            ORDER BY multi_purchase_days DESC
        """, (min_days,))
        
        rituals = []
        for row in cursor.fetchall():
            # Get typical timing
            cursor.execute("""
                SELECT 
                    CAST(strftime('%H', time) AS INTEGER) as hour,
                    COUNT(*) as cnt
                FROM sales
                WHERE sku = ?
                GROUP BY hour
                ORDER BY cnt DESC
                LIMIT 1
            """, (row["sku"],))
            
            hour_row = cursor.fetchone()
            typical_hour = hour_row["hour"] if hour_row else None
            
            confidence = min(row["multi_purchase_days"] / 90, 1.0)  # 90+ days = 100%
            
            rituals.append(CustomerArchetype(
                archetype="DAILY_RITUAL",
                description=f"Regular buying {row['product_name'][:30]} multiple times per day",
                confidence=confidence,
                primary_skus=[row["sku"]],
                primary_categories=[row["category"]] if row["category"] else [],
                typical_hour=typical_hour,
                typical_quantity=int(row["avg_qty_per_day"]),
                estimated_visits_per_month=row["multi_purchase_days"] / 6 * row["avg_purchases_per_day"],
                estimated_monthly_spend=row["total_spend"] / 6,
                same_day_repeat_rate=1.0,  # By definition
            ))
        
        conn.close()
        return rituals
    
    def detect_weekly_stockup(self, min_weeks: int = 8) -> List[CustomerArchetype]:
        """
        Detect WEEKLY_STOCK_UP patterns: consistent weekly purchases.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find SKUs with consistent weekly patterns
        cursor.execute("""
            WITH weekly_purchases AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    strftime('%Y-%W', date) as year_week,
                    CAST(strftime('%w', date) AS INTEGER) as day_of_week,
                    SUM(quantity) as qty,
                    SUM(subtotal) as spend
                FROM sales
                WHERE date >= date('now', '-180 days')
                GROUP BY sku, year_week
            ),
            sku_weekly_stats AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    COUNT(DISTINCT year_week) as weeks_active,
                    AVG(qty) as avg_weekly_qty,
                    AVG(spend) as avg_weekly_spend,
                    -- Most common day of week
                    (SELECT day_of_week FROM weekly_purchases w2 
                     WHERE w2.sku = weekly_purchases.sku 
                     GROUP BY day_of_week ORDER BY COUNT(*) DESC LIMIT 1) as typical_dow
                FROM weekly_purchases
                GROUP BY sku
                HAVING weeks_active >= ?
            )
            SELECT * FROM sku_weekly_stats
            WHERE avg_weekly_qty >= 3  -- Meaningful stock-up quantity
            ORDER BY weeks_active DESC
            LIMIT 50
        """, (min_weeks,))
        
        stockups = []
        for row in cursor.fetchall():
            confidence = min(row["weeks_active"] / 20, 1.0)
            
            stockups.append(CustomerArchetype(
                archetype="WEEKLY_STOCK_UP",
                description=f"Weekly buyer of {row['product_name'][:30]}",
                confidence=confidence,
                primary_skus=[row["sku"]],
                primary_categories=[row["category"]] if row["category"] else [],
                typical_day_of_week=row["typical_dow"],
                typical_quantity=int(row["avg_weekly_qty"]),
                estimated_visits_per_month=4.3,  # Weeks per month
                estimated_monthly_spend=row["avg_weekly_spend"] * 4.3,
            ))
        
        conn.close()
        return stockups
    
    def detect_time_locked(self) -> List[CustomerArchetype]:
        """
        Detect TIME_LOCKED patterns: highly consistent purchase times.
        
        E.g., "lunch break buyer" who always comes at 12:30.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find SKUs with very concentrated time patterns
        cursor.execute("""
            WITH hourly_patterns AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    CAST(strftime('%H', time) AS INTEGER) as hour,
                    COUNT(*) as cnt
                FROM sales
                WHERE date >= date('now', '-90 days')
                  AND time IS NOT NULL
                GROUP BY sku, hour
            ),
            sku_totals AS (
                SELECT sku, SUM(cnt) as total_cnt
                FROM hourly_patterns
                GROUP BY sku
                HAVING total_cnt >= 30
            ),
            peak_hours AS (
                SELECT 
                    hp.sku,
                    hp.product_name,
                    hp.category,
                    hp.hour,
                    hp.cnt,
                    st.total_cnt,
                    CAST(hp.cnt AS FLOAT) / st.total_cnt as hour_concentration
                FROM hourly_patterns hp
                JOIN sku_totals st ON hp.sku = st.sku
                WHERE hp.cnt = (
                    SELECT MAX(cnt) FROM hourly_patterns hp2 WHERE hp2.sku = hp.sku
                )
            )
            SELECT * FROM peak_hours
            WHERE hour_concentration >= 0.4  -- 40%+ of sales in one hour
            ORDER BY hour_concentration DESC
            LIMIT 30
        """)
        
        time_locked = []
        for row in cursor.fetchall():
            time_locked.append(CustomerArchetype(
                archetype="TIME_LOCKED",
                description=f"Always buys {row['product_name'][:25]} around {row['hour']}:00",
                confidence=row["hour_concentration"],
                primary_skus=[row["sku"]],
                primary_categories=[row["category"]] if row["category"] else [],
                typical_hour=row["hour"],
                estimated_visits_per_month=row["total_cnt"] / 3,
            ))
        
        conn.close()
        return time_locked
    
    def detect_category_loyalists(self) -> List[CustomerArchetype]:
        """
        Detect CATEGORY_LOYALIST patterns: customers who only buy from one category.
        
        These show up as high within-category concentration for specific SKUs.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find categories with a few dominant SKUs (suggesting loyalists)
        cursor.execute("""
            WITH category_sku_stats AS (
                SELECT 
                    category,
                    sku,
                    product_name,
                    SUM(quantity) as qty,
                    SUM(subtotal) as spend,
                    COUNT(DISTINCT date) as active_days
                FROM sales
                WHERE date >= date('now', '-90 days')
                  AND category IS NOT NULL
                GROUP BY category, sku
            ),
            category_totals AS (
                SELECT category, SUM(qty) as total_qty
                FROM category_sku_stats
                GROUP BY category
            ),
            sku_share AS (
                SELECT 
                    css.category,
                    css.sku,
                    css.product_name,
                    css.qty,
                    css.spend,
                    css.active_days,
                    ct.total_qty,
                    CAST(css.qty AS FLOAT) / ct.total_qty as category_share
                FROM category_sku_stats css
                JOIN category_totals ct ON css.category = ct.category
                WHERE ct.total_qty >= 100  -- Meaningful category volume
            )
            SELECT * FROM sku_share
            WHERE category_share >= 0.5  -- Single SKU is 50%+ of category
              AND active_days >= 30      -- Consistent activity
            ORDER BY category_share DESC
        """)
        
        loyalists = []
        for row in cursor.fetchall():
            loyalists.append(CustomerArchetype(
                archetype="CATEGORY_LOYALIST",
                description=f"Dominates {row['category']} category ({row['category_share']:.0%} share)",
                confidence=row["category_share"],
                primary_skus=[row["sku"]],
                primary_categories=[row["category"]],
                estimated_visits_per_month=row["active_days"] / 3,
                estimated_monthly_spend=row["spend"] / 3,
            ))
        
        conn.close()
        return loyalists
    
    def detect_payday_buyers(self) -> List[CustomerArchetype]:
        """
        Detect PAYDAY_BUYER patterns: activity spikes on 1st/15th.
        """
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Find SKUs with payday concentration
        cursor.execute("""
            WITH daily_sales AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    date,
                    CAST(strftime('%d', date) AS INTEGER) as day_of_month,
                    SUM(quantity) as qty
                FROM sales
                WHERE date >= date('now', '-180 days')
                GROUP BY sku, date
            ),
            payday_analysis AS (
                SELECT 
                    sku,
                    product_name,
                    category,
                    SUM(qty) as total_qty,
                    SUM(CASE WHEN day_of_month IN (1, 2, 15, 16) THEN qty ELSE 0 END) as payday_qty,
                    COUNT(DISTINCT date) as active_days
                FROM daily_sales
                GROUP BY sku
                HAVING active_days >= 30
                  AND total_qty >= 50
            )
            SELECT 
                *,
                CAST(payday_qty AS FLOAT) / total_qty as payday_concentration
            FROM payday_analysis
            WHERE payday_concentration >= 0.25  -- 25%+ on payday (4 days = 13% expected)
            ORDER BY payday_concentration DESC
            LIMIT 20
        """)
        
        payday_buyers = []
        for row in cursor.fetchall():
            # Normalize: 13% is baseline (4/30 days), so 26% is 2x concentration
            concentration_ratio = row["payday_concentration"] / 0.133
            confidence = min((concentration_ratio - 1) / 2, 1.0)  # 3x = 100%
            
            if confidence > 0:
                payday_buyers.append(CustomerArchetype(
                    archetype="PAYDAY_BUYER",
                    description=f"Buys {row['product_name'][:30]} mainly on paydays",
                    confidence=confidence,
                    primary_skus=[row["sku"]],
                    primary_categories=[row["category"]] if row["category"] else [],
                    estimated_visits_per_month=2,  # Twice monthly
                    estimated_monthly_spend=row["payday_qty"] / 6 * 2,
                ))
        
        conn.close()
        return payday_buyers
    
    def get_all_archetypes(self) -> Dict[str, List[CustomerArchetype]]:
        """Detect all customer archetypes."""
        return {
            "DAILY_RITUAL": self.detect_daily_rituals(),
            "WEEKLY_STOCK_UP": self.detect_weekly_stockup(),
            "TIME_LOCKED": self.detect_time_locked(),
            "CATEGORY_LOYALIST": self.detect_category_loyalists(),
            "PAYDAY_BUYER": self.detect_payday_buyers(),
        }
    
    def summarize_regular_impact(self) -> Dict[str, any]:
        """Summarize the impact of regular customers on the business."""
        
        archetypes = self.get_all_archetypes()
        
        total_monthly_spend = 0
        total_monthly_visits = 0
        sku_coverage = set()
        category_coverage = set()
        
        for archetype_type, profiles in archetypes.items():
            for profile in profiles:
                total_monthly_spend += profile.estimated_monthly_spend
                total_monthly_visits += profile.estimated_visits_per_month
                sku_coverage.update(profile.primary_skus)
                category_coverage.update(profile.primary_categories)
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                SUM(subtotal) / 6 as monthly_revenue,
                COUNT(*) / 6 as monthly_transactions
            FROM sales
            WHERE date >= date('now', '-180 days')
        """)
        
        totals = cursor.fetchone()
        conn.close()
        
        total_monthly_revenue = totals["monthly_revenue"] or 1
        total_monthly_transactions = totals["monthly_transactions"] or 1
        
        return {
            "regular_monthly_spend": total_monthly_spend,
            "regular_monthly_visits": total_monthly_visits,
            "regular_revenue_share": total_monthly_spend / total_monthly_revenue,
            "regular_transaction_share": total_monthly_visits / total_monthly_transactions,
            "skus_with_regulars": len(sku_coverage),
            "categories_with_regulars": len(category_coverage),
            "archetype_counts": {k: len(v) for k, v in archetypes.items()},
        }


def main():
    """Run customer profile analysis."""
    
    profiler = CustomerProfiler()
    
    print("=" * 70)
    print("👥 REGULAR CUSTOMER PROFILE ANALYSIS")
    print("   Detecting behavioral patterns without PII")
    print("=" * 70)
    
    # Detect all archetypes
    archetypes = profiler.get_all_archetypes()
    
    for archetype_type, profiles in archetypes.items():
        if profiles:
            print(f"\n\n{'#'*70}")
            print(f"# {archetype_type} ({len(profiles)} detected)")
            print(f"{'#'*70}")
            
            for profile in profiles[:5]:  # Top 5 of each type
                print(profile)
    
    # Summary
    print("\n\n" + "=" * 70)
    print("📊 REGULAR CUSTOMER IMPACT SUMMARY")
    print("=" * 70)
    
    summary = profiler.summarize_regular_impact()
    
    print(f"\n   Estimated regular customer revenue: ${summary['regular_monthly_spend']:,.0f}/month")
    print(f"   Share of total revenue: {summary['regular_revenue_share']:.1%}")
    print(f"   Estimated regular visits: {summary['regular_monthly_visits']:.0f}/month")
    print(f"   Share of total transactions: {summary['regular_transaction_share']:.1%}")
    print(f"\n   SKUs with detected regulars: {summary['skus_with_regulars']}")
    print(f"   Categories with detected regulars: {summary['categories_with_regulars']}")
    
    print("\n   Archetype breakdown:")
    for archetype, count in summary["archetype_counts"].items():
        print(f"      {archetype}: {count} patterns")


if __name__ == "__main__":
    main()
