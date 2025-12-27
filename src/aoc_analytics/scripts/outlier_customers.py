#!/usr/bin/env python3
"""
Outlier Customer Detection (Without Customer Data)

Identifies SKUs/categories whose performance is skewed by probable repeat 
customers based on transaction pattern analysis.

The "Vortex Afghan Black Problem": One customer buying 1-3x/day makes hash 
look like a hero category when it's really one person's habit.

Detection Methods:
1. Transaction Clustering - Same SKU, similar times, similar quantities
2. Concentration Analysis - Top N transactions as % of total volume  
3. Variance Anomalies - Unusually consistent purchase patterns
4. Interval Detection - Regular purchasing intervals (daily, weekly)
"""

import sqlite3
from datetime import datetime, timedelta
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from pathlib import Path


@dataclass
class OutlierSignature:
    """A detected pattern suggesting repeat customer behavior."""
    sku: str
    product_name: str
    category: str
    
    # Detection metrics
    total_transactions: int
    total_quantity: int
    
    # Concentration - do few transactions dominate?
    top_5_pct_of_volume: float  # Top 5 transactions as % of total qty
    top_10_pct_of_volume: float
    
    # Regularity - is there a pattern?
    same_day_transactions: int  # Multiple purchases same day
    avg_daily_frequency: float  # When it sells, how many times/day?
    
    # Consistency - same quantity each time?
    quantity_cv: float  # Coefficient of variation (low = suspicious)
    modal_quantity: int  # Most common quantity
    modal_quantity_pct: float  # How often is it the modal qty?
    
    # Time patterns
    common_hour: int  # Most frequent purchase hour
    hour_concentration: float  # % of sales in that hour
    
    # Verdict
    confidence: float  # 0-1, how confident we are this is outlier-driven
    estimated_regular_customers: int  # Guess at how many regulars
    
    def __str__(self):
        return (
            f"\n{'='*60}\n"
            f"🎯 {self.product_name} ({self.sku})\n"
            f"   Category: {self.category}\n"
            f"{'='*60}\n"
            f"   Total: {self.total_transactions} transactions, {self.total_quantity} units\n"
            f"\n"
            f"   📊 CONCENTRATION:\n"
            f"      Top 5 transactions = {self.top_5_pct_of_volume:.1%} of volume\n"
            f"      Top 10 transactions = {self.top_10_pct_of_volume:.1%} of volume\n"
            f"\n"
            f"   🔄 REGULARITY:\n"
            f"      Same-day repeat purchases: {self.same_day_transactions}\n"
            f"      Avg transactions per active day: {self.avg_daily_frequency:.1f}\n"
            f"\n"
            f"   📏 CONSISTENCY:\n"
            f"      Quantity CV: {self.quantity_cv:.2f} (lower = more suspicious)\n"
            f"      Modal quantity: {self.modal_quantity} ({self.modal_quantity_pct:.0%} of transactions)\n"
            f"\n"
            f"   🕐 TIME PATTERN:\n"
            f"      Peak hour: {self.common_hour}:00 ({self.hour_concentration:.0%} of sales)\n"
            f"\n"
            f"   🚨 VERDICT: {self.confidence:.0%} confidence outlier-driven\n"
            f"      Estimated regular customers: ~{self.estimated_regular_customers}\n"
        )


class OutlierDetector:
    """Detect SKUs skewed by repeat customer behavior."""
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            # Try multiple paths
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
                raise FileNotFoundError(f"Could not find aoc_sales.db in: {candidates}")
        self.db_path = str(db_path)
        
    def get_connection(self):
        return sqlite3.connect(self.db_path)
    
    def analyze_sku(self, sku: str) -> OutlierSignature | None:
        """Analyze a single SKU for outlier customer patterns."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all transactions for this SKU
        cursor.execute("""
            SELECT 
                date,
                time,
                quantity,
                product_name,
                category
            FROM sales
            WHERE sku = ?
            ORDER BY date, time
        """, (sku,))
        
        rows = cursor.fetchall()
        conn.close()
        
        if len(rows) < 10:
            return None  # Not enough data
            
        # Parse data
        transactions = []
        product_name = rows[0][3]
        category = rows[0][4]
        
        for date, time_str, qty, _, _ in rows:
            try:
                hour = int(time_str.split(':')[0]) if time_str else 12
            except:
                hour = 12
            transactions.append({
                'date': date,
                'hour': hour,
                'quantity': qty or 1
            })
        
        quantities = [t['quantity'] for t in transactions]
        total_qty = sum(quantities)
        
        # === CONCENTRATION ANALYSIS ===
        sorted_qtys = sorted(quantities, reverse=True)
        top_5_qty = sum(sorted_qtys[:5])
        top_10_qty = sum(sorted_qtys[:10])
        top_5_pct = top_5_qty / total_qty if total_qty > 0 else 0
        top_10_pct = top_10_qty / total_qty if total_qty > 0 else 0
        
        # === REGULARITY ANALYSIS ===
        # Count transactions per day
        daily_counts = defaultdict(int)
        for t in transactions:
            daily_counts[t['date']] += 1
        
        same_day_transactions = sum(1 for c in daily_counts.values() if c > 1)
        multi_purchase_days = [c for c in daily_counts.values() if c > 1]
        avg_daily_freq = np.mean(multi_purchase_days) if multi_purchase_days else 1.0
        
        # === CONSISTENCY ANALYSIS ===
        qty_std = np.std(quantities)
        qty_mean = np.mean(quantities)
        qty_cv = qty_std / qty_mean if qty_mean > 0 else 0
        
        # Modal quantity
        qty_counts = defaultdict(int)
        for q in quantities:
            qty_counts[q] += 1
        modal_qty = max(qty_counts, key=qty_counts.get)
        modal_pct = qty_counts[modal_qty] / len(quantities)
        
        # === TIME PATTERN ANALYSIS ===
        hour_counts = defaultdict(int)
        for t in transactions:
            hour_counts[t['hour']] += 1
        common_hour = max(hour_counts, key=hour_counts.get)
        hour_concentration = hour_counts[common_hour] / len(transactions)
        
        # === CONFIDENCE SCORING ===
        # Higher score = more likely outlier-driven
        confidence = 0.0
        
        # High concentration is suspicious
        if top_5_pct > 0.3:
            confidence += 0.2
        if top_10_pct > 0.5:
            confidence += 0.15
            
        # Same-day purchases are very suspicious
        same_day_ratio = same_day_transactions / len(daily_counts) if daily_counts else 0
        if same_day_ratio > 0.1:
            confidence += 0.25
        if avg_daily_freq > 1.5:
            confidence += 0.15
            
        # Low quantity variance is suspicious
        if qty_cv < 0.3:
            confidence += 0.15
        if modal_pct > 0.6:
            confidence += 0.1
            
        # Strong time preference is suspicious
        if hour_concentration > 0.4:
            confidence += 0.1
            
        confidence = min(confidence, 1.0)
        
        # Estimate number of regulars
        # Based on same-day pattern frequency
        if same_day_ratio > 0.3:
            estimated_regulars = 1
        elif same_day_ratio > 0.15:
            estimated_regulars = 2
        elif same_day_ratio > 0.05:
            estimated_regulars = 3
        else:
            estimated_regulars = 0
            
        return OutlierSignature(
            sku=sku,
            product_name=product_name,
            category=category,
            total_transactions=len(transactions),
            total_quantity=total_qty,
            top_5_pct_of_volume=top_5_pct,
            top_10_pct_of_volume=top_10_pct,
            same_day_transactions=same_day_transactions,
            avg_daily_frequency=avg_daily_freq,
            quantity_cv=qty_cv,
            modal_quantity=modal_qty,
            modal_quantity_pct=modal_pct,
            common_hour=common_hour,
            hour_concentration=hour_concentration,
            confidence=confidence,
            estimated_regular_customers=estimated_regulars
        )
    
    def find_outlier_skus(self, min_transactions: int = 20, min_confidence: float = 0.4) -> list[OutlierSignature]:
        """Find all SKUs that appear to be outlier-driven."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get SKUs with enough transactions
        cursor.execute("""
            SELECT sku, COUNT(*) as txn_count
            FROM sales
            WHERE sku IS NOT NULL AND sku != ''
            GROUP BY sku
            HAVING txn_count >= ?
            ORDER BY txn_count DESC
        """, (min_transactions,))
        
        skus = cursor.fetchall()
        conn.close()
        
        print(f"Analyzing {len(skus)} SKUs with >= {min_transactions} transactions...")
        
        outliers = []
        for sku, count in skus:
            signature = self.analyze_sku(sku)
            if signature and signature.confidence >= min_confidence:
                outliers.append(signature)
        
        # Sort by confidence
        outliers.sort(key=lambda x: x.confidence, reverse=True)
        return outliers
    
    def analyze_category_skew(self, category: str) -> dict:
        """Analyze how much a category's performance is skewed by outliers."""
        
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Get all SKUs in category
        cursor.execute("""
            SELECT DISTINCT sku 
            FROM sales 
            WHERE category = ?
        """, (category,))
        
        skus = [row[0] for row in cursor.fetchall()]
        
        # Get total category sales
        cursor.execute("""
            SELECT SUM(quantity), SUM(total)
            FROM sales
            WHERE category = ?
        """, (category,))
        
        total_qty, total_revenue = cursor.fetchone()
        conn.close()
        
        # Analyze each SKU
        outlier_qty = 0
        outlier_revenue = 0
        outlier_skus = []
        
        for sku in skus:
            sig = self.analyze_sku(sku)
            if sig and sig.confidence >= 0.5:
                outlier_skus.append(sig)
                # Estimate outlier contribution
                # If 1 regular buying avg 1.5x/day, that's ~45 transactions/month
                estimated_outlier_pct = min(sig.confidence * 0.5, 0.5)  # Conservative
                outlier_qty += sig.total_quantity * estimated_outlier_pct
        
        return {
            'category': category,
            'total_quantity': total_qty,
            'total_revenue': total_revenue,
            'outlier_skus': len(outlier_skus),
            'estimated_outlier_quantity': outlier_qty,
            'estimated_true_quantity': total_qty - outlier_qty,
            'outlier_skew_pct': outlier_qty / total_qty if total_qty else 0,
            'flagged_products': outlier_skus
        }


def find_vortex_afghan_black():
    """Specifically look for the Vortex Afghan Black pattern."""
    
    detector = OutlierDetector()
    conn = detector.get_connection()
    cursor = conn.cursor()
    
    # Search for it
    cursor.execute("""
        SELECT DISTINCT sku, product_name 
        FROM sales 
        WHERE LOWER(product_name) LIKE '%vortex%' 
           OR LOWER(product_name) LIKE '%afghan%'
           OR LOWER(product_name) LIKE '%hash%'
        LIMIT 20
    """)
    
    matches = cursor.fetchall()
    conn.close()
    
    print("Searching for Vortex Afghan Black and similar...")
    print(f"Found {len(matches)} potential matches:\n")
    
    for sku, name in matches:
        sig = detector.analyze_sku(sku)
        if sig:
            print(sig)


def estimate_true_category_performance(category: str, verbose: bool = True) -> dict:
    """
    Estimate category performance after removing outlier customer contribution.
    
    Returns dict with:
    - raw_quantity: Total units sold
    - estimated_outlier_quantity: Units attributed to regular customers
    - true_quantity: Estimated "real" demand
    - skew_percentage: How much the category is overstated
    """
    
    detector = OutlierDetector()
    conn = detector.get_connection()
    cursor = conn.cursor()
    
    # Get all SKUs in category
    cursor.execute("""
        SELECT sku, product_name, SUM(quantity) as qty, COUNT(*) as txns
        FROM sales
        WHERE category = ?
        GROUP BY sku
        ORDER BY qty DESC
    """, (category,))
    
    products = cursor.fetchall()
    conn.close()
    
    total_qty = sum(p[2] for p in products)
    total_txns = sum(p[3] for p in products)
    
    outlier_contribution = 0
    flagged_products = []
    
    for sku, name, qty, txns in products:
        sig = detector.analyze_sku(sku)
        if sig and sig.confidence >= 0.4:
            # Estimate regular customer contribution
            # Based on same-day repeat pattern
            if sig.same_day_transactions > 10:
                # Conservative: assume 60% of same-day-repeats are regulars
                estimated_regular_txns = sig.same_day_transactions * sig.avg_daily_frequency * 0.6
                estimated_regular_qty = estimated_regular_txns * sig.modal_quantity
                outlier_contribution += estimated_regular_qty
                
                flagged_products.append({
                    'sku': sku,
                    'name': name,
                    'total_qty': qty,
                    'estimated_outlier_qty': estimated_regular_qty,
                    'confidence': sig.confidence,
                    'same_day_repeats': sig.same_day_transactions
                })
    
    true_qty = total_qty - outlier_contribution
    skew_pct = outlier_contribution / total_qty if total_qty > 0 else 0
    
    result = {
        'category': category,
        'raw_quantity': total_qty,
        'raw_transactions': total_txns,
        'estimated_outlier_quantity': outlier_contribution,
        'true_quantity': true_qty,
        'skew_percentage': skew_pct,
        'flagged_products': flagged_products
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print(f"📊 TRUE PERFORMANCE: {category}")
        print(f"{'='*70}")
        print(f"\n   Raw volume:        {total_qty:,} units")
        print(f"   Outlier estimate:  {outlier_contribution:,.0f} units ({skew_pct:.1%})")
        print(f"   TRUE volume:       {true_qty:,.0f} units")
        print(f"\n   Flagged products:")
        for fp in flagged_products[:5]:
            print(f"      • {fp['name'][:40]}")
            print(f"        {fp['estimated_outlier_qty']:.0f} units from regulars ({fp['confidence']:.0%} conf)")
    
    return result


def main():
    """Run outlier detection analysis."""
    
    detector = OutlierDetector()
    
    print("=" * 70)
    print("🔍 OUTLIER CUSTOMER DETECTION")
    print("   Finding SKUs skewed by repeat customer behavior")
    print("=" * 70)
    
    # Find all outlier-driven SKUs
    outliers = detector.find_outlier_skus(min_transactions=30, min_confidence=0.5)
    
    print(f"\n📋 Found {len(outliers)} SKUs with suspected regular customers:\n")
    
    for sig in outliers[:20]:  # Top 20
        print(sig)
    
    # Category analysis
    print("\n" + "=" * 70)
    print("📊 CATEGORY SKEW ANALYSIS")
    print("=" * 70)
    
    conn = detector.get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT category FROM sales WHERE category IS NOT NULL")
    categories = [row[0] for row in cursor.fetchall()]
    conn.close()
    
    category_skews = []
    for cat in categories:
        if cat:
            skew = detector.analyze_category_skew(cat)
            if skew['outlier_skew_pct'] > 0.05:  # >5% outlier-driven
                category_skews.append(skew)
    
    category_skews.sort(key=lambda x: x['outlier_skew_pct'], reverse=True)
    
    print("\nCategories with >5% estimated outlier contribution:\n")
    for skew in category_skews[:10]:
        print(f"  {skew['category']}:")
        print(f"    Total volume: {skew['total_quantity']:,} units")
        print(f"    Estimated outlier volume: {skew['estimated_outlier_quantity']:,.0f} units")
        print(f"    Skew: {skew['outlier_skew_pct']:.1%}")
        print(f"    Flagged products: {skew['outlier_skus']}")
        print()


if __name__ == "__main__":
    main()
