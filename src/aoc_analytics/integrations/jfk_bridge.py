"""
JFK ↔ AOC Brain Integration

This module provides the bridge between JFK's sales data (cova_sales table)
and the AOC brain's expected sales format.

Architecture:
┌─────────────┐     ┌─────────────────┐     ┌─────────────┐
│  JFK        │     │  AOC Brain      │     │  JFK UI     │
│  Backend    │────▶│  (Analytics)    │────▶│  Insights   │
│  cova_sales │     │  sales view     │     │  Tab        │
└─────────────┘     └─────────────────┘     └─────────────┘

Data Flow:
1. JFK ingests sales from Cova emails → cova_sales table
2. This adapter creates a 'sales' view that maps cova_sales columns to brain schema
3. Brain modules query the 'sales' view for analysis
4. Brain generates insights, predictions, recommendations
5. JFK frontend fetches insights via API endpoints

Column Mapping (cova_sales → brain sales):
    transaction_date    → date
    transaction_time    → time
    transaction_id      → invoice_id
    store_id           → location
    product_sku        → sku
    product_name       → product_name
    category           → category
    quantity           → quantity
    unit_price         → unit_price
    total_price        → subtotal
    net_amount         → gross_profit (approximation)
"""

import os
from pathlib import Path
from typing import Optional

from aoc_analytics.core.db_adapter import get_connection


# SQL to create a view mapping cova_sales to brain's expected schema
SALES_VIEW_SQL = """
CREATE VIEW IF NOT EXISTS sales AS
SELECT 
    id,
    transaction_date as date,
    transaction_time as time,
    transaction_date || ' ' || transaction_time as datetime_local,
    store_id as location,
    product_sku as sku,
    product_name,
    category,
    quantity,
    unit_price,
    total_price as subtotal,
    COALESCE(net_amount, total_price * 0.7) as cost,  -- Estimate if not available
    COALESCE(total_price - net_amount, total_price * 0.3) as gross_profit,
    transaction_id as invoice_id,
    source,
    ingested_at as created_at
FROM cova_sales
WHERE total_price > 0;  -- Exclude returns/voids for now
"""

# SQL to check if the view exists
CHECK_VIEW_SQL = """
SELECT name FROM sqlite_master 
WHERE type='view' AND name='sales';
"""


def setup_sales_view(db_path: str) -> bool:
    """
    Create the sales view in JFK's database that maps cova_sales to brain schema.
    
    Args:
        db_path: Path to JFK's SQLite database
    
    Returns:
        True if view created/exists, False on error
    """
    try:
        conn = get_connection(db_path)
        
        # Check if view already exists
        existing = conn.execute(CHECK_VIEW_SQL).fetchone()
        if existing:
            print("✓ Sales view already exists")
            return True
        
        # Create the view
        conn.execute(SALES_VIEW_SQL)
        conn.commit()
        print("✓ Created sales view mapping cova_sales → brain schema")
        
        # Verify
        count = conn.execute("SELECT COUNT(*) FROM sales").fetchone()[0]
        print(f"  → {count:,} sales records available to brain")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"✗ Error setting up sales view: {e}")
        return False


def get_jfk_db_path() -> str:
    """Get the path to JFK's database."""
    # Check common locations
    candidates = [
        Path(os.environ.get("JFK_DB_PATH", "")),
        Path.home() / "Projects" / "JFK" / "backend" / "instance" / "cannabis_retail.db",
        Path.cwd().parent / "JFK" / "backend" / "instance" / "cannabis_retail.db",
    ]
    
    for path in candidates:
        if path.exists():
            return str(path)
    
    raise FileNotFoundError("Could not find JFK database. Set JFK_DB_PATH environment variable.")


def verify_brain_compatibility(db_path: str) -> dict:
    """
    Verify that the database has data compatible with brain modules.
    
    Returns dict with:
        - is_compatible: bool
        - total_sales: int
        - date_range: (min, max)
        - locations: list
        - categories: list
        - issues: list of any problems
    """
    result = {
        "is_compatible": False,
        "total_sales": 0,
        "date_range": (None, None),
        "locations": [],
        "categories": [],
        "issues": []
    }
    
    try:
        conn = get_connection(db_path)
        
        # Check if sales view/table exists
        tables = conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table', 'view') AND name='sales'"
        ).fetchone()
        
        if not tables:
            result["issues"].append("No 'sales' table or view found")
            return result
        
        # Get basic stats
        stats = conn.execute("""
            SELECT 
                COUNT(*) as total,
                MIN(date) as min_date,
                MAX(date) as max_date,
                COUNT(DISTINCT location) as num_locations,
                COUNT(DISTINCT category) as num_categories
            FROM sales
        """).fetchone()
        
        result["total_sales"] = stats[0]
        result["date_range"] = (stats[1], stats[2])
        
        # Get locations
        locations = conn.execute("SELECT DISTINCT location FROM sales WHERE location IS NOT NULL").fetchall()
        result["locations"] = [r[0] for r in locations]
        
        # Get categories
        categories = conn.execute("SELECT DISTINCT category FROM sales WHERE category IS NOT NULL LIMIT 20").fetchall()
        result["categories"] = [r[0] for r in categories]
        
        # Check for minimum data requirements
        if result["total_sales"] < 100:
            result["issues"].append(f"Only {result['total_sales']} sales records - need at least 100 for meaningful analysis")
        
        if result["total_sales"] > 0:
            # Check for required columns
            required_cols = ['date', 'quantity', 'subtotal']
            for col in required_cols:
                try:
                    conn.execute(f"SELECT {col} FROM sales LIMIT 1")
                except:
                    result["issues"].append(f"Missing required column: {col}")
        
        conn.close()
        
        result["is_compatible"] = len(result["issues"]) == 0
        return result
        
    except Exception as e:
        result["issues"].append(f"Database error: {e}")
        return result


if __name__ == "__main__":
    print("=" * 60)
    print("JFK ↔ AOC Brain Integration Setup")
    print("=" * 60)
    
    try:
        db_path = get_jfk_db_path()
        print(f"\nDatabase: {db_path}")
        
        # Set up the view
        setup_sales_view(db_path)
        
        # Verify compatibility
        print("\nVerifying brain compatibility...")
        compat = verify_brain_compatibility(db_path)
        
        print(f"\n  Total sales: {compat['total_sales']:,}")
        print(f"  Date range: {compat['date_range'][0]} to {compat['date_range'][1]}")
        print(f"  Locations: {', '.join(compat['locations'][:5])}")
        print(f"  Categories: {len(compat['categories'])} found")
        
        if compat["is_compatible"]:
            print("\n✅ Database is compatible with AOC brain!")
        else:
            print("\n⚠️  Compatibility issues:")
            for issue in compat["issues"]:
                print(f"    - {issue}")
                
    except FileNotFoundError as e:
        print(f"\n❌ {e}")
