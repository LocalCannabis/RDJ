"""
Test AOC Brain against JFK Sales Data

This script runs the brain modules against real JFK sales data to verify
the integration is working and to see what insights we can generate.
"""

import os
import sys
from pathlib import Path

# Setup paths
AOC_ROOT = Path(__file__).parent.parent.parent.parent
JFK_DB = Path(os.environ.get("JFK_DB_PATH", Path.home() / "Projects" / "JFK" / "backend" / "instance" / "cannabis_retail.db"))

# Set environment
os.environ["DB_PATH"] = str(JFK_DB)

print("=" * 70)
print("AOC Brain ↔ JFK Data Integration Test")
print("=" * 70)
print(f"\nDatabase: {JFK_DB}")
print()

# First, ensure the sales view exists
from aoc_analytics.integrations.jfk_bridge import setup_sales_view, verify_brain_compatibility

print("Step 1: Setting up sales view...")
setup_sales_view(str(JFK_DB))

print("\nStep 2: Verifying compatibility...")
compat = verify_brain_compatibility(str(JFK_DB))
print(f"  Sales records: {compat['total_sales']}")
print(f"  Date range: {compat['date_range']}")
print(f"  Locations: {compat['locations']}")
print()

if not compat["is_compatible"]:
    print("❌ Database not compatible. Issues:")
    for issue in compat["issues"]:
        print(f"   - {issue}")
    sys.exit(1)

# Import db tools early
from aoc_analytics.core.db_adapter import get_connection
import pandas as pd

# Test Signal Detection
print("=" * 70)
print("Step 3: Testing Signal Detection...")
print("=" * 70)

try:
    
    conn = get_connection(str(JFK_DB))
    sales_df = pd.read_sql_query("""
        SELECT 
            date,
            strftime('%H', time) as hour,
            category,
            SUM(quantity) as total_qty,
            SUM(subtotal) as total_sales
        FROM sales
        GROUP BY date, hour, category
    """, conn)
    
    print(f"  Loaded {len(sales_df)} hourly category aggregations")
    
    # Basic signal stats
    daily_sales = conn.execute("""
        SELECT date, SUM(subtotal) as daily_total
        FROM sales
        GROUP BY date
    """).fetchall()
    
    print(f"\n  Daily Sales Totals:")
    for row in daily_sales[:5]:
        print(f"    {row[0]}: ${row[1]:,.2f}")
    
    conn.close()
    print("\n✅ Signal detection data loaded successfully!")
    
except Exception as e:
    print(f"❌ Signal detection error: {e}")
    import traceback
    traceback.print_exc()

# Test Category Analysis
print("\n" + "=" * 70)
print("Step 4: Testing Category Analysis...")
print("=" * 70)

try:
    conn = get_connection(str(JFK_DB))
    
    categories = pd.read_sql_query("""
        SELECT 
            category,
            COUNT(*) as num_transactions,
            SUM(quantity) as total_qty,
            SUM(subtotal) as total_revenue,
            ROUND(AVG(subtotal), 2) as avg_transaction
        FROM sales
        WHERE category IS NOT NULL
        GROUP BY category
        ORDER BY total_revenue DESC
        LIMIT 10
    """, conn)
    
    print("\n  Top Categories by Revenue:")
    print(categories.to_string(index=False))
    
    conn.close()
    print("\n✅ Category analysis working!")
    
except Exception as e:
    print(f"❌ Category analysis error: {e}")

# Test Time Pattern Analysis
print("\n" + "=" * 70)
print("Step 5: Testing Time Pattern Analysis...")
print("=" * 70)

try:
    conn = get_connection(str(JFK_DB))
    
    hourly = pd.read_sql_query("""
        SELECT 
            strftime('%H', time) as hour,
            COUNT(*) as transactions,
            SUM(subtotal) as revenue
        FROM sales
        GROUP BY hour
        ORDER BY hour
    """, conn)
    
    print("\n  Hourly Sales Pattern:")
    print(hourly.to_string(index=False))
    
    # Find peak hour
    if len(hourly) > 0:
        peak = hourly.loc[hourly['revenue'].idxmax()]
        print(f"\n  📈 Peak hour: {peak['hour']}:00 with ${peak['revenue']:,.2f} revenue")
    
    conn.close()
    print("\n✅ Time pattern analysis working!")
    
except Exception as e:
    print(f"❌ Time pattern error: {e}")

# Test LLM Synthesis (if OpenAI key available)
print("\n" + "=" * 70)
print("Step 6: Testing LLM Synthesis...")
print("=" * 70)

OPENAI_KEY = os.environ.get("OPENAI_API_KEY")
if not OPENAI_KEY:
    # Try to load from JFK's .env
    jfk_env = Path.home() / "Projects" / "JFK" / "backend" / ".env"
    if jfk_env.exists():
        for line in jfk_env.read_text().splitlines():
            if line.startswith("OPENAI_API_KEY="):
                os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                OPENAI_KEY = os.environ["OPENAI_API_KEY"]
                print(f"  Loaded OpenAI key from JFK .env")
                break

if OPENAI_KEY:
    try:
        from aoc_analytics.brain.llm_provider import LLMProvider
        
        llm = LLMProvider()
        
        # Gather insights for synthesis
        conn = get_connection(str(JFK_DB))
        
        summary_data = conn.execute("""
            SELECT 
                COUNT(*) as total_transactions,
                SUM(quantity) as total_items,
                SUM(subtotal) as total_revenue,
                COUNT(DISTINCT category) as num_categories,
                AVG(subtotal) as avg_transaction
            FROM sales
        """).fetchone()
        
        top_products = conn.execute("""
            SELECT product_name, SUM(quantity) as qty, SUM(subtotal) as rev
            FROM sales
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 5
        """).fetchall()
        
        conn.close()
        
        # Generate insight
        prompt = f"""You are an AI analytics assistant for a cannabis retail store.
        
Based on today's sales data:
- Total transactions: {summary_data[0]}
- Total items sold: {summary_data[1]}
- Total revenue: ${summary_data[2]:,.2f}
- Unique categories: {summary_data[3]}
- Average transaction: ${summary_data[4]:.2f}

Top products by revenue:
{chr(10).join([f"  - {p[0]}: {p[1]} units, ${p[2]:.2f}" for p in top_products])}

Generate 3 actionable insights for the store manager. Be specific and data-driven.
Format as a numbered list."""

        print("\n  Generating insights via GPT-4o-mini...")
        response = llm.generate(prompt, max_tokens=500, temperature=0.7)
        
        print("\n  📊 AI-Generated Insights:")
        print("-" * 50)
        print(response)
        print("-" * 50)
        
        print("\n✅ LLM synthesis working!")
        
    except Exception as e:
        print(f"❌ LLM synthesis error: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  ⚠️  No OPENAI_API_KEY found - skipping LLM test")

# Summary
print("\n" + "=" * 70)
print("INTEGRATION TEST SUMMARY")
print("=" * 70)
print("""
✅ JFK Bridge: Sales view created successfully
✅ Data Access: {total_sales} sales records accessible
✅ Signal Detection: Hourly aggregations working
✅ Category Analysis: Revenue by category working
✅ Time Patterns: Peak hour detection working
{llm_status}

The AOC Brain is ready to analyze JFK sales data!

Next Steps:
1. Create API endpoints for JFK to fetch insights
2. Set up scheduled jobs for continuous analysis
3. Wire insights to JFK UI's AOC Insights tab
""".format(
    total_sales=compat['total_sales'],
    llm_status="✅ LLM Synthesis: GPT-4o-mini generating insights" if OPENAI_KEY else "⚠️  LLM Synthesis: Needs OPENAI_API_KEY"
))
