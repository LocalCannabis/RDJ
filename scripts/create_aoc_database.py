#!/usr/bin/env python3
"""
Create AOC's standalone database with historical data.

This migrates analytics data OUT of JFK into AOC's own database,
establishing clean separation of concerns:

JFK (Mission Critical):
  - products, inventory, stores, users
  - recent cova_sales (for POS operations)
  
AOC (Analytics Engine):
  - Historical sales (6 years)
  - Weather data
  - Mood features
  - Brain cache
"""

import sqlite3
import os
from pathlib import Path
from datetime import datetime

# Paths
JFK_DB = Path("/Users/localcannabis/Projects/JFK/backend/instance/cannabis_retail.db")
AOC_DB = Path("/Users/localcannabis/Projects/aoc-analytics/aoc_analytics.db")

def create_aoc_schema(conn):
    """Create AOC database schema."""
    cursor = conn.cursor()
    
    # Sales table - matches JFK's sales view structure
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            transaction_id TEXT NOT NULL,
            store_id TEXT NOT NULL,
            date DATE NOT NULL,
            time TEXT NOT NULL,
            datetime_local TEXT,
            product_name TEXT NOT NULL,
            sku TEXT,
            category TEXT,
            subcategory TEXT,
            brand TEXT,
            quantity REAL NOT NULL,
            unit_price REAL,
            subtotal REAL,
            discount REAL,
            tax_amount REAL,
            gross_profit REAL,
            employee_id TEXT,
            customer_type TEXT,
            source TEXT NOT NULL,
            created_at TEXT NOT NULL
        )
    """)
    
    # Weather hourly
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS weather_hourly (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id TEXT NOT NULL,
            date DATE NOT NULL,
            hour INTEGER NOT NULL,
            temperature_c REAL,
            humidity_pct REAL,
            precipitation_mm REAL,
            weather_code INTEGER,
            weather_desc TEXT,
            weather_icon TEXT,
            cloud_cover_pct REAL,
            wind_speed_kmh REAL,
            wind_gust_kmh REAL,
            fetched_at TEXT NOT NULL,
            UNIQUE(store_id, date, hour)
        )
    """)
    
    # Mood features (for future use)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS mood_features_daily (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id TEXT NOT NULL,
            date DATE NOT NULL,
            day_of_week INTEGER,
            is_weekend INTEGER,
            is_holiday INTEGER,
            holiday_name TEXT,
            avg_temp_c REAL,
            precipitation_mm REAL,
            weather_score REAL,
            created_at TEXT NOT NULL,
            UNIQUE(store_id, date)
        )
    """)
    
    # Brain cache for computed insights
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS brain_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            cache_key TEXT UNIQUE NOT NULL,
            store_id TEXT,
            date DATE,
            data_json TEXT NOT NULL,
            created_at TEXT NOT NULL,
            expires_at TEXT
        )
    """)
    
    # Anomaly registry
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS anomaly_registry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            store_id TEXT NOT NULL,
            date DATE NOT NULL,
            anomaly_type TEXT NOT NULL,
            severity REAL,
            description TEXT,
            detected_at TEXT NOT NULL
        )
    """)
    
    # Create indexes
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_store_date ON sales(store_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_sales_category ON sales(category)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_weather_store_date ON weather_hourly(store_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_mood_store_date ON mood_features_daily(store_id, date)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_brain_cache_key ON brain_cache(cache_key)")
    
    conn.commit()
    print("✅ Created AOC schema")


def migrate_sales(jfk_conn, aoc_conn):
    """Migrate sales data from JFK to AOC."""
    print("\n📊 Migrating sales data...")
    
    jfk_cursor = jfk_conn.cursor()
    aoc_cursor = aoc_conn.cursor()
    
    # Check source - try cova_sales first
    jfk_cursor.execute("SELECT COUNT(*) FROM cova_sales")
    count = jfk_cursor.fetchone()[0]
    print(f"   Found {count:,} records in cova_sales")
    
    # Read from cova_sales and transform
    jfk_cursor.execute("""
        SELECT 
            transaction_id,
            store_id,
            transaction_date,
            transaction_time,
            transaction_date || ' ' || transaction_time,
            product_name,
            product_sku,
            category,
            subcategory,
            brand,
            quantity,
            unit_price,
            total_price,
            discount,
            tax_amount,
            net_amount,
            employee_id,
            customer_type,
            source,
            ingested_at
        FROM cova_sales
    """)
    
    batch = []
    batch_size = 10000
    total = 0
    
    for row in jfk_cursor:
        batch.append(row)
        if len(batch) >= batch_size:
            aoc_cursor.executemany("""
                INSERT INTO sales (
                    transaction_id, store_id, date, time, datetime_local,
                    product_name, sku, category, subcategory, brand,
                    quantity, unit_price, subtotal, discount, tax_amount, gross_profit,
                    employee_id, customer_type, source, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            total += len(batch)
            print(f"   Migrated {total:,} records...", end='\r')
            batch = []
    
    if batch:
        aoc_cursor.executemany("""
            INSERT INTO sales (
                transaction_id, store_id, date, time, datetime_local,
                product_name, sku, category, subcategory, brand,
                quantity, unit_price, subtotal, discount, tax_amount, gross_profit,
                employee_id, customer_type, source, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, batch)
        total += len(batch)
    
    aoc_conn.commit()
    print(f"\n   ✅ Migrated {total:,} sales records")
    return total


def migrate_weather(jfk_conn, aoc_conn):
    """Migrate weather data from JFK to AOC."""
    print("\n🌤️  Migrating weather data...")
    
    jfk_cursor = jfk_conn.cursor()
    aoc_cursor = aoc_conn.cursor()
    
    try:
        jfk_cursor.execute("SELECT COUNT(*) FROM weather_hourly")
        count = jfk_cursor.fetchone()[0]
        print(f"   Found {count:,} weather records")
        
        if count == 0:
            print("   ⚠️  No weather data to migrate")
            return 0
        
        # Copy all weather data
        jfk_cursor.execute("SELECT * FROM weather_hourly")
        
        batch = []
        batch_size = 10000
        total = 0
        
        for row in jfk_cursor:
            # Skip the id column (first one)
            batch.append(row[1:])
            if len(batch) >= batch_size:
                aoc_cursor.executemany("""
                    INSERT OR IGNORE INTO weather_hourly (
                        store_id, date, hour, temperature_c, humidity_pct,
                        precipitation_mm, weather_code, weather_desc, weather_icon,
                        cloud_cover_pct, wind_speed_kmh, wind_gust_kmh, fetched_at
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, batch)
                total += len(batch)
                print(f"   Migrated {total:,} records...", end='\r')
                batch = []
        
        if batch:
            aoc_cursor.executemany("""
                INSERT OR IGNORE INTO weather_hourly (
                    store_id, date, hour, temperature_c, humidity_pct,
                    precipitation_mm, weather_code, weather_desc, weather_icon,
                    cloud_cover_pct, wind_speed_kmh, wind_gust_kmh, fetched_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, batch)
            total += len(batch)
        
        aoc_conn.commit()
        print(f"\n   ✅ Migrated {total:,} weather records")
        return total
        
    except sqlite3.OperationalError as e:
        print(f"   ⚠️  Weather table not found: {e}")
        return 0


def main():
    print("=" * 60)
    print("AOC Database Migration")
    print("Separating analytics data from JFK")
    print("=" * 60)
    
    print(f"\n📂 Source (JFK): {JFK_DB}")
    print(f"📂 Target (AOC): {AOC_DB}")
    
    # Backup existing AOC db if it exists
    if AOC_DB.exists():
        backup = AOC_DB.with_suffix('.db.backup')
        print(f"\n⚠️  AOC database exists, backing up to {backup.name}")
        import shutil
        shutil.copy(AOC_DB, backup)
    
    # Connect to both databases
    jfk_conn = sqlite3.connect(JFK_DB)
    aoc_conn = sqlite3.connect(AOC_DB)
    
    # Create AOC schema
    create_aoc_schema(aoc_conn)
    
    # Migrate data
    sales_count = migrate_sales(jfk_conn, aoc_conn)
    weather_count = migrate_weather(jfk_conn, aoc_conn)
    
    # Verify
    print("\n" + "=" * 60)
    print("📊 Migration Summary")
    print("=" * 60)
    
    aoc_cursor = aoc_conn.cursor()
    
    aoc_cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM sales")
    row = aoc_cursor.fetchone()
    print(f"\n✅ Sales: {row[0]:,} records ({row[1]} to {row[2]})")
    
    aoc_cursor.execute("SELECT COUNT(*), MIN(date), MAX(date) FROM weather_hourly")
    row = aoc_cursor.fetchone()
    if row[0]:
        print(f"✅ Weather: {row[0]:,} records ({row[1]} to {row[2]})")
    
    # Show file sizes
    jfk_size = JFK_DB.stat().st_size / (1024 * 1024)
    aoc_size = AOC_DB.stat().st_size / (1024 * 1024)
    print(f"\n📦 JFK DB size: {jfk_size:.1f} MB")
    print(f"📦 AOC DB size: {aoc_size:.1f} MB")
    
    jfk_conn.close()
    aoc_conn.close()
    
    print("\n" + "=" * 60)
    print("✅ Migration complete!")
    print("\nNext steps:")
    print("1. Update AOC to use aoc_analytics.db")
    print("2. Remove analytics tables from JFK (optional)")
    print("3. Test both systems independently")
    print("=" * 60)


if __name__ == "__main__":
    main()
