#!/usr/bin/env python3
"""
Backfill Sales Data into AOC

Loads historical Cova sales CSV data into AOC's database for hero scoring
and behavioral signal generation.

Usage:
    # From directory of CSVs
    python -m aoc_analytics.scripts.backfill_sales --input-dir ./backfill_data/
    
    # From single combined file
    python -m aoc_analytics.scripts.backfill_sales --input-file ./combined_sales.csv
    
    # With custom column mappings
    python -m aoc_analytics.scripts.backfill_sales \
        --input-file ./data.csv \
        --date-col "Sale Date" \
        --sku-col "Item SKU"
"""

import argparse
import csv
import glob
import logging
import os
import sqlite3
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s"
)
logger = logging.getLogger(__name__)


# Default column name variations to try
DATE_COLUMNS = ["date", "sale_date", "Date", "Sale Date", "transaction_date", "Date (Local)"]
DATETIME_COLUMNS = ["datetime", "Date Time (Local)", "DateTime", "date_time", "timestamp"]
TIME_COLUMNS = ["time", "sale_time", "Time", "Sale Time", "transaction_time"]
SKU_COLUMNS = ["sku", "SKU", "item_sku", "Item SKU", "product_sku", "Product SKU"]
NAME_COLUMNS = ["product_name", "Product Name", "name", "Name", "item_name", "Item Name", "Product"]
QTY_COLUMNS = ["quantity", "Quantity", "qty", "Qty", "qty_sold", "Qty Sold", "units"]
PRICE_COLUMNS = ["unit_price", "Unit Price", "price", "Price", "Sold At Price", "sold_at_price"]
SUBTOTAL_COLUMNS = ["subtotal", "Subtotal", "line_total", "Line Total", "total", "Total", "amount"]
COST_COLUMNS = ["cost", "Cost", "unit_cost", "Unit Cost", "wholesale_cost", "Total Cost"]
CATEGORY_COLUMNS = ["category", "Category", "product_category", "Product Category", "Classification", "Category Path"]
INVOICE_COLUMNS = ["invoice_id", "Invoice", "invoice", "receipt", "Receipt", "transaction_id", "Invoice #"]
LOCATION_COLUMNS = ["location", "Location", "store", "Store", "site", "Site"]
GROSS_PROFIT_COLUMNS = ["gross_profit", "Gross Profit", "profit", "Profit", "margin"]


def find_column(headers: List[str], candidates: List[str]) -> Optional[str]:
    """Find matching column from list of candidates."""
    headers_lower = {h.lower().strip(): h for h in headers}
    for candidate in candidates:
        if candidate.lower() in headers_lower:
            return headers_lower[candidate.lower()]
    return None


def detect_columns(headers: List[str], overrides: Dict[str, str]) -> Dict[str, Optional[str]]:
    """
    Auto-detect column mappings from CSV headers.
    Returns dict mapping our internal names to actual CSV column names.
    """
    mappings = {
        "date": overrides.get("date") or find_column(headers, DATE_COLUMNS),
        "datetime": overrides.get("datetime") or find_column(headers, DATETIME_COLUMNS),
        "time": overrides.get("time") or find_column(headers, TIME_COLUMNS),
        "sku": overrides.get("sku") or find_column(headers, SKU_COLUMNS),
        "product_name": overrides.get("product_name") or find_column(headers, NAME_COLUMNS),
        "quantity": overrides.get("quantity") or find_column(headers, QTY_COLUMNS),
        "unit_price": overrides.get("unit_price") or find_column(headers, PRICE_COLUMNS),
        "subtotal": overrides.get("subtotal") or find_column(headers, SUBTOTAL_COLUMNS),
        "cost": overrides.get("cost") or find_column(headers, COST_COLUMNS),
        "category": overrides.get("category") or find_column(headers, CATEGORY_COLUMNS),
        "invoice_id": overrides.get("invoice_id") or find_column(headers, INVOICE_COLUMNS),
        "location": overrides.get("location") or find_column(headers, LOCATION_COLUMNS),
        "gross_profit": overrides.get("gross_profit") or find_column(headers, GROSS_PROFIT_COLUMNS),
    }
    return mappings


def parse_date(date_str: str) -> Optional[str]:
    """Parse date string into YYYY-MM-DD format."""
    if not date_str:
        return None
    
    # Try common formats
    formats = [
        "%Y-%m-%d",
        "%m/%d/%Y",
        "%d/%m/%Y",
        "%Y/%m/%d",
        "%m-%d-%Y",
        "%d-%m-%Y",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(date_str.strip(), fmt)
            return dt.strftime("%Y-%m-%d")
        except ValueError:
            continue
    
    # Try ISO format with time
    try:
        dt = datetime.fromisoformat(date_str.strip().replace("Z", "+00:00"))
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        pass
    
    logger.warning(f"Could not parse date: {date_str}")
    return None


def parse_datetime(datetime_str: str) -> Optional[Tuple[str, str]]:
    """Parse combined datetime string into (date, time) tuple."""
    if not datetime_str:
        return None
    
    # Formats like "07/03/2021 09:02:42 AM"
    formats = [
        "%m/%d/%Y %I:%M:%S %p",
        "%m/%d/%Y %H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d %I:%M:%S %p",
        "%d/%m/%Y %H:%M:%S",
        "%d/%m/%Y %I:%M:%S %p",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(datetime_str.strip(), fmt)
            return (dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"))
        except ValueError:
            continue
    
    # Try ISO format
    try:
        dt = datetime.fromisoformat(datetime_str.strip().replace("Z", "+00:00"))
        return (dt.strftime("%Y-%m-%d"), dt.strftime("%H:%M:%S"))
    except ValueError:
        pass
    
    return None


def parse_time(time_str: str) -> Optional[str]:
    """Parse time string into HH:MM:SS format."""
    if not time_str:
        return None
    
    formats = [
        "%H:%M:%S",
        "%H:%M",
        "%I:%M:%S %p",
        "%I:%M %p",
    ]
    
    for fmt in formats:
        try:
            dt = datetime.strptime(time_str.strip(), fmt)
            return dt.strftime("%H:%M:%S")
        except ValueError:
            continue
    
    return None


def parse_number(value: str) -> Optional[float]:
    """Parse numeric value, handling currency symbols."""
    if not value:
        return None
    
    # Remove common currency symbols and whitespace
    cleaned = value.strip().replace("$", "").replace(",", "").replace(" ", "")
    
    try:
        return float(cleaned)
    except ValueError:
        return None


def init_database(db_path: str) -> sqlite3.Connection:
    """Initialize AOC database with sales table."""
    conn = sqlite3.connect(db_path)
    
    # Create sales table if not exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS sales (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            date TEXT NOT NULL,
            time TEXT,
            datetime_local TEXT,
            location TEXT DEFAULT 'central',
            sku TEXT NOT NULL,
            product_name TEXT,
            category TEXT,
            quantity INTEGER NOT NULL,
            unit_price REAL,
            subtotal REAL NOT NULL,
            cost REAL,
            gross_profit REAL,
            invoice_id TEXT,
            source TEXT DEFAULT 'backfill',
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        )
    """)
    
    # Create indexes
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_date ON sales(date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_sku ON sales(sku)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_location_date ON sales(location, date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_sales_datetime ON sales(datetime_local)")
    
    conn.commit()
    return conn


def check_duplicate(
    conn: sqlite3.Connection,
    date: str,
    time: Optional[str],
    sku: str,
    quantity: int,
    subtotal: float
) -> bool:
    """Check if this transaction already exists."""
    if time:
        cursor = conn.execute(
            """
            SELECT 1 FROM sales 
            WHERE date = ? AND time = ? AND sku = ? AND quantity = ? AND subtotal = ?
            LIMIT 1
            """,
            (date, time, sku, quantity, subtotal)
        )
    else:
        cursor = conn.execute(
            """
            SELECT 1 FROM sales 
            WHERE date = ? AND sku = ? AND quantity = ? AND subtotal = ?
            LIMIT 1
            """,
            (date, sku, quantity, subtotal)
        )
    
    return cursor.fetchone() is not None


def process_csv_file(
    conn: sqlite3.Connection,
    file_path: str,
    location: str,
    column_overrides: Dict[str, str],
    skip_duplicates: bool = True,
    skip_header_rows: int = 0
) -> Tuple[int, int, int]:
    """
    Process a single CSV file and insert into database.
    
    Args:
        conn: Database connection
        file_path: Path to CSV file
        location: Default location if not in CSV
        column_overrides: Manual column name mappings
        skip_duplicates: Skip rows that look like duplicates
        skip_header_rows: Number of metadata rows to skip before CSV header
    
    Returns: (rows_processed, rows_inserted, rows_skipped)
    """
    rows_processed = 0
    rows_inserted = 0
    rows_skipped = 0
    
    with open(file_path, 'r', encoding='utf-8-sig') as f:
        # Skip metadata/parameter rows if specified
        for _ in range(skip_header_rows):
            f.readline()
        
        # Sniff dialect
        sample = f.read(4096)
        f.seek(0)
        # Re-skip header rows after seeking
        for _ in range(skip_header_rows):
            f.readline()
        
        try:
            dialect = csv.Sniffer().sniff(sample)
        except csv.Error:
            dialect = csv.excel
        
        reader = csv.DictReader(f, dialect=dialect)
        headers = reader.fieldnames or []
        
        if not headers:
            logger.warning(f"No headers found in {file_path}")
            return 0, 0, 0
        
        # Detect column mappings
        mappings = detect_columns(headers, column_overrides)
        
        # Validate required columns - datetime can substitute for date
        has_date = mappings["date"] or mappings["datetime"]
        if not has_date:
            logger.error(f"Could not find date column in {file_path}. Headers: {headers}")
            return 0, 0, 0
        if not mappings["sku"]:
            logger.error(f"Could not find SKU column in {file_path}. Headers: {headers}")
            return 0, 0, 0
        if not mappings["quantity"] and not mappings["subtotal"]:
            logger.error(f"Could not find quantity or subtotal column in {file_path}")
            return 0, 0, 0
        
        logger.info(f"Column mappings for {Path(file_path).name}: {mappings}")
        
        # Process rows
        batch = []
        for row in reader:
            rows_processed += 1
            
            # Extract date and time - prefer datetime column if available
            date = None
            time = None
            
            if mappings["datetime"]:
                dt_result = parse_datetime(row.get(mappings["datetime"], ""))
                if dt_result:
                    date, time = dt_result
            
            if not date and mappings["date"]:
                date = parse_date(row.get(mappings["date"], ""))
            
            if not time and mappings["time"]:
                time = parse_time(row.get(mappings["time"], ""))
            
            if not date:
                rows_skipped += 1
                continue
            
            sku = row.get(mappings["sku"], "").strip()
            if not sku:
                rows_skipped += 1
                continue
            
            product_name = row.get(mappings["product_name"] or "", "").strip() or None
            category = row.get(mappings["category"] or "", "").strip() or None
            invoice_id = row.get(mappings["invoice_id"] or "", "").strip() or None
            
            # Use location from CSV if available, otherwise default
            row_location = location
            if mappings["location"]:
                csv_location = row.get(mappings["location"], "").strip()
                if csv_location:
                    row_location = csv_location
            
            quantity = parse_number(row.get(mappings["quantity"] or "", ""))
            unit_price = parse_number(row.get(mappings["unit_price"] or "", ""))
            subtotal = parse_number(row.get(mappings["subtotal"] or "", ""))
            cost = parse_number(row.get(mappings["cost"] or "", ""))
            
            # Need at least quantity or subtotal
            if quantity is None and subtotal is None:
                rows_skipped += 1
                continue
            
            # Default quantity to 1 if we have subtotal but no quantity
            if quantity is None:
                quantity = 1
            
            # Compute subtotal if missing
            if subtotal is None and unit_price:
                subtotal = quantity * unit_price
            elif subtotal is None:
                subtotal = 0.0
            
            # Use gross profit from CSV if available, otherwise compute
            gross_profit = None
            if mappings["gross_profit"]:
                gross_profit = parse_number(row.get(mappings["gross_profit"], ""))
            if gross_profit is None and cost and quantity:
                gross_profit = subtotal - (cost * quantity)
            
            # Build datetime_local
            datetime_local = f"{date}T{time}" if time else f"{date}T12:00:00"
            
            # Check for duplicates
            if skip_duplicates and check_duplicate(conn, date, time, sku, int(quantity), subtotal):
                rows_skipped += 1
                continue
            
            batch.append((
                date,
                time,
                datetime_local,
                row_location,
                sku,
                product_name,
                category,
                int(quantity),
                unit_price,
                subtotal,
                cost,
                gross_profit,
                invoice_id,
                "backfill"
            ))
            
            # Batch insert every 1000 rows
            if len(batch) >= 1000:
                conn.executemany(
                    """
                    INSERT INTO sales (
                        date, time, datetime_local, location, sku, product_name,
                        category, quantity, unit_price, subtotal, cost, gross_profit,
                        invoice_id, source
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                    batch
                )
                rows_inserted += len(batch)
                batch = []
        
        # Insert remaining
        if batch:
            conn.executemany(
                """
                INSERT INTO sales (
                    date, time, datetime_local, location, sku, product_name,
                    category, quantity, unit_price, subtotal, cost, gross_profit,
                    invoice_id, source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                batch
            )
            rows_inserted += len(batch)
        
        conn.commit()
    
    return rows_processed, rows_inserted, rows_skipped


def main():
    parser = argparse.ArgumentParser(
        description="Backfill historical sales data into AOC database"
    )
    
    # Input options
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--input-file",
        help="Path to single CSV file"
    )
    input_group.add_argument(
        "--input-dir",
        help="Path to directory containing CSV files"
    )
    
    # Database
    parser.add_argument(
        "--database",
        default=os.environ.get("AOC_DATABASE_PATH", "weather_sales.db"),
        help="Path to AOC SQLite database"
    )
    
    # Location
    parser.add_argument(
        "--location",
        default="central",
        help="Store location identifier (default: central)"
    )
    
    # Column overrides
    parser.add_argument("--date-col", help="Override date column name")
    parser.add_argument("--time-col", help="Override time column name")
    parser.add_argument("--datetime-col", help="Override combined datetime column name")
    parser.add_argument("--sku-col", help="Override SKU column name")
    parser.add_argument("--qty-col", help="Override quantity column name")
    parser.add_argument("--subtotal-col", help="Override subtotal column name")
    parser.add_argument("--name-col", help="Override product name column name")
    parser.add_argument("--category-col", help="Override category column name")
    
    # Options
    parser.add_argument(
        "--skip-rows",
        type=int,
        default=0,
        help="Number of metadata rows to skip before CSV header (e.g., 11 for Cova exports)"
    )
    parser.add_argument(
        "--allow-duplicates",
        action="store_true",
        help="Don't skip duplicate transactions"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse files but don't insert into database"
    )
    
    args = parser.parse_args()
    
    # Build column overrides
    column_overrides = {}
    if args.date_col:
        column_overrides["date"] = args.date_col
    if args.time_col:
        column_overrides["time"] = args.time_col
    if args.datetime_col:
        column_overrides["datetime"] = args.datetime_col
    if args.sku_col:
        column_overrides["sku"] = args.sku_col
    if args.qty_col:
        column_overrides["quantity"] = args.qty_col
    if args.subtotal_col:
        column_overrides["subtotal"] = args.subtotal_col
    if args.name_col:
        column_overrides["product_name"] = args.name_col
    if args.category_col:
        column_overrides["category"] = args.category_col
    
    # Collect CSV files
    if args.input_file:
        csv_files = [args.input_file]
    else:
        csv_files = sorted(glob.glob(os.path.join(args.input_dir, "**/*.csv"), recursive=True))
    
    if not csv_files:
        logger.error("No CSV files found")
        sys.exit(1)
    
    logger.info(f"Found {len(csv_files)} CSV file(s) to process")
    
    # Initialize database
    if args.dry_run:
        logger.info("DRY RUN - no data will be inserted")
        conn = init_database(":memory:")
    else:
        conn = init_database(args.database)
        logger.info(f"Using database: {args.database}")
    
    # Process files
    total_processed = 0
    total_inserted = 0
    total_skipped = 0
    
    for csv_file in csv_files:
        logger.info(f"Processing: {csv_file}")
        
        try:
            processed, inserted, skipped = process_csv_file(
                conn=conn,
                file_path=csv_file,
                location=args.location,
                column_overrides=column_overrides,
                skip_duplicates=not args.allow_duplicates,
                skip_header_rows=args.skip_rows
            )
            
            total_processed += processed
            total_inserted += inserted
            total_skipped += skipped
            
            logger.info(
                f"  → Processed: {processed}, Inserted: {inserted}, Skipped: {skipped}"
            )
            
        except Exception as e:
            logger.error(f"  → Failed: {e}")
    
    # Summary
    logger.info("=" * 60)
    logger.info("BACKFILL COMPLETE")
    logger.info(f"  Total rows processed: {total_processed}")
    logger.info(f"  Total rows inserted:  {total_inserted}")
    logger.info(f"  Total rows skipped:   {total_skipped}")
    
    if not args.dry_run:
        # Show date range
        cursor = conn.execute(
            "SELECT MIN(date), MAX(date), COUNT(*) FROM sales WHERE location = ?",
            (args.location,)
        )
        min_date, max_date, total_count = cursor.fetchone()
        logger.info(f"  Date range: {min_date} to {max_date}")
        logger.info(f"  Total sales records: {total_count}")
        
        # Show top SKUs
        cursor = conn.execute(
            """
            SELECT sku, SUM(quantity) as total_qty, SUM(subtotal) as total_revenue
            FROM sales WHERE location = ?
            GROUP BY sku ORDER BY total_revenue DESC LIMIT 5
            """,
            (args.location,)
        )
        logger.info("  Top 5 SKUs by revenue:")
        for row in cursor:
            logger.info(f"    {row[0]}: {row[1]} units, ${row[2]:.2f}")
    
    logger.info("=" * 60)
    logger.info("Next steps:")
    logger.info("  1. Fetch weather data: python -m aoc_analytics.cli fetch-weather")
    logger.info("  2. Rebuild signals:    python -m aoc_analytics.cli rebuild-signals")
    logger.info("  3. Test heroes:        python -m aoc_analytics.cli heroes")
    
    conn.close()


if __name__ == "__main__":
    main()
