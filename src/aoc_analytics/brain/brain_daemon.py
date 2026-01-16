#!/usr/bin/env python3
"""
Brain Daemon - Runs 24/7

This daemon:
1. Watches for new data imports
2. Processes sales data as it arrives
3. Runs hypothesis tests during idle time
4. Generates predictions daily
5. Sends alerts when anomalies detected

Run with: python brain_daemon.py
Or as systemd service for production
"""

import os
import sys
import time
import json
import logging
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Optional
import signal

from aoc_analytics.core.db_adapter import get_connection

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(Path(__file__).parent.parent.parent.parent.parent / "logs" / "brain_daemon.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class BrainDaemon:
    """
    The always-on brain daemon.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.cwd() / "aoc_sales.db")
        
        self.db_path = db_path
        self.running = True
        self.last_hypothesis_test = None
        self.last_prediction = None
        self.last_import_check = None
        
        # Intervals (in seconds)
        self.IMPORT_CHECK_INTERVAL = 60  # Check for new files every minute
        self.HYPOTHESIS_INTERVAL = 300   # Test a hypothesis every 5 minutes
        self.PREDICTION_INTERVAL = 3600  # Update predictions hourly
        self.HEALTH_CHECK_INTERVAL = 60  # Log health every minute
        
        # Import directory
        self.import_dir = Path(__file__).parent.parent.parent.parent.parent / "data" / "imports"
        self.processed_dir = self.import_dir / "processed"
        
        # Ensure directories exist
        self.import_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Track processed files
        self.processed_files = set()
        self._load_processed_files()
        
        # Setup signal handlers
        signal.signal(signal.SIGINT, self._handle_shutdown)
        signal.signal(signal.SIGTERM, self._handle_shutdown)
    
    def _handle_shutdown(self, signum, frame):
        """Handle graceful shutdown."""
        logger.info("Shutdown signal received, stopping daemon...")
        self.running = False
    
    def _load_processed_files(self):
        """Load list of already processed files."""
        processed_log = self.processed_dir / "processed.json"
        if processed_log.exists():
            with open(processed_log) as f:
                self.processed_files = set(json.load(f))
    
    def _save_processed_files(self):
        """Save list of processed files."""
        processed_log = self.processed_dir / "processed.json"
        with open(processed_log, "w") as f:
            json.dump(list(self.processed_files), f)
    
    def check_for_imports(self):
        """Check for and process new import files."""
        sales_dir = self.import_dir / "sales"
        if not sales_dir.exists():
            return
        
        for file_path in sales_dir.glob("*.csv"):
            if str(file_path) not in self.processed_files:
                try:
                    self.process_sales_import(file_path)
                    self.processed_files.add(str(file_path))
                    self._save_processed_files()
                except Exception as e:
                    logger.error(f"Error processing {file_path}: {e}")
    
    def process_sales_import(self, file_path: Path):
        """Process a sales import file."""
        logger.info(f"Processing sales import: {file_path.name}")
        
        import csv
        
        conn = get_connection(self.db_path)
        cur = conn.cursor()
        
        with open(file_path) as f:
            reader = csv.DictReader(f)
            rows_imported = 0
            
            for row in reader:
                try:
                    # Map CSV columns to database columns
                    cur.execute("""
                        INSERT OR IGNORE INTO sales 
                        (date, time, datetime_local, location, sku, product_name, 
                         category, quantity, unit_price, subtotal, cost, 
                         gross_profit, invoice_id, source, created_at)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        row.get('date'),
                        row.get('time'),
                        row.get('datetime'),
                        row.get('location'),
                        row.get('sku'),
                        row.get('product_name'),
                        row.get('category'),
                        float(row.get('quantity', 0)),
                        float(row.get('unit_price', 0)),
                        float(row.get('subtotal', 0)),
                        float(row.get('cost', 0) or 0),
                        float(row.get('gross_profit', 0) or 0),
                        row.get('invoice_id'),
                        'cova_import',
                        datetime.now().isoformat(),
                    ))
                    rows_imported += 1
                except Exception as e:
                    logger.warning(f"Error importing row: {e}")
        
        conn.commit()
        conn.close()
        
        logger.info(f"Imported {rows_imported} rows from {file_path.name}")
        
        # Check for anomalies
        self.check_anomalies()
    
    def check_anomalies(self):
        """Check for real-time anomalies."""
        try:
            from aoc_analytics.brain.pattern_alerts import PatternAlertSystem
            
            alerts = PatternAlertSystem(self.db_path)
            new_alerts = alerts.scan_for_alerts()
            
            if new_alerts:
                logger.warning(f"🚨 {len(new_alerts)} new alerts detected!")
                for alert in new_alerts:
                    logger.warning(f"  - {alert}")
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Anomaly check failed: {e}")
    
    def run_hypothesis_test(self):
        """Run one hypothesis test."""
        try:
            from aoc_analytics.brain.curiosity_engine import CuriosityEngine
            
            engine = CuriosityEngine(self.db_path)
            result = engine.explore_one()
            
            if result and result.proven:
                logger.info(f"✅ DISCOVERY: {result.name} - Effect: {result.effect_size:+.1%}")
            
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Hypothesis test failed: {e}")
    
    def update_predictions(self):
        """Update predictions for upcoming days."""
        try:
            from aoc_analytics.brain.predictor import DemandPredictor
            
            predictor = DemandPredictor(self.db_path)
            
            # Predict next 7 days
            for i in range(7):
                target_date = date.today() + timedelta(days=i)
                prediction = predictor.predict_day(target_date)
                logger.info(f"Prediction for {target_date}: {prediction}")
                
        except ImportError:
            pass
        except Exception as e:
            logger.error(f"Prediction update failed: {e}")
    
    def log_health(self):
        """Log daemon health status."""
        try:
            conn = get_connection(self.db_path)
            cur = conn.cursor()
            
            # Latest transaction date
            cur.execute("SELECT MAX(date), COUNT(*) FROM sales")
            latest_date, total_rows = cur.fetchone()
            
            # Today's sales
            today = date.today().isoformat()
            cur.execute("SELECT COUNT(*), SUM(quantity) FROM sales WHERE date = ?", (today,))
            today_txns, today_units = cur.fetchone()
            
            conn.close()
            
            logger.info(f"Health: {total_rows:,} total rows, latest: {latest_date}, today: {today_txns or 0} txns, {today_units or 0} units")
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
    
    def run(self):
        """Main daemon loop."""
        logger.info("=" * 60)
        logger.info("🧠 BRAIN DAEMON STARTING")
        logger.info("=" * 60)
        
        # Initialize
        self.last_import_check = datetime.now()
        self.last_hypothesis_test = datetime.now()
        self.last_prediction = datetime.now() - timedelta(hours=1)  # Force initial prediction
        last_health = datetime.now()
        
        while self.running:
            now = datetime.now()
            
            try:
                # Check for new imports
                if (now - self.last_import_check).seconds >= self.IMPORT_CHECK_INTERVAL:
                    self.check_for_imports()
                    self.last_import_check = now
                
                # Run hypothesis test (during idle)
                if (now - self.last_hypothesis_test).seconds >= self.HYPOTHESIS_INTERVAL:
                    self.run_hypothesis_test()
                    self.last_hypothesis_test = now
                
                # Update predictions
                if (now - self.last_prediction).seconds >= self.PREDICTION_INTERVAL:
                    self.update_predictions()
                    self.last_prediction = now
                
                # Log health
                if (now - last_health).seconds >= self.HEALTH_CHECK_INTERVAL:
                    self.log_health()
                    last_health = now
                
            except Exception as e:
                logger.error(f"Daemon loop error: {e}")
            
            # Sleep briefly
            time.sleep(1)
        
        logger.info("Brain daemon stopped")


def main():
    """Entry point."""
    daemon = BrainDaemon()
    daemon.run()


if __name__ == "__main__":
    main()
