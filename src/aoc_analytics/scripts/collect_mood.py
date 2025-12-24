#!/usr/bin/env python3
"""
Daily Mood Collection Script

Run this via cron to collect mood data daily:
    0 6 * * * cd /home/macklemoron/Projects/aoc-analytics && .venv/bin/python -m aoc_analytics.scripts.collect_mood

Or manually:
    python -m aoc_analytics.scripts.collect_mood
"""

import logging
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Load environment from JFK backend if available
JFK_ENV = Path("/home/macklemoron/Projects/JFK/backend/.env")
if JFK_ENV.exists():
    with open(JFK_ENV) as f:
        for line in f:
            line = line.strip()
            if line and not line.startswith('#') and '=' in line:
                key, value = line.split('=', 1)
                os.environ.setdefault(key.strip(), value.strip())


def setup_logging():
    """Configure logging for cron-friendly output."""
    log_dir = PROJECT_ROOT / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / f"mood_collection_{datetime.now().strftime('%Y%m')}.log"
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ]
    )
    return logging.getLogger(__name__)


def main():
    logger = setup_logging()
    logger.info("=" * 50)
    logger.info(f"Starting mood collection at {datetime.now()}")
    
    # Suppress noisy warnings
    import warnings
    warnings.filterwarnings('ignore')
    
    try:
        from aoc_analytics.core.mood import MoodOrchestrator, ProviderStatus
        
        # Database path
        db_path = PROJECT_ROOT / "aoc_sales.db"
        
        # Initialize orchestrator
        orchestrator = MoodOrchestrator(db_path=str(db_path), geo="CA")
        
        # Check providers
        logger.info("Checking providers...")
        health = orchestrator.check_providers()
        
        available_count = sum(
            1 for h in health.values() 
            if h.status == ProviderStatus.AVAILABLE
        )
        logger.info(f"  {available_count}/{len(health)} providers available")
        
        for name, h in health.items():
            logger.info(f"    {name}: {h.status.value}")
        
        # Fetch mood
        logger.info("Fetching daily mood...")
        mood = orchestrator.fetch_daily_mood()
        
        logger.info(f"Raw results:")
        logger.info(f"  Vibe: {mood.local_vibe_score:.2f}")
        logger.info(f"  Anxiety: {mood.local_vibe_anxiety:.2f}")
        logger.info(f"  Cozy: {mood.local_vibe_cozy:.2f}")
        logger.info(f"  Party: {mood.local_vibe_party:.2f}")
        logger.info(f"  Quality: {mood.overall_quality:.2f}")
        logger.info(f"  Providers: {', '.join(mood.providers_used)}")
        
        # Normalize against baseline
        logger.info("Computing z-scores against baseline...")
        normalized = orchestrator.normalize_mood(mood)
        
        logger.info(f"Z-scored results:")
        logger.info(f"  Vibe Z: {mood.local_vibe_z:+.2f}")
        logger.info(f"  Anxiety Z: {mood.anxiety_z:+.2f}")
        logger.info(f"  Cozy Z: {mood.cozy_z:+.2f}")
        logger.info(f"  Party Z: {mood.party_z:+.2f}")
        logger.info(f"  Baseline coverage: {normalized.baseline_coverage:.0%}")
        
        # Store
        orchestrator.store_mood(mood)
        logger.info(f"Stored to {db_path}")
        
        logger.info("Mood collection completed successfully")
        return 0
        
    except Exception as e:
        logger.exception(f"Mood collection failed: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
