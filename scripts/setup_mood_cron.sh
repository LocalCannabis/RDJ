#!/bin/bash
# Setup cron job for mood collection
# Run this once: ./setup_mood_cron.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_DIR="$(dirname "$SCRIPT_DIR")"
VENV_PYTHON="$PROJECT_DIR/.venv/bin/python"
CRON_CMD="cd $PROJECT_DIR && $VENV_PYTHON -m aoc_analytics.scripts.collect_mood >> $PROJECT_DIR/logs/cron.log 2>&1"

echo "Setting up mood collection cron jobs..."
echo ""
echo "Schedule:"
echo "  6:00 AM - Morning baseline (before store opens)"
echo "  2:00 PM - Midday check (peak activity)"  
echo "  9:00 PM - Evening mood (after-work crowd)"
echo ""

# Remove existing mood collection jobs
crontab -l 2>/dev/null | grep -v "collect_mood" > /tmp/crontab.tmp

# Add new cron jobs (3x daily)
echo "0 6 * * * $CRON_CMD" >> /tmp/crontab.tmp
echo "0 14 * * * $CRON_CMD" >> /tmp/crontab.tmp
echo "0 21 * * * $CRON_CMD" >> /tmp/crontab.tmp

crontab /tmp/crontab.tmp
rm /tmp/crontab.tmp

echo "✓ Cron jobs installed!"
echo ""
echo "To verify: crontab -l | grep collect_mood"
echo "To remove: crontab -l | grep -v 'collect_mood' | crontab -"
echo ""
echo "Logs: $PROJECT_DIR/logs/"
