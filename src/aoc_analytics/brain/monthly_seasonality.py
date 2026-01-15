"""
Monthly Seasonality Signal

Calibrated from actual historical data:
- Summer (Jul/Aug): +10-12%
- Pre-holiday dip (Nov/Dec): -9-14%
- Winter slow (Jan/Feb): -7-13%
- Spring/Fall shoulder: ±2-5%
"""

import logging
from datetime import date
from typing import Optional

logger = logging.getLogger(__name__)

# Calibrated from 2024-2025 data
# These are lifts relative to overall daily average
MONTHLY_SEASONALITY = {
    1: -0.065,   # January: -6.5%
    2: -0.127,   # February: -12.7%
    3: +0.020,   # March: +2.0%
    4: +0.013,   # April: +1.3%
    5: +0.053,   # May: +5.3%
    6: +0.049,   # June: +4.9%
    7: +0.092,   # July: +9.2%
    8: +0.119,   # August: +11.9%
    9: -0.037,   # September: -3.7%
    10: +0.051,  # October: +5.1%
    11: -0.085,  # November: -8.5%
    12: -0.137,  # December: -13.7% (excluding holiday spikes)
}

# Day-of-week seasonality (huge impact!)
# Friday is 40% above average, Sunday is 13% below
DOW_SEASONALITY = {
    0: -0.135,   # Sunday: -13.5%
    1: -0.082,   # Monday: -8.2%
    2: -0.102,   # Tuesday: -10.2%
    3: -0.111,   # Wednesday: -11.1%
    4: -0.054,   # Thursday: -5.4%
    5: +0.405,   # Friday: +40.5%
    6: +0.082,   # Saturday: +8.2%
}


def get_monthly_adjustment(target_date: Optional[date] = None) -> float:
    """
    Get the monthly seasonality adjustment for a date.
    
    Returns lift relative to overall average (e.g., 0.10 = 10% above average)
    """
    if target_date is None:
        target_date = date.today()
    
    month = target_date.month
    adjustment = MONTHLY_SEASONALITY.get(month, 0.0)
    
    logger.debug(f"Monthly adjustment for {target_date.strftime('%B')}: {adjustment:+.1%}")
    return adjustment


def get_dow_adjustment(target_date: Optional[date] = None) -> float:
    """
    Get the day-of-week seasonality adjustment for a date.
    
    Returns lift relative to overall average (e.g., 0.40 = 40% above average for Friday)
    """
    if target_date is None:
        target_date = date.today()
    
    dow = target_date.weekday()
    # Convert Python weekday (Mon=0) to our format (Sun=0)
    dow_sunday_first = (dow + 1) % 7
    
    adjustment = DOW_SEASONALITY.get(dow_sunday_first, 0.0)
    
    dow_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    logger.debug(f"DOW adjustment for {dow_names[dow]}: {adjustment:+.1%}")
    return adjustment


def get_baseline_adjustment(target_date: Optional[date] = None) -> dict:
    """
    Get both monthly and DOW adjustments for comprehensive baseline.
    
    Returns dict with:
    - monthly: Monthly seasonality lift
    - dow: Day-of-week lift
    - combined: Total expected lift from baseline factors
    """
    if target_date is None:
        target_date = date.today()
    
    monthly = get_monthly_adjustment(target_date)
    dow = get_dow_adjustment(target_date)
    
    # Combine multiplicatively: (1 + monthly) * (1 + dow) - 1
    combined = (1 + monthly) * (1 + dow) - 1
    
    return {
        "date": target_date.isoformat(),
        "monthly": monthly,
        "dow": dow,
        "combined": combined,
        "month_name": target_date.strftime("%B"),
        "dow_name": target_date.strftime("%A"),
    }


def get_dow_baseline_for_date(target_date: date) -> float:
    """
    Get the expected baseline for this day-of-week.
    
    This is used for DOW-relative predictions:
    Instead of asking "is this day above overall average?"
    We ask "is this Friday above typical Friday?"
    
    Returns the multiplier for this DOW (e.g., 1.405 for Friday)
    """
    dow = target_date.weekday()
    dow_sunday_first = (dow + 1) % 7
    adjustment = DOW_SEASONALITY.get(dow_sunday_first, 0.0)
    return 1.0 + adjustment


def format_seasonality_report(target_date: Optional[date] = None) -> str:
    """Format a human-readable seasonality report."""
    if target_date is None:
        target_date = date.today()
    
    adj = get_baseline_adjustment(target_date)
    
    lines = [
        f"📅 Seasonality for {target_date.strftime('%A, %B %d, %Y')}",
        "",
        f"  Monthly ({adj['month_name']}): {adj['monthly']:+.1%}",
        f"  Day-of-week ({adj['dow_name']}): {adj['dow']:+.1%}",
        f"  Combined baseline: {adj['combined']:+.1%}",
    ]
    
    # Add interpretation
    if adj['combined'] > 0.20:
        lines.append("")
        lines.append("  ⚡ Expect HIGH volume day (baseline factors alone)")
    elif adj['combined'] < -0.15:
        lines.append("")
        lines.append("  📉 Expect SLOW day (baseline factors alone)")
    
    return "\n".join(lines)


if __name__ == "__main__":
    from datetime import timedelta
    
    print("Monthly Seasonality Signal")
    print("=" * 50)
    
    # Show next 7 days
    today = date.today()
    print(f"\nNext 7 days:")
    print("-" * 50)
    
    for i in range(7):
        d = today + timedelta(days=i)
        adj = get_baseline_adjustment(d)
        print(f"{d.strftime('%a %b %d')}: "
              f"Monthly {adj['monthly']:+5.1%}, "
              f"DOW {adj['dow']:+5.1%}, "
              f"Combined {adj['combined']:+5.1%}")
    
    print()
    print(format_seasonality_report())
