# Path to Grade A: Action Plan

## Current State (Grade C)
- **Value**: $26-39/day ($771-$1182/month)
- **Rush Day Accuracy**: 3-19% (catching 1-7 of 36 rush days)
- **Signal Calibration**: Much improved, individual signals within ±3%

## Why We're Stuck

### The Fundamental Problem: High Variance
- Daily sales have **15.6% standard deviation**
- Only 32% of days are "normal" (±5% from baseline)
- Our signals explain maybe ±5%, leaving ±10% as noise

### What We CAN'T Predict
1. **Cruise day variance**: Average +3.2%, but ranges from -20% to +78%
2. **Random monthly surges**: Oct 2025 was +21% vs trend (no known cause)
3. **Walk-in variance**: Random foot traffic swings

### What We CAN Predict (And Are Getting Right)
1. ✅ **Pre-Christmas** (+65% on Dec 23)
2. ✅ **Cruise season baseline** (+3.2% average)
3. ✅ **Canucks games** (+6.7%)
4. ✅ **Academic periods** (-6.4% during finals)
5. ✅ **Fridays** (+3%)

## Path to Grade B ($40+/day, 25%+ accuracy)

### Option 1: Better Signals (Hard)
Need external data we don't have:
- [ ] Actual cruise ship schedules (which ships, passenger counts)
- [ ] Local event database (concerts, conferences)
- [ ] Weather forecasts integrated into predictions
- [ ] Payday calendar (government, major employers)

### Option 2: Probabilistic Predictions (Medium)
Instead of point predictions, predict ranges:
- "Expect 350-500 units (cruise season variance)"
- "High confidence: 400-450 units"
- This matches reality better than false precision

### Option 3: Focus on Actionable Alerts (Easy)
Instead of daily predictions, focus on:
- **Alert when BIG events are coming** (pre-Christmas, Canucks, etc.)
- **Flag high-uncertainty periods** (cruise season = "expect variance")
- **Catch the slow days** (academic finals = reduce orders)

## Immediate Improvements to Implement

### 1. Add Cruise Schedule Data
If we can get actual cruise schedules from Port of Vancouver:
- Know which days have 3+ ships vs 1 ship
- Predict monster days (+50%+) vs normal cruise days (+3%)

### 2. Add Monthly Seasonality
October trends higher (students back, pre-Halloween buildup):
```python
MONTHLY_ADJUSTMENT = {
    "January": -0.05,
    "February": 0.0,
    "March": 0.0,
    "April": 0.0,
    "May": +0.02,
    "June": +0.03,
    "July": +0.05,  # Tourist season
    "August": +0.05,
    "September": -0.02,  # Post-summer dip
    "October": +0.05,  # Students back + pre-Halloween
    "November": -0.03,
    "December": +0.05,  # Holiday shopping
}
```

### 3. Day-of-Week in Baseline
Compare to same-day-of-week average, not overall average:
- "Is this Friday busier than typical Friday?"
- Removes 30-50% of variance

### 4. Weather Integration
Already have weather API - wire it into predictions:
- Cold + rain = stay-home effect
- First nice day = outdoor activity effect

## Realistic Grade Targets

| Timeframe | Target | What's Needed |
|-----------|--------|---------------|
| Now | C | Current state |
| 1 week | C+ | Add monthly seasonality |
| 2 weeks | B- | Add DOW-relative baseline |
| 1 month | B | Add weather to predictions |
| 3 months | B+ | Get cruise schedules |
| 6 months | A | Full data integration + ML |

## Bottom Line

**Grade A requires data we don't have.** 

With current signals, realistic ceiling is **Grade B** (~$50/day, ~35% rush accuracy).

To achieve Grade A ($75+/day, 40%+ accuracy), we need:
1. Actual cruise schedules
2. Local event calendar
3. Weather prediction integration
4. 6+ months of calibration data

**Recommendation**: Focus on Grade B first. That's achievable with:
- Monthly seasonality adjustments
- DOW-relative predictions  
- Weather integration (already built)
- Probabilistic ranges instead of point estimates
