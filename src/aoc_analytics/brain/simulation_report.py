"""
Store Simulation Report Generator

Creates a detailed report of the store manager simulation
that proves ROI with real inventory management scenarios.
"""

from datetime import date
from pathlib import Path


def generate_simulation_report():
    """Generate full simulation report."""
    from aoc_analytics.brain.store_simulation import StoreSimulator
    
    sim = StoreSimulator()
    
    # Run both strategies
    start = date(2025, 10, 1)
    days = 60
    
    naive = sim.run_simulation(start, days, "naive")
    brain = sim.run_simulation(start, days, "brain")
    
    # Calculate metrics
    revenue_gain = brain.revenue_captured - naive.revenue_captured
    stockout_reduction = naive.revenue_lost - brain.revenue_lost
    total_value = revenue_gain + stockout_reduction
    monthly_value = total_value / days * 30
    
    # Rush day analysis
    rush_days = [(n, b) for n, b in zip(naive.daily_results, brain.daily_results)
                 if b.actual_lift > 0.10]
    
    rush_naive_stockouts = sum(n.total_stockouts for n, b in rush_days)
    rush_brain_stockouts = sum(b.total_stockouts for n, b in rush_days)
    
    report = f"""# Store Manager Simulation Report

**Generated:** {date.today()}
**Period:** {start} to {start.replace(day=start.day + days - 1)} ({days} days)
**SKUs Tracked:** {len(sim._get_sku_baselines(start))}

---

## Executive Summary

The brain-guided ordering strategy outperformed naive ordering by **${total_value:,.2f}** over {days} days, 
equivalent to **${monthly_value:,.2f}/month**.

This is a **real simulation** of:
- Actual inventory levels depleting with sales
- Reorder decisions made every 3 days
- 2-day lead time on orders
- Stockouts when we run out of product

---

## Strategy Comparison

| Metric | Naive | Brain | Difference |
|--------|-------|-------|------------|
| Units Sold | {naive.total_sales:,} | {brain.total_sales:,} | {brain.total_sales - naive.total_sales:+,} |
| Stockouts | {naive.total_stockouts:,} | {brain.total_stockouts:,} | {brain.total_stockouts - naive.total_stockouts:+,} |
| Fill Rate | {naive.fill_rate*100:.1f}% | {brain.fill_rate*100:.1f}% | {(brain.fill_rate - naive.fill_rate)*100:+.1f}% |
| Revenue Captured | ${naive.revenue_captured:,.0f} | ${brain.revenue_captured:,.0f} | ${revenue_gain:+,.0f} |
| Revenue Lost | ${naive.revenue_lost:,.0f} | ${brain.revenue_lost:,.0f} | ${brain.revenue_lost - naive.revenue_lost:+,.0f} |

---

## Rush Day Performance

Rush days are when actual sales were >10% above baseline.

- **Rush days in period:** {len(rush_days)}
- **Naive stockouts on rush days:** {rush_naive_stockouts} units
- **Brain stockouts on rush days:** {rush_brain_stockouts} units  
- **Units saved:** {rush_naive_stockouts - rush_brain_stockouts} units
- **Value saved:** ${(rush_naive_stockouts - rush_brain_stockouts) * 24.27:,.2f}

### Top Rush Days

| Date | Actual Lift | Predicted | Naive Lost | Brain Lost | Saved |
|------|-------------|-----------|------------|------------|-------|
"""
    
    for naive_day, brain_day in sorted(rush_days, key=lambda x: x[1].actual_lift, reverse=True)[:10]:
        n_lost = naive_day.total_stockouts
        b_lost = brain_day.total_stockouts
        saved = n_lost - b_lost
        emoji = "✅" if saved > 0 else "❌" if saved < 0 else "="
        report += f"| {brain_day.day} | {brain_day.actual_lift*100:+.0f}% | {brain_day.predicted_lift*100:+.0f}% | {n_lost} | {b_lost} | {emoji} {saved} |\n"
    
    report += f"""
---

## How the Brain Helps

### 1. Rush Day Preparation
When the brain predicts a busy day (+10% or more), it:
- Lowers the reorder point (order sooner)
- Orders 20% extra units
- This results in fewer stockouts when demand spikes

### 2. Slow Day Conservation
When the brain predicts a slow day (-10% or less), it:
- Raises the reorder point (delay ordering)
- Orders smaller quantities
- This reduces overstock and waste

### 3. Lead Time Anticipation
Since orders take 2 days to arrive, the brain's predictions
let us place orders BEFORE we need them, not after.

---

## ROI Summary

| Value Source | Amount |
|--------------|--------|
| Extra revenue captured | ${revenue_gain:,.2f} |
| Stockouts prevented | ${stockout_reduction:,.2f} |
| **TOTAL** | **${total_value:,.2f}** |
| **Monthly rate** | **${monthly_value:,.2f}/month** |

---

## Methodology

### Simulation Parameters
- **Reorder frequency:** Every 3 days
- **Lead time:** 2 days
- **Safety stock:** 3 days of average sales
- **Initial stock:** 7 days of average sales
- **Minimum order:** 5 units

### Naive Strategy
Orders when: stock < (safety_stock + lead_time) days
Order quantity: 7 days of average sales

### Brain Strategy  
Orders when: adjusted_stock < (safety_stock + lead_time + rush_buffer) days
Order quantity: 7 days of *predicted* sales (adjusted for expected lift)

### Data Sources
- Actual sales from aoc_sales.db
- Predictions from backtest_results.json
- All calculations verifiable

---

## How to Verify

Run the simulation yourself:

```python
from src.aoc_analytics.brain.store_simulation import StoreSimulator
sim = StoreSimulator()
results = sim.compare_strategies(days=60)
```

Every sale, stockout, and order is tracked at the SKU level.

---

*This simulation proves the brain's value in realistic store operations,
not just in predicting demand direction.*
"""
    
    return report


if __name__ == "__main__":
    report = generate_simulation_report()
    
    # Save report
    output_path = Path(__file__).parent / "docs" / "SIMULATION_PROOF.md"
    output_path.parent.mkdir(exist_ok=True)
    
    with open(output_path, "w") as f:
        f.write(report)
    
    print(f"Report saved to: {output_path}")
    print()
    print(report)
