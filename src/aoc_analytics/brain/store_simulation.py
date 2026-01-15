"""
Store Manager Simulation

Simulates what a store manager actually does:
1. Start the week with current inventory
2. Predict sales for each day
3. Place orders to maintain stock
4. Track stockouts and lost sales
5. Compare brain-guided vs naive ordering

This is the REAL test: can the brain help a manager
make better ordering decisions?
"""

import sqlite3
import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict
import random


@dataclass
class SKUState:
    """Current state of a single SKU."""
    sku: str
    name: str
    category: str
    on_hand: int = 0
    on_order: int = 0
    avg_daily_sales: float = 0.0
    last_order_date: Optional[date] = None
    
    # Tracking
    total_sold: int = 0
    total_stockouts: int = 0  # Days with stockout
    total_lost_sales: int = 0  # Units we couldn't sell


@dataclass
class OrderDecision:
    """A reorder decision."""
    sku: str
    quantity: int
    reason: str
    day: date
    arrives: date  # Lead time


@dataclass 
class DayResult:
    """Results for a single simulated day."""
    day: date
    predicted_lift: float
    actual_lift: float
    
    # By-SKU results
    sales: Dict[str, int] = field(default_factory=dict)
    stockouts: Dict[str, int] = field(default_factory=dict)  # Lost sales by SKU
    orders_placed: List[OrderDecision] = field(default_factory=list)
    
    @property
    def total_sales(self) -> int:
        return sum(self.sales.values())
    
    @property
    def total_stockouts(self) -> int:
        return sum(self.stockouts.values())
    
    @property
    def stockout_rate(self) -> float:
        total_demand = self.total_sales + self.total_stockouts
        if total_demand == 0:
            return 0.0
        return self.total_stockouts / total_demand


@dataclass
class SimulationResult:
    """Full simulation results."""
    strategy: str  # "naive" or "brain"
    days_simulated: int
    
    total_sales: int = 0
    total_stockouts: int = 0
    total_orders: int = 0
    total_order_units: int = 0
    
    # Value metrics
    revenue_captured: float = 0.0
    revenue_lost: float = 0.0
    
    daily_results: List[DayResult] = field(default_factory=list)
    
    @property
    def stockout_rate(self) -> float:
        total_demand = self.total_sales + self.total_stockouts
        if total_demand == 0:
            return 0.0
        return self.total_stockouts / total_demand
    
    @property
    def fill_rate(self) -> float:
        return 1.0 - self.stockout_rate


class StoreSimulator:
    """
    Simulates store operations with inventory management.
    
    Two modes:
    1. Naive: Order based on simple trailing average
    2. Brain: Use predictions to adjust orders up/down
    """
    
    # Configuration
    LEAD_TIME_DAYS = 2  # Orders arrive in 2 days
    REORDER_FREQUENCY = 3  # Can order every 3 days
    SAFETY_STOCK_DAYS = 3  # Keep 3 days of safety stock
    MIN_ORDER_QTY = 5  # Minimum order quantity
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.cwd() / "aoc_sales.db")
        self.db_path = db_path
        self.brain_dir = Path(__file__).parent / "data"
        
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def _get_sku_baselines(self, start_date: date) -> Dict[str, SKUState]:
        """Calculate baseline sales velocity for each SKU."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Get 30 days of history before simulation starts
        history_start = (start_date - timedelta(days=30)).strftime("%Y-%m-%d")
        history_end = (start_date - timedelta(days=1)).strftime("%Y-%m-%d")
        
        cur.execute("""
            SELECT 
                sku, 
                product_name,
                category,
                SUM(quantity) as total_sold,
                COUNT(DISTINCT date) as days_with_sales
            FROM sales
            WHERE date BETWEEN ? AND ?
            GROUP BY sku
            HAVING total_sold >= 5  -- Only track SKUs with meaningful volume
        """, (history_start, history_end))
        
        skus = {}
        for sku, name, category, total, days in cur.fetchall():
            avg_daily = total / 30  # Average over full period
            
            # Initialize with ~7 days of stock
            initial_stock = max(10, int(avg_daily * 7))
            
            skus[sku] = SKUState(
                sku=sku,
                name=name[:50] if name else sku,
                category=category or "Unknown",
                on_hand=initial_stock,
                avg_daily_sales=avg_daily,
            )
        
        conn.close()
        return skus
    
    def _get_actual_sales(self, day: date) -> Dict[str, int]:
        """Get actual sales for a specific day."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT sku, SUM(quantity)
            FROM sales
            WHERE date = ?
            GROUP BY sku
        """, (day.strftime("%Y-%m-%d"),))
        
        sales = {row[0]: row[1] for row in cur.fetchall()}
        conn.close()
        return sales
    
    def _get_prediction_for_date(self, day: date) -> float:
        """Get brain's prediction for a date."""
        # Load from backtest results
        backtest_path = self.brain_dir / "backtest_results.json"
        if backtest_path.exists():
            with open(backtest_path) as f:
                data = json.load(f)
            
            for d in data.get("daily_comparison", []):
                if d["date"] == day.strftime("%Y-%m-%d"):
                    return d["predicted_lift"]
        
        return 0.0  # Default: no lift
    
    def _get_actual_lift(self, day: date, baseline: float) -> float:
        """Calculate actual lift for a day."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT SUM(quantity) FROM sales WHERE date = ?
        """, (day.strftime("%Y-%m-%d"),))
        
        actual = cur.fetchone()[0] or 0
        conn.close()
        
        if baseline == 0:
            return 0.0
        return (actual - baseline) / baseline
    
    def _decide_order_naive(self, sku_state: SKUState, day: date) -> Optional[OrderDecision]:
        """
        Naive ordering: maintain safety stock based on trailing average.
        """
        # Calculate days of stock
        if sku_state.avg_daily_sales == 0:
            return None
        
        days_of_stock = sku_state.on_hand / sku_state.avg_daily_sales
        
        # Reorder if below safety stock + lead time
        reorder_point = self.SAFETY_STOCK_DAYS + self.LEAD_TIME_DAYS
        
        if days_of_stock < reorder_point:
            # Order enough for 7 days
            target_days = 7
            order_qty = max(
                self.MIN_ORDER_QTY,
                int(sku_state.avg_daily_sales * target_days) - sku_state.on_hand
            )
            
            return OrderDecision(
                sku=sku_state.sku,
                quantity=order_qty,
                reason=f"Below reorder point ({days_of_stock:.1f} days)",
                day=day,
                arrives=day + timedelta(days=self.LEAD_TIME_DAYS),
            )
        
        return None
    
    def _decide_order_brain(self, sku_state: SKUState, day: date, 
                           predicted_lift: float) -> Optional[OrderDecision]:
        """
        Brain-guided ordering: adjust for predicted demand changes.
        """
        if sku_state.avg_daily_sales == 0:
            return None
        
        # Adjust expected sales by prediction
        adjusted_daily = sku_state.avg_daily_sales * (1 + predicted_lift)
        
        # Calculate days of stock with adjusted rate
        days_of_stock = sku_state.on_hand / adjusted_daily if adjusted_daily > 0 else 999
        
        # More aggressive reorder point if expecting rush
        if predicted_lift > 0.10:
            reorder_point = self.SAFETY_STOCK_DAYS + self.LEAD_TIME_DAYS + 2
        elif predicted_lift < -0.10:
            reorder_point = self.SAFETY_STOCK_DAYS + self.LEAD_TIME_DAYS - 1
        else:
            reorder_point = self.SAFETY_STOCK_DAYS + self.LEAD_TIME_DAYS
        
        if days_of_stock < reorder_point:
            # Order based on adjusted forecast
            target_days = 7
            order_qty = max(
                self.MIN_ORDER_QTY,
                int(adjusted_daily * target_days) - sku_state.on_hand
            )
            
            # If expecting rush, order extra
            if predicted_lift > 0.15:
                order_qty = int(order_qty * 1.2)
            
            reason = f"Below reorder ({days_of_stock:.1f} days)"
            if predicted_lift > 0.10:
                reason += f" + rush expected ({predicted_lift:+.0%})"
            elif predicted_lift < -0.10:
                reason += f" + slow expected ({predicted_lift:+.0%})"
            
            return OrderDecision(
                sku=sku_state.sku,
                quantity=order_qty,
                reason=reason,
                day=day,
                arrives=day + timedelta(days=self.LEAD_TIME_DAYS),
            )
        
        return None
    
    def run_simulation(self, start_date: date, days: int, 
                       strategy: str = "naive") -> SimulationResult:
        """
        Run the full simulation.
        
        Args:
            start_date: First day to simulate
            days: Number of days to simulate
            strategy: "naive" or "brain"
        """
        print(f"Running {strategy} simulation for {days} days...")
        
        # Initialize SKU states
        skus = self._get_sku_baselines(start_date)
        print(f"  Tracking {len(skus)} SKUs")
        
        # Calculate baseline for lift calculations
        conn = self.get_connection()
        cur = conn.cursor()
        cur.execute("""
            SELECT AVG(daily_total) FROM (
                SELECT date, SUM(quantity) as daily_total
                FROM sales
                WHERE date < ?
                GROUP BY date
            )
        """, (start_date.strftime("%Y-%m-%d"),))
        baseline_units = cur.fetchone()[0] or 400
        conn.close()
        
        # Track pending orders
        pending_orders: List[OrderDecision] = []
        
        result = SimulationResult(
            strategy=strategy,
            days_simulated=days,
        )
        
        avg_basket = 24.27  # From our data
        
        for i in range(days):
            current_day = start_date + timedelta(days=i)
            
            # Receive any orders that arrived
            for order in pending_orders[:]:
                if order.arrives <= current_day:
                    if order.sku in skus:
                        skus[order.sku].on_hand += order.quantity
                    pending_orders.remove(order)
            
            # Get prediction
            predicted_lift = self._get_prediction_for_date(current_day)
            actual_lift = self._get_actual_lift(current_day, baseline_units)
            
            # Get actual sales
            actual_sales = self._get_actual_sales(current_day)
            
            day_result = DayResult(
                day=current_day,
                predicted_lift=predicted_lift,
                actual_lift=actual_lift,
            )
            
            # Process sales for each SKU
            for sku, demand in actual_sales.items():
                if sku not in skus:
                    continue
                
                sku_state = skus[sku]
                
                # How much can we actually sell?
                can_sell = min(demand, sku_state.on_hand)
                stockout = demand - can_sell
                
                # Update state
                sku_state.on_hand -= can_sell
                sku_state.total_sold += can_sell
                if stockout > 0:
                    sku_state.total_stockouts += 1
                    sku_state.total_lost_sales += stockout
                
                day_result.sales[sku] = can_sell
                if stockout > 0:
                    day_result.stockouts[sku] = stockout
            
            # Make ordering decisions (every REORDER_FREQUENCY days)
            if i % self.REORDER_FREQUENCY == 0:
                for sku, sku_state in skus.items():
                    if strategy == "naive":
                        order = self._decide_order_naive(sku_state, current_day)
                    else:
                        order = self._decide_order_brain(sku_state, current_day, predicted_lift)
                    
                    if order:
                        pending_orders.append(order)
                        day_result.orders_placed.append(order)
                        sku_state.on_order += order.quantity
                        result.total_orders += 1
                        result.total_order_units += order.quantity
            
            # Update totals
            result.total_sales += day_result.total_sales
            result.total_stockouts += day_result.total_stockouts
            result.revenue_captured += day_result.total_sales * avg_basket
            result.revenue_lost += day_result.total_stockouts * avg_basket
            
            result.daily_results.append(day_result)
        
        return result
    
    def compare_strategies(self, start_date: date = None, days: int = 60) -> dict:
        """
        Run both strategies and compare results.
        """
        if start_date is None:
            # Use recent historical period
            start_date = date(2025, 10, 1)
        
        naive = self.run_simulation(start_date, days, "naive")
        brain = self.run_simulation(start_date, days, "brain")
        
        print("\n" + "=" * 70)
        print("📊 STRATEGY COMPARISON")
        print("=" * 70)
        
        print(f"\n{'Metric':<30} {'Naive':>15} {'Brain':>15} {'Diff':>15}")
        print("-" * 75)
        
        def compare(label, naive_val, brain_val, fmt="{:,.0f}", better="higher"):
            diff = brain_val - naive_val
            if better == "higher":
                indicator = "✅" if diff > 0 else "❌" if diff < 0 else "="
            else:
                indicator = "✅" if diff < 0 else "❌" if diff > 0 else "="
            
            print(f"{label:<30} {fmt.format(naive_val):>15} {fmt.format(brain_val):>15} {indicator} {fmt.format(diff):>12}")
        
        compare("Units Sold", naive.total_sales, brain.total_sales)
        compare("Units Lost (Stockouts)", naive.total_stockouts, brain.total_stockouts, better="lower")
        compare("Fill Rate", naive.fill_rate * 100, brain.fill_rate * 100, fmt="{:.1f}%")
        compare("Revenue Captured", naive.revenue_captured, brain.revenue_captured, fmt="${:,.0f}")
        compare("Revenue Lost", naive.revenue_lost, brain.revenue_lost, fmt="${:,.0f}", better="lower")
        compare("Orders Placed", naive.total_orders, brain.total_orders, better="lower")
        compare("Units Ordered", naive.total_order_units, brain.total_order_units)
        
        # Calculate true ROI
        revenue_gain = brain.revenue_captured - naive.revenue_captured
        stockout_reduction = naive.revenue_lost - brain.revenue_lost
        total_value = revenue_gain + stockout_reduction
        
        print("\n" + "=" * 70)
        print("💰 TRUE ROI FROM BRAIN")
        print("=" * 70)
        print(f"\n   Extra revenue captured: ${revenue_gain:,.2f}")
        print(f"   Stockout reduction: ${stockout_reduction:,.2f}")
        print(f"   TOTAL VALUE: ${total_value:,.2f}")
        print(f"   Over {days} days = ${total_value/days*30:,.2f}/month")
        
        return {
            "naive": naive,
            "brain": brain,
            "revenue_gain": revenue_gain,
            "stockout_reduction": stockout_reduction,
            "total_value": total_value,
            "monthly_value": total_value / days * 30,
        }


if __name__ == "__main__":
    sim = StoreSimulator()
    results = sim.compare_strategies(days=60)
