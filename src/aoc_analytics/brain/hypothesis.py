"""
Hypothesis Engine - Forms and tests theories about what drives sales.

The scientific method for retail:
1. OBSERVE - Notice a pattern in data
2. HYPOTHESIZE - Form a testable theory
3. PREDICT - State what should happen if theory is true
4. TEST - Wait for conditions and check outcome
5. REFINE - Update beliefs based on results

This engine automates this cycle.
"""

import json
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Optional, Callable
import requests

from aoc_analytics.core.db_adapter import get_connection
from aoc_analytics.brain.memory import BrainMemory, Hypothesis, MemoryEntry


@dataclass
class TestResult:
    """Result of testing a hypothesis."""
    hypothesis_id: str
    conditions_observed: dict
    predicted_outcome: str
    actual_outcome: str
    prediction_correct: bool
    confidence_delta: float


class HypothesisEngine:
    """
    Forms, tests, and refines hypotheses about sales drivers.
    
    The engine can:
    1. Generate hypotheses from observations
    2. Identify when conditions are right to test a hypothesis
    3. Evaluate whether predictions came true
    4. Update confidence based on outcomes
    """
    
    def __init__(self, memory: BrainMemory, 
                 sales_db_path: str = "aoc_sales.db",
                 ollama_url: str = "http://localhost:11434"):
        self.memory = memory
        self.sales_db = sales_db_path
        self.ollama_url = ollama_url
    
    def generate_hypotheses_from_data(self, lookback_days: int = 90) -> list[Hypothesis]:
        """
        Analyze recent data and generate hypotheses about patterns.
        
        This is where the brain "thinks" - finding patterns and
        formulating theories about why they exist.
        """
        conn = get_connection(self.sales_db)
        
        # Gather data summaries for pattern detection
        patterns = self._detect_patterns(conn, lookback_days)
        conn.close()
        
        if not patterns:
            return []
        
        # Use LLM to generate hypotheses from patterns
        hypotheses = []
        for pattern in patterns:
            h = self._pattern_to_hypothesis(pattern)
            if h:
                hypotheses.append(h)
        
        return hypotheses
    
    def _detect_patterns(self, conn, days: int) -> list[dict]:
        """Detect statistical patterns in sales data."""
        patterns = []
        
        # Pattern 1: Day-of-week anomalies
        dow_stats = conn.execute("""
            SELECT 
                strftime('%w', date) as dow,
                AVG(daily_total) as avg_rev,
                COUNT(*) as days,
                (AVG(daily_total) - (SELECT AVG(daily_total) FROM (
                    SELECT SUM(subtotal) as daily_total FROM sales 
                    WHERE date >= date('now', ?) GROUP BY date
                ))) / (SELECT AVG(daily_total) FROM (
                    SELECT SUM(subtotal) as daily_total FROM sales 
                    WHERE date >= date('now', ?) GROUP BY date
                )) * 100 as pct_diff
            FROM (
                SELECT date, SUM(subtotal) as daily_total 
                FROM sales WHERE date >= date('now', ?)
                GROUP BY date
            )
            GROUP BY dow
            HAVING ABS(pct_diff) > 10
        """, (f'-{days} days', f'-{days} days', f'-{days} days')).fetchall()
        
        dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        for dow, avg_rev, days_count, pct_diff in dow_stats:
            direction = "higher" if pct_diff > 0 else "lower"
            patterns.append({
                'type': 'dow_anomaly',
                'description': f"{dow_names[int(dow)]} sales are {abs(pct_diff):.1f}% {direction} than average",
                'conditions': {'day_of_week': int(dow)},
                'metric': 'daily_revenue',
                'expected_delta': pct_diff,
                'evidence': {'avg_revenue': avg_rev, 'sample_days': days_count}
            })
        
        # Pattern 2: Category correlations with day of week
        category_dow = conn.execute("""
            WITH daily_cat AS (
                SELECT 
                    date,
                    strftime('%w', date) as dow,
                    category,
                    SUM(subtotal) as cat_revenue
                FROM sales 
                WHERE date >= date('now', ?)
                GROUP BY date, category
            ),
            cat_avg AS (
                SELECT category, AVG(cat_revenue) as overall_avg
                FROM daily_cat GROUP BY category
            )
            SELECT 
                d.dow,
                d.category,
                AVG(d.cat_revenue) as dow_avg,
                c.overall_avg,
                (AVG(d.cat_revenue) - c.overall_avg) / c.overall_avg * 100 as pct_diff
            FROM daily_cat d
            JOIN cat_avg c ON d.category = c.category
            GROUP BY d.dow, d.category
            HAVING ABS(pct_diff) > 20 AND c.overall_avg > 100
            ORDER BY ABS(pct_diff) DESC
            LIMIT 10
        """, (f'-{days} days',)).fetchall()
        
        for dow, category, dow_avg, overall_avg, pct_diff in category_dow:
            direction = "higher" if pct_diff > 0 else "lower"
            patterns.append({
                'type': 'category_dow_correlation',
                'description': f"{category} sales are {abs(pct_diff):.1f}% {direction} on {dow_names[int(dow)]}",
                'conditions': {'day_of_week': int(dow), 'category': category},
                'metric': 'category_revenue',
                'expected_delta': pct_diff,
                'evidence': {'dow_avg': dow_avg, 'overall_avg': overall_avg}
            })
        
        # Pattern 3: Hour patterns
        hour_patterns = conn.execute("""
            WITH hourly AS (
                SELECT 
                    CAST(substr(time, 1, 2) AS INTEGER) as hour,
                    strftime('%w', date) as dow,
                    SUM(subtotal) as revenue,
                    COUNT(*) as txns
                FROM sales 
                WHERE date >= date('now', ?)
                GROUP BY hour, dow
            ),
            hour_avg AS (
                SELECT hour, AVG(revenue) as avg_rev FROM hourly GROUP BY hour
            )
            SELECT 
                h.hour,
                h.dow,
                h.revenue,
                a.avg_rev,
                (h.revenue - a.avg_rev) / a.avg_rev * 100 as pct_diff
            FROM hourly h
            JOIN hour_avg a ON h.hour = a.hour
            WHERE ABS((h.revenue - a.avg_rev) / a.avg_rev * 100) > 30
            ORDER BY ABS(pct_diff) DESC
            LIMIT 5
        """, (f'-{days} days',)).fetchall()
        
        for hour, dow, revenue, avg_rev, pct_diff in hour_patterns:
            direction = "busier" if pct_diff > 0 else "slower"
            patterns.append({
                'type': 'hour_dow_pattern',
                'description': f"{hour}:00 on {dow_names[int(dow)]} is {abs(pct_diff):.1f}% {direction} than typical {hour}:00",
                'conditions': {'hour': hour, 'day_of_week': int(dow)},
                'metric': 'hourly_revenue',
                'expected_delta': pct_diff,
                'evidence': {'actual_rev': revenue, 'avg_rev': avg_rev}
            })
        
        return patterns
    
    def _pattern_to_hypothesis(self, pattern: dict) -> Optional[Hypothesis]:
        """Convert a detected pattern into a testable hypothesis."""
        # Generate hypothesis statement
        statement = f"When {self._conditions_to_english(pattern['conditions'])}, {pattern['metric']} will be {pattern['expected_delta']:+.1f}% from baseline"
        
        # Create prediction
        if pattern['expected_delta'] > 0:
            prediction = f"{pattern['metric']}_change_pct > {pattern['expected_delta'] * 0.5:.1f}"
        else:
            prediction = f"{pattern['metric']}_change_pct < {pattern['expected_delta'] * 0.5:.1f}"
        
        h = Hypothesis(
            id=self.memory._generate_id(statement, "hyp"),
            statement=statement,
            conditions=pattern['conditions'],
            prediction=prediction,
            confidence=0.5,  # Start neutral
        )
        
        # Store it
        self.memory.store_hypothesis(h)
        
        return h
    
    def _conditions_to_english(self, conditions: dict) -> str:
        """Convert condition dict to readable string."""
        parts = []
        dow_names = ['Sunday', 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday']
        
        for key, value in conditions.items():
            if key == 'day_of_week':
                parts.append(f"it's {dow_names[value]}")
            elif key == 'hour':
                parts.append(f"the hour is {value}:00")
            elif key == 'category':
                parts.append(f"category is {value}")
            elif key == 'weather':
                parts.append(f"weather is {value}")
            else:
                parts.append(f"{key} is {value}")
        
        return " AND ".join(parts)
    
    def get_todays_testable_hypotheses(self) -> list[Hypothesis]:
        """Get hypotheses that can be tested today."""
        today = datetime.now()
        current_conditions = {
            'day_of_week': today.weekday(),  # 0=Monday in Python
            'hour': today.hour,
            'month': today.month,
            'is_weekend': today.weekday() >= 5,
        }
        
        # SQLite uses 0=Sunday, so adjust
        current_conditions['day_of_week'] = (today.weekday() + 1) % 7
        
        return self.memory.get_testable_hypotheses(current_conditions)
    
    def test_hypothesis(self, hypothesis: Hypothesis, 
                        actual_value: float, baseline_value: float) -> TestResult:
        """
        Test a hypothesis against actual data.
        
        Returns whether the prediction was correct and updates the hypothesis.
        """
        # Calculate actual change
        actual_change_pct = ((actual_value - baseline_value) / baseline_value) * 100
        
        # Parse prediction to get expected threshold
        prediction_correct = self._evaluate_prediction(
            hypothesis.prediction, 
            actual_change_pct
        )
        
        # Record the test
        self.memory.record_hypothesis_test(
            hypothesis.id,
            correct=prediction_correct,
            actual_outcome=f"change_pct={actual_change_pct:.1f}%"
        )
        
        # Calculate confidence adjustment
        if prediction_correct:
            confidence_delta = 0.1 * (1 - hypothesis.confidence)  # Diminishing returns
        else:
            confidence_delta = -0.1 * hypothesis.confidence
        
        return TestResult(
            hypothesis_id=hypothesis.id,
            conditions_observed=hypothesis.conditions,
            predicted_outcome=hypothesis.prediction,
            actual_outcome=f"{actual_change_pct:+.1f}% change",
            prediction_correct=prediction_correct,
            confidence_delta=confidence_delta,
        )
    
    def _evaluate_prediction(self, prediction: str, actual_value: float) -> bool:
        """Evaluate if a prediction was correct."""
        # Parse predictions like "metric_change_pct > 10" or "metric_change_pct < -5"
        match = re.search(r'([<>]=?)\s*([-\d.]+)', prediction)
        if not match:
            return False
        
        operator, threshold = match.groups()
        threshold = float(threshold)
        
        if operator == '>':
            return actual_value > threshold
        elif operator == '>=':
            return actual_value >= threshold
        elif operator == '<':
            return actual_value < threshold
        elif operator == '<=':
            return actual_value <= threshold
        
        return False
    
    def generate_insight_from_validated(self) -> Optional[str]:
        """
        Generate actionable insight from validated hypotheses.
        
        Uses LLM to synthesize multiple validated hypotheses into
        a coherent recommendation.
        """
        validated = self.memory.get_validated_hypotheses()
        
        if len(validated) < 2:
            return None
        
        # Format hypotheses for LLM
        hypotheses_text = "\n".join([
            f"- {h.statement} (confidence: {h.confidence:.0%}, tested {h.times_tested} times)"
            for h in validated[:10]
        ])
        
        prompt = f"""Based on these validated hypotheses from our sales data, generate ONE specific, actionable business recommendation.

VALIDATED HYPOTHESES:
{hypotheses_text}

Requirements:
- Be specific (mention days, times, categories)
- Quantify expected impact
- Consider operational feasibility
- Explain the reasoning

Format:
RECOMMENDATION: [one sentence action]
EXPECTED IMPACT: [quantified benefit]
REASONING: [why this works based on the hypotheses]
"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.3}
            },
            timeout=30
        )
        
        if response.status_code == 200:
            return response.json()['response']
        return None
    
    def get_hypothesis_summary(self) -> dict:
        """Get summary of hypothesis testing progress."""
        stats = {
            'total': 0,
            'untested': 0,
            'testing': 0,
            'validated': 0,
            'disproven': 0,
            'top_validated': [],
            'top_disproven': [],
        }
        
        rows = self.memory.conn.execute("""
            SELECT status, COUNT(*) FROM hypotheses GROUP BY status
        """).fetchall()
        
        for status, count in rows:
            stats[status] = count
            stats['total'] += count
        
        # Top validated
        validated = self.memory.conn.execute("""
            SELECT statement, confidence, times_tested, times_correct
            FROM hypotheses 
            WHERE status = 'validated'
            ORDER BY confidence DESC
            LIMIT 5
        """).fetchall()
        
        stats['top_validated'] = [
            {
                'statement': row[0],
                'confidence': row[1],
                'accuracy': row[3] / row[2] if row[2] > 0 else 0
            }
            for row in validated
        ]
        
        return stats
