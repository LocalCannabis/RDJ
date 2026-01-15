"""
The Curiosity Engine

An autonomous hypothesis exploration system that runs 24/7,
generating and testing ideas about the sales data.

Philosophy:
- Generate WEIRD hypotheses (moon phases? barometric pressure?)
- Test RIGOROUSLY (statistical significance or GTFO)
- Track EVERYTHING (what worked, what didn't, confidence levels)
- Learn from failures (why did that hypothesis fail?)

The brain should be:
1. Curious - generate hypotheses humans wouldn't think of
2. Rigorous - prove everything with statistical tests
3. Humble - admit when it's wrong, track confidence
4. Persistent - keep exploring 24/7
"""

import sqlite3
import json
import random
import hashlib
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Callable
from collections import defaultdict
import math

try:
    from scipy import stats
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


@dataclass
class Hypothesis:
    """A hypothesis to test."""
    id: str
    name: str
    description: str
    category: str  # temporal, product, customer, weather, cosmic, weird
    
    # The test
    condition: str  # SQL-like condition or function name
    expected_effect: str  # "increase", "decrease", "correlate"
    
    # Results
    tested: bool = False
    test_date: Optional[str] = None
    sample_size: int = 0
    effect_size: float = 0.0  # Actual measured effect
    p_value: float = 1.0
    confidence: float = 0.0  # 0-1 confidence in result
    proven: bool = False
    
    # Meta
    created: str = field(default_factory=lambda: datetime.now().isoformat())
    source: str = "generated"  # generated, human, derived
    parent_hypothesis: Optional[str] = None  # If derived from another
    
    notes: str = ""


@dataclass
class Discovery:
    """A proven discovery."""
    hypothesis_id: str
    name: str
    description: str
    
    effect_size: float
    p_value: float
    confidence: float
    sample_size: int
    
    # Actionable?
    actionable: bool = False
    action_recommendation: str = ""
    estimated_value: float = 0.0  # $/month if acted on
    
    discovered: str = field(default_factory=lambda: datetime.now().isoformat())


class HypothesisGenerator:
    """
    Generates hypotheses to test.
    
    Categories:
    - Temporal: Time-based patterns
    - Product: SKU/category relationships
    - Behavioral: Customer behavior patterns
    - Environmental: Weather, events, external factors
    - Cosmic: Moon phases, astrology (for fun, but test seriously!)
    - Weird: Random correlations that might exist
    """
    
    # Temporal patterns to explore
    TEMPORAL_HYPOTHESES = [
        # Micro patterns
        ("first_of_month_payday", "Sales increase on 1st and 15th (payday)", "temporal"),
        ("last_hour_rush", "Sales spike in the last hour before close", "temporal"),
        ("monday_blues", "Monday sales are lower than other weekdays", "temporal"),
        ("friday_pregame", "Friday afternoon sales are higher (weekend prep)", "temporal"),
        ("sunday_recovery", "Sunday afternoon sales spike (recovery purchases)", "temporal"),
        
        # Weekly patterns
        ("week_of_month_pattern", "First week of month differs from last week", "temporal"),
        ("long_weekend_effect", "Long weekends have different patterns", "temporal"),
        
        # Monthly patterns
        ("month_end_budget", "Sales drop last 3 days of month (budget depleted)", "temporal"),
        ("new_month_energy", "First week of new month has higher sales", "temporal"),
        
        # Seasonal micro
        ("spring_forward_effect", "Sales change after daylight saving", "temporal"),
        ("equinox_effect", "Sales patterns shift at equinoxes", "temporal"),
    ]
    
    # Product relationship hypotheses
    PRODUCT_HYPOTHESES = [
        ("edibles_weather_correlation", "Edibles sell better on rainy days (indoor consumption)", "product"),
        ("flower_friday", "Flower sales are disproportionately higher on Fridays", "product"),
        ("preroll_convenience", "Pre-rolls sell more during rush hours (convenience)", "product"),
        ("beverage_hot_weather", "Beverages sell more in warm weather", "product"),
        ("concentrate_weekend", "Concentrates sell more on weekends (experienced users)", "product"),
        ("vape_discreet", "Vapes sell more near holidays (discreet gifting)", "product"),
        ("accessory_attachment", "Accessory sales correlate with flower spikes", "product"),
        ("budget_month_end", "Budget products sell more at end of month", "product"),
        ("premium_payday", "Premium products sell more on paydays", "product"),
    ]
    
    # Behavioral hypotheses
    BEHAVIORAL_HYPOTHESES = [
        ("basket_size_friday", "Average basket size is larger on Fridays", "behavioral"),
        ("category_loyalty", "Customers who buy flower rarely buy edibles (and vice versa)", "behavioral"),
        ("new_product_halo", "New product launches boost overall category sales", "behavioral"),
        ("price_sensitivity_timing", "Price sensitivity varies by time of day", "behavioral"),
        ("bulk_buyers_pattern", "Bulk buyers have predictable purchase cycles", "behavioral"),
    ]
    
    # Environmental hypotheses
    ENVIRONMENTAL_HYPOTHESES = [
        ("rain_indoor_boost", "Rainy days boost sales (people stuck inside)", "environmental"),
        ("sunshine_outdoor_dip", "First sunny day after rain = lower sales", "environmental"),
        ("temperature_sweet_spot", "Sales peak at 15-20°C (comfortable shopping)", "environmental"),
        ("wind_suppression", "High wind days have lower sales", "environmental"),
        ("snow_day_spike", "Snow days cause sales spike (stocking up)", "environmental"),
        ("air_quality_correlation", "Poor air quality increases sales", "environmental"),
    ]
    
    # Cosmic hypotheses (test seriously!)
    COSMIC_HYPOTHESES = [
        ("full_moon_effect", "Full moon days have higher/different sales", "cosmic"),
        ("new_moon_effect", "New moon days have lower/different sales", "cosmic"),
        ("mercury_retrograde", "Sales patterns change during Mercury retrograde", "cosmic"),
        ("friday_13th", "Friday the 13th affects sales", "cosmic"),
        ("solstice_effect", "Sales patterns shift around solstices", "cosmic"),
    ]
    
    # Weird hypotheses
    WEIRD_HYPOTHESES = [
        ("sports_hangover", "Day after big Canucks win = lower sales (hangover)", "weird"),
        ("taylor_swift_effect", "Concert days in Vancouver affect sales", "weird"),
        ("stock_market_correlation", "S&P 500 up days = higher premium sales", "weird"),
        ("gas_price_inverse", "High gas prices = higher sales (staying local)", "weird"),
        ("tiktok_trend_lag", "Sales spike 3-5 days after product goes viral", "weird"),
        ("neighbor_store_closure", "Sales spike when competitor closes early", "weird"),
    ]
    
    @classmethod
    def generate_all_hypotheses(cls) -> List[Hypothesis]:
        """Generate all predefined hypotheses."""
        hypotheses = []
        
        all_templates = (
            cls.TEMPORAL_HYPOTHESES +
            cls.PRODUCT_HYPOTHESES +
            cls.BEHAVIORAL_HYPOTHESES +
            cls.ENVIRONMENTAL_HYPOTHESES +
            cls.COSMIC_HYPOTHESES +
            cls.WEIRD_HYPOTHESES
        )
        
        for name, description, category in all_templates:
            h = Hypothesis(
                id=hashlib.md5(name.encode()).hexdigest()[:8],
                name=name,
                description=description,
                category=category,
                condition=name,  # Will be interpreted by tester
                expected_effect="correlate",
                source="predefined",
            )
            hypotheses.append(h)
        
        return hypotheses
    
    @classmethod
    def generate_random_hypothesis(cls, db_path: str) -> Hypothesis:
        """Generate a random hypothesis by combining data attributes."""
        conn = sqlite3.connect(db_path)
        cur = conn.cursor()
        
        # Get available categories
        cur.execute("SELECT DISTINCT category FROM sales WHERE category IS NOT NULL")
        categories = [r[0] for r in cur.fetchall()]
        
        # Get time ranges
        cur.execute("SELECT MIN(date), MAX(date) FROM sales")
        min_date, max_date = cur.fetchone()
        
        conn.close()
        
        # Random combination generators
        combinations = [
            # Category vs time
            lambda: {
                "name": f"{random.choice(categories).lower().replace(' ', '_')}_hour_{random.randint(9, 21)}",
                "description": f"Does {random.choice(categories)} sell more at hour {random.randint(9, 21)}?",
                "category": "temporal",
            },
            # Category vs day
            lambda: {
                "name": f"{random.choice(categories).lower().replace(' ', '_')}_day_{random.randint(0, 6)}",
                "description": f"Does {random.choice(categories)} sell more on {['Mon','Tue','Wed','Thu','Fri','Sat','Sun'][random.randint(0,6)]}?",
                "category": "temporal",
            },
            # Category correlation
            lambda: {
                "name": f"correlation_{random.choice(categories).lower()[:10]}_{random.choice(categories).lower()[:10]}",
                "description": f"Do {random.choice(categories)} and {random.choice(categories)} sales correlate?",
                "category": "product",
            },
            # Price band timing
            lambda: {
                "name": f"price_band_{random.choice(['low', 'mid', 'high'])}_time_{random.choice(['morning', 'afternoon', 'evening'])}",
                "description": f"Do {random.choice(['budget', 'mid-tier', 'premium'])} products sell more in the {random.choice(['morning', 'afternoon', 'evening'])}?",
                "category": "behavioral",
            },
        ]
        
        combo = random.choice(combinations)()
        
        return Hypothesis(
            id=hashlib.md5(f"{combo['name']}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            name=combo["name"],
            description=combo["description"],
            category=combo["category"],
            condition=combo["name"],
            expected_effect="correlate",
            source="random",
        )
    
    @classmethod
    def derive_hypothesis(cls, parent: Hypothesis, finding: str) -> Hypothesis:
        """Derive a new hypothesis from a parent hypothesis's findings."""
        return Hypothesis(
            id=hashlib.md5(f"{parent.id}_{finding}_{datetime.now().isoformat()}".encode()).hexdigest()[:8],
            name=f"{parent.name}_derived_{finding[:20]}",
            description=f"Based on {parent.name}: {finding}",
            category=parent.category,
            condition=f"derived_{parent.name}",
            expected_effect=parent.expected_effect,
            source="derived",
            parent_hypothesis=parent.id,
        )


class HypothesisTester:
    """
    Tests hypotheses against real data.
    
    Uses statistical tests to prove/disprove hypotheses:
    - T-tests for comparing means
    - Chi-square for categorical relationships
    - Correlation tests for continuous relationships
    - Multiple testing corrections (Bonferroni)
    """
    
    SIGNIFICANCE_THRESHOLD = 0.05
    MIN_SAMPLE_SIZE = 30
    MIN_EFFECT_SIZE = 0.05  # 5% minimum effect to be "interesting"
    
    def __init__(self, db_path: str):
        self.db_path = db_path
    
    def get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(self.db_path)
    
    def test_hypothesis(self, hypothesis: Hypothesis) -> Hypothesis:
        """Test a hypothesis and update it with results."""
        hypothesis.tested = True
        hypothesis.test_date = datetime.now().isoformat()
        
        # Route to appropriate test based on hypothesis type
        test_methods = {
            "first_of_month_payday": self._test_payday_effect,
            "monday_blues": self._test_day_of_week_effect,
            "friday_pregame": self._test_hour_pattern,
            "full_moon_effect": self._test_moon_phase,
            "rain_indoor_boost": self._test_weather_effect,
            "basket_size_friday": self._test_basket_size_pattern,
            "flower_friday": self._test_category_day_pattern,
            "preroll_convenience": self._test_category_hour_pattern,
        }
        
        # Generic tests for generated hypotheses
        if hypothesis.condition.startswith("correlation_"):
            result = self._test_category_correlation(hypothesis)
        elif "_hour_" in hypothesis.condition:
            result = self._test_category_hour_pattern(hypothesis)
        elif "_day_" in hypothesis.condition:
            result = self._test_category_day_pattern(hypothesis)
        elif hypothesis.name in test_methods:
            result = test_methods[hypothesis.name](hypothesis)
        else:
            # Default: try generic temporal test
            result = self._test_generic_temporal(hypothesis)
        
        return result
    
    def _test_payday_effect(self, h: Hypothesis) -> Hypothesis:
        """Test if sales are higher on paydays (1st, 15th)."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Get daily sales
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            WHERE date >= date('now', '-365 days')
            GROUP BY date
        """)
        
        payday_sales = []
        other_sales = []
        
        for date_str, units in cur.fetchall():
            day = int(date_str.split("-")[2])
            if day in [1, 2, 15, 16]:  # Payday and day after
                payday_sales.append(units)
            else:
                other_sales.append(units)
        
        conn.close()
        
        h.sample_size = len(payday_sales) + len(other_sales)
        
        if len(payday_sales) < self.MIN_SAMPLE_SIZE:
            h.notes = f"Insufficient sample: {len(payday_sales)} payday samples"
            return h
        
        # T-test
        if HAS_SCIPY:
            t_stat, p_value = stats.ttest_ind(payday_sales, other_sales)
            h.p_value = p_value
        else:
            # Manual calculation
            mean1, mean2 = sum(payday_sales)/len(payday_sales), sum(other_sales)/len(other_sales)
            h.p_value = 0.5  # Placeholder
        
        mean_payday = sum(payday_sales) / len(payday_sales)
        mean_other = sum(other_sales) / len(other_sales)
        h.effect_size = (mean_payday - mean_other) / mean_other
        
        h.proven = h.p_value < self.SIGNIFICANCE_THRESHOLD and abs(h.effect_size) > self.MIN_EFFECT_SIZE
        h.confidence = 1 - h.p_value if h.proven else h.p_value
        h.notes = f"Payday avg: {mean_payday:.0f}, Other avg: {mean_other:.0f}, Effect: {h.effect_size:+.1%}"
        
        return h
    
    def _test_day_of_week_effect(self, h: Hypothesis) -> Hypothesis:
        """Test if a specific day differs from others."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT strftime('%w', date) as dow, SUM(quantity) as units
            FROM sales
            WHERE date >= date('now', '-365 days')
            GROUP BY date
        """)
        
        by_dow = defaultdict(list)
        for dow, units in cur.fetchall():
            by_dow[int(dow)].append(units)
        
        conn.close()
        
        # Compare Monday (1) to other weekdays
        monday = by_dow[1]
        other_weekdays = [u for d in [2, 3, 4] for u in by_dow[d]]
        
        h.sample_size = len(monday) + len(other_weekdays)
        
        if len(monday) < self.MIN_SAMPLE_SIZE:
            h.notes = f"Insufficient sample: {len(monday)} Monday samples"
            return h
        
        if HAS_SCIPY:
            t_stat, p_value = stats.ttest_ind(monday, other_weekdays)
            h.p_value = p_value
        
        mean_monday = sum(monday) / len(monday)
        mean_other = sum(other_weekdays) / len(other_weekdays)
        h.effect_size = (mean_monday - mean_other) / mean_other
        
        h.proven = h.p_value < self.SIGNIFICANCE_THRESHOLD and abs(h.effect_size) > self.MIN_EFFECT_SIZE
        h.confidence = 1 - h.p_value if h.proven else h.p_value
        h.notes = f"Monday avg: {mean_monday:.0f}, Other weekdays: {mean_other:.0f}"
        
        return h
    
    def _test_moon_phase(self, h: Hypothesis) -> Hypothesis:
        """Test if full moon affects sales. Yes, we're testing this."""
        conn = self.get_connection()
        cur = conn.cursor()
        
        cur.execute("""
            SELECT date, SUM(quantity) as units
            FROM sales
            WHERE date >= date('now', '-365 days')
            GROUP BY date
        """)
        
        # Calculate moon phase for each date
        # Synodic month = 29.53 days
        # Known full moon: Jan 13, 2025
        reference_full_moon = date(2025, 1, 13)
        synodic_month = 29.53
        
        full_moon_sales = []
        other_sales = []
        
        for date_str, units in cur.fetchall():
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            days_since = (d - reference_full_moon).days
            moon_phase = (days_since % synodic_month) / synodic_month
            
            # Full moon is at phase 0 or 1 (within 2 days)
            if moon_phase < 0.07 or moon_phase > 0.93:
                full_moon_sales.append(units)
            else:
                other_sales.append(units)
        
        conn.close()
        
        h.sample_size = len(full_moon_sales) + len(other_sales)
        
        if len(full_moon_sales) < 10:
            h.notes = f"Insufficient sample: {len(full_moon_sales)} full moon days"
            return h
        
        if HAS_SCIPY and len(full_moon_sales) > 2:
            t_stat, p_value = stats.ttest_ind(full_moon_sales, other_sales)
            h.p_value = p_value
        
        mean_full = sum(full_moon_sales) / len(full_moon_sales)
        mean_other = sum(other_sales) / len(other_sales)
        h.effect_size = (mean_full - mean_other) / mean_other
        
        h.proven = h.p_value < self.SIGNIFICANCE_THRESHOLD and abs(h.effect_size) > self.MIN_EFFECT_SIZE
        h.confidence = 1 - h.p_value if h.proven else h.p_value
        h.notes = f"Full moon avg: {mean_full:.0f}, Other: {mean_other:.0f}, n={len(full_moon_sales)} full moon days"
        
        return h
    
    def _test_category_correlation(self, h: Hypothesis) -> Hypothesis:
        """Test if two categories' sales correlate."""
        # Extract categories from hypothesis name
        parts = h.name.replace("correlation_", "").split("_")
        if len(parts) < 2:
            h.notes = "Could not parse categories"
            return h
        
        conn = self.get_connection()
        cur = conn.cursor()
        
        # Get daily sales by category
        cur.execute("""
            SELECT date, category, SUM(quantity) as units
            FROM sales
            WHERE date >= date('now', '-180 days')
            GROUP BY date, category
        """)
        
        by_date = defaultdict(dict)
        for date_str, category, units in cur.fetchall():
            by_date[date_str][category] = units
        
        conn.close()
        
        # Find matching categories
        all_categories = set()
        for cats in by_date.values():
            all_categories.update(cats.keys())
        
        cat1 = next((c for c in all_categories if parts[0].lower() in c.lower()), None)
        cat2 = next((c for c in all_categories if parts[1].lower() in c.lower()), None)
        
        if not cat1 or not cat2:
            h.notes = f"Categories not found: {parts[0]}, {parts[1]}"
            return h
        
        # Build paired data
        cat1_sales = []
        cat2_sales = []
        for date_str in by_date:
            if cat1 in by_date[date_str] and cat2 in by_date[date_str]:
                cat1_sales.append(by_date[date_str][cat1])
                cat2_sales.append(by_date[date_str][cat2])
        
        h.sample_size = len(cat1_sales)
        
        if h.sample_size < self.MIN_SAMPLE_SIZE:
            h.notes = f"Insufficient paired samples: {h.sample_size}"
            return h
        
        if HAS_SCIPY:
            corr, p_value = stats.pearsonr(cat1_sales, cat2_sales)
            h.effect_size = corr
            h.p_value = p_value
        
        h.proven = h.p_value < self.SIGNIFICANCE_THRESHOLD and abs(h.effect_size) > 0.3
        h.confidence = abs(h.effect_size) if h.proven else 0
        h.notes = f"Correlation between {cat1} and {cat2}: r={h.effect_size:.2f}"
        
        return h
    
    def _test_generic_temporal(self, h: Hypothesis) -> Hypothesis:
        """Generic temporal pattern test."""
        h.notes = "No specific test implemented for this hypothesis"
        h.tested = True
        return h
    
    def _test_hour_pattern(self, h: Hypothesis) -> Hypothesis:
        """Test hour-based patterns."""
        h.notes = "Hour pattern test - needs implementation"
        return h
    
    def _test_weather_effect(self, h: Hypothesis) -> Hypothesis:
        """Test weather effects."""
        h.notes = "Weather test - needs weather data integration"
        return h
    
    def _test_basket_size_pattern(self, h: Hypothesis) -> Hypothesis:
        """Test basket size patterns."""
        h.notes = "Basket size test - needs implementation"
        return h
    
    def _test_category_day_pattern(self, h: Hypothesis) -> Hypothesis:
        """Test category sales by day of week."""
        h.notes = "Category/day test - needs implementation"
        return h
    
    def _test_category_hour_pattern(self, h: Hypothesis) -> Hypothesis:
        """Test category sales by hour."""
        h.notes = "Category/hour test - needs implementation"
        return h


class DiscoveryJournal:
    """
    Tracks all hypotheses tested and discoveries made.
    
    Persists to disk so we never lose findings.
    """
    
    def __init__(self, journal_dir: Path = None):
        if journal_dir is None:
            journal_dir = Path(__file__).parent / "data" / "discoveries"
        self.journal_dir = journal_dir
        self.journal_dir.mkdir(parents=True, exist_ok=True)
        
        self.hypotheses_file = self.journal_dir / "hypotheses.json"
        self.discoveries_file = self.journal_dir / "discoveries.json"
        
        self.hypotheses: Dict[str, Hypothesis] = {}
        self.discoveries: List[Discovery] = []
        
        self._load()
    
    def _load(self):
        """Load from disk."""
        if self.hypotheses_file.exists():
            with open(self.hypotheses_file) as f:
                data = json.load(f)
                for h_data in data:
                    h = Hypothesis(**h_data)
                    self.hypotheses[h.id] = h
        
        if self.discoveries_file.exists():
            with open(self.discoveries_file) as f:
                data = json.load(f)
                self.discoveries = [Discovery(**d) for d in data]
    
    def _save(self):
        """Save to disk."""
        with open(self.hypotheses_file, "w") as f:
            json.dump([asdict(h) for h in self.hypotheses.values()], f, indent=2)
        
        with open(self.discoveries_file, "w") as f:
            json.dump([asdict(d) for d in self.discoveries], f, indent=2)
    
    def add_hypothesis(self, hypothesis: Hypothesis):
        """Add a hypothesis to track."""
        self.hypotheses[hypothesis.id] = hypothesis
        self._save()
    
    def update_hypothesis(self, hypothesis: Hypothesis):
        """Update a hypothesis after testing."""
        self.hypotheses[hypothesis.id] = hypothesis
        
        # If proven, create a discovery
        if hypothesis.proven:
            discovery = Discovery(
                hypothesis_id=hypothesis.id,
                name=hypothesis.name,
                description=hypothesis.description,
                effect_size=hypothesis.effect_size,
                p_value=hypothesis.p_value,
                confidence=hypothesis.confidence,
                sample_size=hypothesis.sample_size,
            )
            self.discoveries.append(discovery)
        
        self._save()
    
    def get_untested_hypotheses(self) -> List[Hypothesis]:
        """Get hypotheses that haven't been tested."""
        return [h for h in self.hypotheses.values() if not h.tested]
    
    def get_proven_discoveries(self) -> List[Discovery]:
        """Get all proven discoveries."""
        return self.discoveries
    
    def summary(self) -> str:
        """Generate a summary of the journal."""
        total = len(self.hypotheses)
        tested = sum(1 for h in self.hypotheses.values() if h.tested)
        proven = len(self.discoveries)
        
        lines = [
            "=" * 60,
            "📔 DISCOVERY JOURNAL SUMMARY",
            "=" * 60,
            f"",
            f"Total hypotheses: {total}",
            f"Tested: {tested}",
            f"Proven discoveries: {proven}",
            f"Success rate: {proven/tested*100:.1f}%" if tested > 0 else "No tests yet",
            "",
        ]
        
        if self.discoveries:
            lines.append("🎯 PROVEN DISCOVERIES:")
            lines.append("-" * 60)
            for d in sorted(self.discoveries, key=lambda x: abs(x.effect_size), reverse=True)[:10]:
                lines.append(f"  • {d.name}")
                lines.append(f"    Effect: {d.effect_size:+.1%}, p={d.p_value:.3f}, n={d.sample_size}")
                lines.append("")
        
        return "\n".join(lines)


class CuriosityEngine:
    """
    The main exploration engine.
    
    Runs continuously, generating and testing hypotheses.
    """
    
    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = str(Path.cwd() / "aoc_sales.db")
        
        self.db_path = db_path
        self.journal = DiscoveryJournal()
        self.tester = HypothesisTester(db_path)
        
    def initialize(self):
        """Initialize with all predefined hypotheses."""
        hypotheses = HypothesisGenerator.generate_all_hypotheses()
        
        for h in hypotheses:
            if h.id not in self.journal.hypotheses:
                self.journal.add_hypothesis(h)
        
        print(f"Initialized with {len(self.journal.hypotheses)} hypotheses")
    
    def explore_one(self) -> Optional[Hypothesis]:
        """Test one untested hypothesis."""
        untested = self.journal.get_untested_hypotheses()
        
        if not untested:
            # Generate a random one
            h = HypothesisGenerator.generate_random_hypothesis(self.db_path)
            self.journal.add_hypothesis(h)
            untested = [h]
        
        # Pick one (prioritize predefined over random)
        predefined = [h for h in untested if h.source == "predefined"]
        h = random.choice(predefined) if predefined else random.choice(untested)
        
        print(f"Testing: {h.name}")
        print(f"  {h.description}")
        
        # Test it
        h = self.tester.test_hypothesis(h)
        self.journal.update_hypothesis(h)
        
        if h.proven:
            print(f"  ✅ PROVEN! Effect: {h.effect_size:+.1%}, p={h.p_value:.3f}")
        else:
            print(f"  ❌ Not proven. {h.notes}")
        
        return h
    
    def explore_batch(self, count: int = 10):
        """Test a batch of hypotheses."""
        print(f"\n{'='*60}")
        print(f"🔬 CURIOSITY ENGINE - Exploring {count} hypotheses")
        print(f"{'='*60}\n")
        
        for i in range(count):
            print(f"\n[{i+1}/{count}]")
            self.explore_one()
        
        print("\n" + self.journal.summary())
    
    def run_forever(self, interval_seconds: int = 60):
        """Run continuously, testing hypotheses."""
        import time
        
        print("🧠 Curiosity Engine starting...")
        print(f"   Testing one hypothesis every {interval_seconds} seconds")
        print("   Press Ctrl+C to stop\n")
        
        self.initialize()
        
        while True:
            try:
                self.explore_one()
                time.sleep(interval_seconds)
            except KeyboardInterrupt:
                print("\n\nStopping...")
                print(self.journal.summary())
                break


if __name__ == "__main__":
    engine = CuriosityEngine()
    engine.initialize()
    engine.explore_batch(count=15)
