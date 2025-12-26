"""
Knowledge Learner - Ingests retail/marketing knowledge and converts it to actionable memories.

Sources:
- Textbooks on retail management
- Marketing best practices articles
- Cannabis retail specific guides
- Academic papers on consumer behavior
- The brain's own validated observations

The learner extracts PRINCIPLES, not just facts.
"""

import json
import re
import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Optional
import requests

from aoc_analytics.brain.memory import BrainMemory, MemoryEntry


@dataclass
class KnowledgeSource:
    """A source of knowledge to learn from."""
    name: str
    source_type: str  # 'textbook', 'article', 'observation', 'principle'
    content: str
    url: Optional[str] = None


# Built-in retail knowledge to bootstrap the brain
RETAIL_FUNDAMENTALS = """
=== RETAIL MARKETING FUNDAMENTALS ===

PRINCIPLE: The 80/20 Rule (Pareto Principle)
80% of sales typically come from 20% of products. Identify and prioritize high-velocity SKUs.
Application: Focus inventory, displays, and promotions on top performers.

PRINCIPLE: Price Anchoring
Customers perceive value relative to reference points. High-price items make medium-price items seem reasonable.
Application: Display premium products prominently even if they don't sell much - they make other products look like deals.

PRINCIPLE: Loss Leader Strategy  
Sell certain products at or below cost to drive traffic, then profit on complementary purchases.
Application: Discount high-visibility categories, profit on accessories and impulse buys.

PRINCIPLE: Day-Part Marketing
Customer behavior and needs vary by time of day. Morning customers differ from evening customers.
Application: Adjust staffing, promotions, and product emphasis by time period.

PRINCIPLE: The Rule of Three
Customers prefer having 3 options (good/better/best). Too many choices cause decision paralysis.
Application: Curate selections. For any category, highlight 3 clear tiers.

PRINCIPLE: Impulse Purchase Zones
Items near checkout get 3-5x more impulse purchases. Small, low-consideration items work best.
Application: Place high-margin accessories and consumables at point of sale.

PRINCIPLE: Scarcity Effect
Limited availability increases perceived value and urgency.
Application: "Limited batch" and "while supplies last" messaging drives action.

PRINCIPLE: Social Proof
People follow what others do. Bestseller tags and "popular" labels increase sales 10-30%.
Application: Highlight top sellers, display review counts, show "X customers bought this today".

PRINCIPLE: Reciprocity
People feel obligated to return favors. Free samples and education create purchase obligation.
Application: Free samples, loyalty rewards, educational budtending increase basket size.

PRINCIPLE: Peak-End Rule
Customers remember the peak moment and the end of an experience most vividly.
Application: End every transaction positively. Resolve problems gracefully.

=== CANNABIS RETAIL SPECIFIC ===

PRINCIPLE: Education Converts
Cannabis customers, especially new ones, buy more when educated about products.
Application: Train staff on effects, terpenes, consumption methods. Informed recommendations increase basket size.

PRINCIPLE: Strain Loyalty is Low
Unlike alcohol brands, cannabis strain loyalty is weak. Customers experiment frequently.
Application: New product launches drive traffic. Rotate featured items frequently.

PRINCIPLE: Payday Correlation
Cannabis purchases strongly correlate with payday cycles (weekly, bi-weekly, monthly).
Application: Time promotions around pay periods. Stock up before month-end.

PRINCIPLE: Weather Impact
Bad weather increases cannabis sales (people staying home). Good weather decreases slightly.
Application: Adjust forecasts for weather. Don't over-discount on rainy days.

PRINCIPLE: Accessory Attachment Rate
Average accessory attachment (papers, grinders, lighters) should be 15-25% of transactions.
Application: Train staff to suggest accessories. Track attachment rate as KPI.

=== PRICING PSYCHOLOGY ===

PRINCIPLE: Charm Pricing
Prices ending in .99 or .95 are perceived as significantly lower than round numbers.
Application: $24.99 feels much cheaper than $25.00 despite 1 cent difference.

PRINCIPLE: Bundle Psychology
Bundles feel like deals even when mathematically equivalent. "Buy 3 for $30" beats "$10 each".
Application: Create product bundles at small discount. Increases units per transaction.

PRINCIPLE: Reference Price Effect
Showing original price next to sale price increases conversion, even if original was inflated.
Application: Always show "was/now" pricing on promotions.
"""

CANNABIS_SEASONALITY = """
=== CANNABIS RETAIL SEASONALITY ===

PATTERN: 4/20 (April 20)
The biggest cannabis retail day. Sales typically 2-3x normal. Plan 4-6 weeks ahead.
Preparation: Extra inventory, extended hours, special promotions, additional staff.

PATTERN: Green Wednesday
Day before US Thanksgiving. Strong sales day as people stock up for long weekend.

PATTERN: Summer Slowdown (June-August)
Slight decrease as people spend more time outdoors. Festival season can offset.

PATTERN: Back to School (September)
Sales uptick as routines normalize. Student-heavy markets see significant increase.

PATTERN: Holiday Season (Nov-Dec)
Gift-giving drives accessory sales. Edibles popular for holiday parties.

PATTERN: New Year's Eve
Strong pre-roll and party-pack sales. Stock up on shareable formats.

PATTERN: January Slowdown
Post-holiday budget tightening. Value offerings perform better.

PATTERN: Tax Season (Feb-April)
Refunds drive discretionary spending. Opportunity for premium product push.

PATTERN: Harvest Season (Sept-Nov)
Fresh outdoor crop hits market. Prices often drop. Quality peaks.

PATTERN: Tourism Cycles
Tourist-heavy locations see summer peaks. Know your local tourist season.
"""


class KnowledgeLearner:
    """
    Learns from knowledge sources and converts to actionable memories.
    
    The learner doesn't just store facts - it extracts:
    1. PRINCIPLES - Universal rules that apply to our business
    2. PATTERNS - Recurring phenomena to watch for
    3. TACTICS - Specific actions that work
    4. METRICS - What to measure and target values
    """
    
    def __init__(self, memory: BrainMemory, ollama_url: str = "http://localhost:11434"):
        self.memory = memory
        self.ollama_url = ollama_url
    
    def bootstrap_fundamentals(self):
        """Load built-in retail knowledge into memory."""
        print("📚 Bootstrapping retail fundamentals...")
        
        # Parse and store fundamentals
        self._learn_from_text(
            RETAIL_FUNDAMENTALS,
            source="retail_fundamentals_v1",
            source_type="textbook"
        )
        
        self._learn_from_text(
            CANNABIS_SEASONALITY,
            source="cannabis_seasonality_v1", 
            source_type="textbook"
        )
        
        print(f"✓ Loaded fundamental knowledge")
    
    def _learn_from_text(self, text: str, source: str, source_type: str):
        """Extract principles and patterns from text."""
        # Check if already processed
        content_hash = hashlib.md5(text.encode()).hexdigest()
        existing = self.memory.conn.execute(
            "SELECT id FROM learning_sessions WHERE content_hash = ?",
            (content_hash,)
        ).fetchone()
        
        if existing:
            print(f"  (already learned from {source})")
            return
        
        # Parse structured knowledge
        principles = self._extract_principles(text)
        patterns = self._extract_patterns(text)
        
        memories_created = 0
        
        # Store principles
        for principle in principles:
            entry = MemoryEntry(
                id=self.memory._generate_id(principle['name'], "prin"),
                category="knowledge",
                content=f"PRINCIPLE: {principle['name']}\n{principle['description']}\nAPPLICATION: {principle['application']}",
                source=source,
                confidence=0.8,  # Textbook knowledge starts high
                metadata={
                    "type": "principle",
                    "name": principle['name'],
                    "application": principle['application']
                }
            )
            self.memory.store_memory(entry)
            memories_created += 1
        
        # Store patterns
        for pattern in patterns:
            entry = MemoryEntry(
                id=self.memory._generate_id(pattern['name'], "patt"),
                category="knowledge",
                content=f"PATTERN: {pattern['name']}\n{pattern['description']}",
                source=source,
                confidence=0.7,
                metadata={
                    "type": "pattern",
                    "name": pattern['name']
                }
            )
            self.memory.store_memory(entry)
            memories_created += 1
        
        # Record learning session
        self.memory.conn.execute("""
            INSERT INTO learning_sessions 
            (source_type, source_name, content_hash, memories_created, processed_at)
            VALUES (?, ?, ?, ?, ?)
        """, (source_type, source, content_hash, memories_created, datetime.now().isoformat()))
        self.memory.conn.commit()
        
        print(f"  Extracted {len(principles)} principles, {len(patterns)} patterns from {source}")
    
    def _extract_principles(self, text: str) -> list[dict]:
        """Extract PRINCIPLE blocks from text."""
        principles = []
        
        # Pattern: PRINCIPLE: Name\nDescription\nApplication: ...
        pattern = r'PRINCIPLE:\s*([^\n]+)\n([^A]+?)(?:Application:|PRINCIPLE:|PATTERN:|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for name, description in matches:
            # Try to extract application
            app_match = re.search(r'Application:\s*([^\n]+)', description + text[text.find(name):text.find(name)+500])
            application = app_match.group(1).strip() if app_match else ""
            
            principles.append({
                'name': name.strip(),
                'description': description.strip(),
                'application': application
            })
        
        return principles
    
    def _extract_patterns(self, text: str) -> list[dict]:
        """Extract PATTERN blocks from text."""
        patterns = []
        
        # Pattern: PATTERN: Name\nDescription
        pattern = r'PATTERN:\s*([^\n]+)\n([^P]+?)(?:PATTERN:|PRINCIPLE:|$)'
        matches = re.findall(pattern, text, re.IGNORECASE | re.DOTALL)
        
        for name, description in matches:
            patterns.append({
                'name': name.strip(),
                'description': description.strip()
            })
        
        return patterns
    
    def learn_from_observation(self, observation: str, data_evidence: dict):
        """
        Learn from a pattern observed in our own data.
        
        Unlike textbook knowledge, observations start with lower confidence
        and must be validated through hypothesis testing.
        """
        # Use LLM to extract the principle from the observation
        prompt = f"""Based on this observation from our sales data, extract the underlying principle.

OBSERVATION: {observation}
DATA EVIDENCE: {json.dumps(data_evidence, indent=2)}

Respond in this exact format:
PRINCIPLE_NAME: [short name for this principle]
DESCRIPTION: [what the principle is]
CONDITIONS: [when this applies]
TESTABLE_PREDICTION: [a specific, measurable prediction]
"""
        
        response = requests.post(
            f"{self.ollama_url}/api/generate",
            json={
                "model": "llama3.2",
                "prompt": prompt,
                "stream": False,
                "options": {"temperature": 0.2}
            },
            timeout=30
        )
        
        if response.status_code != 200:
            print(f"LLM error: {response.status_code}")
            return
        
        result = response.json()['response']
        
        # Parse response
        parsed = self._parse_llm_principle(result)
        if not parsed:
            return
        
        # Store as observation (lower confidence than textbook)
        entry = MemoryEntry(
            id=self.memory._generate_id(observation, "obs"),
            category="observation",
            content=f"OBSERVED: {parsed['name']}\n{parsed['description']}\nCONDITIONS: {parsed['conditions']}",
            source="data_observation",
            confidence=0.4,  # Observations start lower
            metadata={
                "type": "observation",
                "name": parsed['name'],
                "evidence": data_evidence,
                "testable_prediction": parsed['prediction']
            }
        )
        self.memory.store_memory(entry)
        
        print(f"📝 Learned observation: {parsed['name']}")
        return entry
    
    def _parse_llm_principle(self, text: str) -> Optional[dict]:
        """Parse LLM response into structured principle."""
        try:
            name_match = re.search(r'PRINCIPLE_NAME:\s*(.+?)(?:\n|$)', text)
            desc_match = re.search(r'DESCRIPTION:\s*(.+?)(?:CONDITIONS:|$)', text, re.DOTALL)
            cond_match = re.search(r'CONDITIONS:\s*(.+?)(?:TESTABLE_PREDICTION:|$)', text, re.DOTALL)
            pred_match = re.search(r'TESTABLE_PREDICTION:\s*(.+?)(?:\n\n|$)', text, re.DOTALL)
            
            if not all([name_match, desc_match]):
                return None
                
            return {
                'name': name_match.group(1).strip(),
                'description': desc_match.group(1).strip(),
                'conditions': cond_match.group(1).strip() if cond_match else "",
                'prediction': pred_match.group(1).strip() if pred_match else ""
            }
        except Exception:
            return None
    
    def learn_from_url(self, url: str):
        """
        Learn from a web article or document.
        
        Uses LLM to extract retail principles from arbitrary content.
        """
        # This would fetch and process web content
        # For now, placeholder
        print(f"📖 Learning from {url}...")
        print("  (Web learning not yet implemented)")
    
    def get_relevant_knowledge(self, context: str, limit: int = 5) -> list[MemoryEntry]:
        """Retrieve knowledge relevant to a given context."""
        return self.memory.recall(context, category="knowledge", limit=limit)
    
    def get_observations(self, limit: int = 10) -> list[MemoryEntry]:
        """Get all observations for review."""
        rows = self.memory.conn.execute("""
            SELECT * FROM memories 
            WHERE category = 'observation'
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        return [self.memory._row_to_memory(row) for row in rows]
