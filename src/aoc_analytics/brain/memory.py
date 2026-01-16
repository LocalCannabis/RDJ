"""Brain Memory - Long-term knowledge storage with semantic retrieval.

The brain needs to remember:
1. Learned knowledge (retail best practices, marketing principles)
2. Observations (patterns noticed in data)
3. Hypotheses (theories about what drives sales)
4. Outcomes (did predictions come true?)
5. Refined beliefs (updated understanding)
"""

import json
from dataclasses import dataclass, field, asdict
from datetime import datetime
from typing import Optional
from pathlib import Path
import hashlib

import numpy as np

from aoc_analytics.core.db_adapter import get_connection


@dataclass
class MemoryEntry:
    """A single piece of knowledge or observation."""
    id: str
    category: str  # 'knowledge', 'observation', 'hypothesis', 'outcome', 'belief'
    content: str
    source: str  # where this came from
    confidence: float  # 0-1, how confident are we in this
    embedding: Optional[list] = None
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_accessed: str = field(default_factory=lambda: datetime.now().isoformat())
    access_count: int = 0
    reinforcement_score: float = 0.0  # positive if validated, negative if disproven


@dataclass 
class Hypothesis:
    """A testable theory about what drives sales."""
    id: str
    statement: str  # "Friday sales increase when weather is rainy"
    conditions: dict  # {"day_of_week": 5, "weather": "rain"}
    prediction: str  # "sales_increase_pct > 10"
    confidence: float
    times_tested: int = 0
    times_correct: int = 0
    created_at: str = field(default_factory=lambda: datetime.now().isoformat())
    last_tested: Optional[str] = None
    status: str = "untested"  # untested, testing, validated, disproven, inconclusive
    
    @property
    def accuracy(self) -> float:
        if self.times_tested == 0:
            return 0.0
        return self.times_correct / self.times_tested
    
    @property
    def is_reliable(self) -> bool:
        """Need at least 5 tests with >70% accuracy to trust."""
        return self.times_tested >= 5 and self.accuracy >= 0.7


class BrainMemory:
    """
    Persistent memory store for the AI brain.
    
    Uses SQLite for durability + numpy for embeddings.
    Implements memory consolidation (forget unimportant things).
    """
    
    def __init__(self, db_path: str = "brain_memory.db"):
        self.db_path = Path(db_path)
        self.conn = get_connection(str(self.db_path))
        self._init_schema()
        self._embedding_model = None
    
    def _init_schema(self):
        """Create tables for memory storage."""
        self.conn.executescript("""
            -- General memories (knowledge, observations, beliefs)
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                category TEXT NOT NULL,
                content TEXT NOT NULL,
                source TEXT,
                confidence REAL DEFAULT 0.5,
                embedding BLOB,
                metadata TEXT,  -- JSON
                created_at TEXT,
                last_accessed TEXT,
                access_count INTEGER DEFAULT 0,
                reinforcement_score REAL DEFAULT 0.0
            );
            
            -- Hypotheses (testable theories)
            CREATE TABLE IF NOT EXISTS hypotheses (
                id TEXT PRIMARY KEY,
                statement TEXT NOT NULL,
                conditions TEXT,  -- JSON
                prediction TEXT,
                confidence REAL DEFAULT 0.5,
                times_tested INTEGER DEFAULT 0,
                times_correct INTEGER DEFAULT 0,
                created_at TEXT,
                last_tested TEXT,
                status TEXT DEFAULT 'untested'
            );
            
            -- Hypothesis test results
            CREATE TABLE IF NOT EXISTS hypothesis_tests (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                hypothesis_id TEXT,
                test_date TEXT,
                conditions_met INTEGER,  -- bool
                prediction_correct INTEGER,  -- bool
                actual_outcome TEXT,
                notes TEXT,
                FOREIGN KEY (hypothesis_id) REFERENCES hypotheses(id)
            );
            
            -- Learning sessions (what sources were processed)
            CREATE TABLE IF NOT EXISTS learning_sessions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                source_type TEXT,  -- 'textbook', 'article', 'observation'
                source_name TEXT,
                content_hash TEXT,
                memories_created INTEGER,
                processed_at TEXT
            );
            
            CREATE INDEX IF NOT EXISTS idx_memories_category ON memories(category);
            CREATE INDEX IF NOT EXISTS idx_memories_confidence ON memories(confidence);
            CREATE INDEX IF NOT EXISTS idx_hypotheses_status ON hypotheses(status);
        """)
        self.conn.commit()
    
    def _get_embedding_model(self):
        """Lazy load embedding model for semantic search."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                import torch
                device = "cuda" if torch.cuda.is_available() else "cpu"
                self._embedding_model = SentenceTransformer(
                    'all-MiniLM-L6-v2', 
                    device=device
                )
            except ImportError:
                return None
        return self._embedding_model
    
    def _compute_embedding(self, text: str) -> Optional[np.ndarray]:
        """Compute embedding for semantic search."""
        model = self._get_embedding_model()
        if model is None:
            return None
        return model.encode(text, convert_to_numpy=True)
    
    def _generate_id(self, content: str, prefix: str = "mem") -> str:
        """Generate unique ID for content."""
        hash_val = hashlib.md5(content.encode()).hexdigest()[:12]
        return f"{prefix}_{hash_val}"
    
    # === MEMORY OPERATIONS ===
    
    def store_memory(self, entry: MemoryEntry) -> str:
        """Store a memory entry."""
        # Compute embedding if not provided
        if entry.embedding is None:
            emb = self._compute_embedding(entry.content)
            entry.embedding = emb.tolist() if emb is not None else None
        
        # Store in database
        self.conn.execute("""
            INSERT OR REPLACE INTO memories 
            (id, category, content, source, confidence, embedding, metadata,
             created_at, last_accessed, access_count, reinforcement_score)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            entry.id,
            entry.category,
            entry.content,
            entry.source,
            entry.confidence,
            json.dumps(entry.embedding) if entry.embedding else None,
            json.dumps(entry.metadata),
            entry.created_at,
            entry.last_accessed,
            entry.access_count,
            entry.reinforcement_score,
        ))
        self.conn.commit()
        return entry.id
    
    def recall(self, query: str, category: Optional[str] = None, 
               limit: int = 10) -> list[MemoryEntry]:
        """
        Recall memories similar to query using semantic search.
        Updates access patterns for memory consolidation.
        """
        query_emb = self._compute_embedding(query)
        
        # Fetch candidates
        if category:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE category = ? AND embedding IS NOT NULL",
                (category,)
            ).fetchall()
        else:
            rows = self.conn.execute(
                "SELECT * FROM memories WHERE embedding IS NOT NULL"
            ).fetchall()
        
        if not rows or query_emb is None:
            # Fall back to text search
            return self._text_search(query, category, limit)
        
        # Compute similarities
        results = []
        for row in rows:
            emb = np.array(json.loads(row[5])) if row[5] else None
            if emb is not None:
                similarity = np.dot(query_emb, emb) / (
                    np.linalg.norm(query_emb) * np.linalg.norm(emb) + 1e-8
                )
                results.append((similarity, row))
        
        # Sort by similarity
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Convert to MemoryEntry objects
        memories = []
        for sim, row in results[:limit]:
            entry = MemoryEntry(
                id=row[0],
                category=row[1],
                content=row[2],
                source=row[3],
                confidence=row[4],
                embedding=json.loads(row[5]) if row[5] else None,
                metadata=json.loads(row[6]) if row[6] else {},
                created_at=row[7],
                last_accessed=row[8],
                access_count=row[9],
                reinforcement_score=row[10],
            )
            # Update access pattern
            self._touch_memory(entry.id)
            memories.append(entry)
        
        return memories
    
    def _text_search(self, query: str, category: Optional[str], 
                     limit: int) -> list[MemoryEntry]:
        """Fallback text-based search."""
        words = query.lower().split()
        where = " AND ".join([f"LOWER(content) LIKE '%{w}%'" for w in words[:5]])
        if category:
            where = f"category = '{category}' AND ({where})"
        
        rows = self.conn.execute(f"""
            SELECT * FROM memories WHERE {where}
            ORDER BY confidence DESC, reinforcement_score DESC
            LIMIT ?
        """, (limit,)).fetchall()
        
        return [self._row_to_memory(row) for row in rows]
    
    def _row_to_memory(self, row) -> MemoryEntry:
        return MemoryEntry(
            id=row[0],
            category=row[1],
            content=row[2],
            source=row[3],
            confidence=row[4],
            embedding=json.loads(row[5]) if row[5] else None,
            metadata=json.loads(row[6]) if row[6] else {},
            created_at=row[7],
            last_accessed=row[8],
            access_count=row[9],
            reinforcement_score=row[10],
        )
    
    def _touch_memory(self, memory_id: str):
        """Update access time and count (for memory consolidation)."""
        self.conn.execute("""
            UPDATE memories 
            SET last_accessed = ?, access_count = access_count + 1
            WHERE id = ?
        """, (datetime.now().isoformat(), memory_id))
        self.conn.commit()
    
    def reinforce(self, memory_id: str, delta: float):
        """
        Reinforce or weaken a memory based on validation.
        Positive delta = memory was useful/correct
        Negative delta = memory was wrong/unhelpful
        """
        self.conn.execute("""
            UPDATE memories 
            SET reinforcement_score = reinforcement_score + ?
            WHERE id = ?
        """, (delta, memory_id))
        self.conn.commit()
    
    # === HYPOTHESIS OPERATIONS ===
    
    def store_hypothesis(self, h: Hypothesis) -> str:
        """Store a hypothesis for testing."""
        self.conn.execute("""
            INSERT OR REPLACE INTO hypotheses
            (id, statement, conditions, prediction, confidence,
             times_tested, times_correct, created_at, last_tested, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            h.id,
            h.statement,
            json.dumps(h.conditions),
            h.prediction,
            h.confidence,
            h.times_tested,
            h.times_correct,
            h.created_at,
            h.last_tested,
            h.status,
        ))
        self.conn.commit()
        return h.id
    
    def get_testable_hypotheses(self, conditions: dict) -> list[Hypothesis]:
        """Get hypotheses that match current conditions and can be tested."""
        rows = self.conn.execute("""
            SELECT * FROM hypotheses 
            WHERE status IN ('untested', 'testing', 'validated')
        """).fetchall()
        
        matching = []
        for row in rows:
            h_conditions = json.loads(row[2]) if row[2] else {}
            # Check if current conditions match hypothesis conditions
            if self._conditions_match(h_conditions, conditions):
                matching.append(Hypothesis(
                    id=row[0],
                    statement=row[1],
                    conditions=h_conditions,
                    prediction=row[3],
                    confidence=row[4],
                    times_tested=row[5],
                    times_correct=row[6],
                    created_at=row[7],
                    last_tested=row[8],
                    status=row[9],
                ))
        return matching
    
    def _conditions_match(self, required: dict, actual: dict) -> bool:
        """Check if actual conditions satisfy hypothesis requirements."""
        for key, value in required.items():
            if key not in actual:
                return False
            if isinstance(value, list):
                if actual[key] not in value:
                    return False
            elif actual[key] != value:
                return False
        return True
    
    def record_hypothesis_test(self, hypothesis_id: str, 
                               correct: bool, actual_outcome: str):
        """Record the result of testing a hypothesis."""
        now = datetime.now().isoformat()
        
        # Record the test
        self.conn.execute("""
            INSERT INTO hypothesis_tests 
            (hypothesis_id, test_date, prediction_correct, actual_outcome)
            VALUES (?, ?, ?, ?)
        """, (hypothesis_id, now, int(correct), actual_outcome))
        
        # Update hypothesis stats
        self.conn.execute("""
            UPDATE hypotheses 
            SET times_tested = times_tested + 1,
                times_correct = times_correct + ?,
                last_tested = ?,
                confidence = (times_correct + ?) * 1.0 / (times_tested + 1),
                status = CASE 
                    WHEN times_tested + 1 >= 5 AND (times_correct + ?) * 1.0 / (times_tested + 1) >= 0.7 
                        THEN 'validated'
                    WHEN times_tested + 1 >= 5 AND (times_correct + ?) * 1.0 / (times_tested + 1) < 0.3 
                        THEN 'disproven'
                    ELSE 'testing'
                END
            WHERE id = ?
        """, (int(correct), now, int(correct), int(correct), int(correct), hypothesis_id))
        
        self.conn.commit()
    
    def get_validated_hypotheses(self) -> list[Hypothesis]:
        """Get all hypotheses that have been validated."""
        rows = self.conn.execute("""
            SELECT * FROM hypotheses WHERE status = 'validated'
            ORDER BY confidence DESC
        """).fetchall()
        
        return [Hypothesis(
            id=row[0],
            statement=row[1],
            conditions=json.loads(row[2]) if row[2] else {},
            prediction=row[3],
            confidence=row[4],
            times_tested=row[5],
            times_correct=row[6],
            created_at=row[7],
            last_tested=row[8],
            status=row[9],
        ) for row in rows]
    
    # === MEMORY CONSOLIDATION ===
    
    def consolidate(self, keep_top_n: int = 1000):
        """
        Memory consolidation - forget unimportant memories.
        
        Keeps memories that are:
        - Frequently accessed
        - Recently accessed  
        - Highly reinforced
        - High confidence
        """
        # Score all memories
        self.conn.execute("""
            DELETE FROM memories 
            WHERE id NOT IN (
                SELECT id FROM memories
                ORDER BY 
                    (access_count * 0.3 + 
                     reinforcement_score * 0.3 + 
                     confidence * 0.2 +
                     JULIANDAY('now') - JULIANDAY(last_accessed) * -0.2
                    ) DESC
                LIMIT ?
            )
        """, (keep_top_n,))
        self.conn.commit()
    
    def get_stats(self) -> dict:
        """Get memory statistics."""
        stats = {}
        
        # Memory counts by category
        rows = self.conn.execute("""
            SELECT category, COUNT(*), AVG(confidence), AVG(reinforcement_score)
            FROM memories GROUP BY category
        """).fetchall()
        stats['memories'] = {
            row[0]: {'count': row[1], 'avg_confidence': row[2], 'avg_reinforcement': row[3]}
            for row in rows
        }
        
        # Hypothesis stats
        rows = self.conn.execute("""
            SELECT status, COUNT(*) FROM hypotheses GROUP BY status
        """).fetchall()
        stats['hypotheses'] = {row[0]: row[1] for row in rows}
        
        return stats
    
    def close(self):
        self.conn.close()
