"""
RAG (Retrieval Augmented Generation) over sales history.

Enables semantic queries like:
- "Find days similar to a rainy Saturday in July"
- "What happened on the last BC Day long weekend?"
- "Show me the best sales days in summer"
"""

import json
import logging
import sqlite3
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SalesDay:
    """A single day's sales data with context."""
    
    date: date
    store: str
    revenue: float
    transactions: int
    
    # Context
    day_of_week: str
    month: str
    is_holiday: bool
    holiday_name: Optional[str]
    is_weekend: bool
    weather_summary: Optional[str]
    
    # Computed
    revenue_vs_avg: float  # percentage above/below average
    
    def to_text(self) -> str:
        """Convert to natural language description for embedding."""
        parts = [
            f"{self.day_of_week}, {self.date.strftime('%B %d, %Y')}",
            f"at {self.store}",
            f"with ${self.revenue:,.0f} revenue ({self.transactions} transactions)",
        ]
        
        if self.is_holiday and self.holiday_name:
            parts.append(f"on {self.holiday_name}")
        elif self.is_weekend:
            parts.append("(weekend)")
        
        if self.weather_summary:
            parts.append(f"Weather: {self.weather_summary}")
        
        if abs(self.revenue_vs_avg) > 10:
            direction = "above" if self.revenue_vs_avg > 0 else "below"
            parts.append(f"({abs(self.revenue_vs_avg):.0f}% {direction} average)")
        
        return " ".join(parts)
    
    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "store": self.store,
            "revenue": self.revenue,
            "transactions": self.transactions,
            "day_of_week": self.day_of_week,
            "month": self.month,
            "is_holiday": self.is_holiday,
            "holiday_name": self.holiday_name,
            "is_weekend": self.is_weekend,
            "weather_summary": self.weather_summary,
            "revenue_vs_avg": self.revenue_vs_avg,
        }


class SalesRAG:
    """RAG system for querying sales history."""
    
    def __init__(
        self,
        db_path: str,
        embeddings_path: Optional[str] = None,
        llm_client=None,
    ):
        """
        Initialize the RAG system.
        
        Args:
            db_path: Path to SQLite database with sales data
            embeddings_path: Optional path to cache embeddings
            llm_client: Optional LLM client for generation
        """
        self.db_path = db_path
        self.embeddings_path = embeddings_path
        self._client = llm_client
        
        # In-memory stores
        self._sales_days: list[SalesDay] = []
        self._embeddings: Optional[np.ndarray] = None
        self._embedding_model = None
    
    @property
    def client(self):
        if self._client is None:
            from aoc_analytics.llm.client import get_default_client
            self._client = get_default_client()
        return self._client
    
    def load_sales_data(self, store: str, start_date: Optional[date] = None):
        """Load sales data from database."""
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                date,
                SUM(subtotal) as revenue,
                COUNT(DISTINCT invoice_id) as transactions
            FROM sales
            WHERE location = ?
        """
        params = [store]
        
        if start_date:
            query += " AND date >= ?"
            params.append(start_date.isoformat())
        
        query += " GROUP BY date ORDER BY date"
        
        cursor = conn.execute(query, params)
        rows = cursor.fetchall()
        
        # Calculate average for comparison
        revenues = [r[1] for r in rows]
        avg_revenue = np.mean(revenues) if revenues else 0
        
        # Build SalesDay objects
        self._sales_days = []
        for row in rows:
            dt = datetime.strptime(row[0], "%Y-%m-%d").date()
            revenue = row[1]
            transactions = row[2]
            
            # Determine day context
            dow = dt.strftime("%A")
            month = dt.strftime("%B")
            is_weekend = dt.weekday() >= 5
            
            # TODO: Look up actual holiday info
            is_holiday = False
            holiday_name = None
            
            # Revenue vs average
            revenue_vs_avg = ((revenue - avg_revenue) / avg_revenue * 100) if avg_revenue > 0 else 0
            
            sales_day = SalesDay(
                date=dt,
                store=store,
                revenue=revenue,
                transactions=transactions,
                day_of_week=dow,
                month=month,
                is_holiday=is_holiday,
                holiday_name=holiday_name,
                is_weekend=is_weekend,
                weather_summary=None,  # TODO: Add weather lookup
                revenue_vs_avg=revenue_vs_avg,
            )
            self._sales_days.append(sales_day)
        
        conn.close()
        logger.info(f"Loaded {len(self._sales_days)} days of sales data for {store}")
    
    def _get_embedding_model(self):
        """Get or initialize embedding model."""
        if self._embedding_model is None:
            try:
                from sentence_transformers import SentenceTransformer
                self._embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
            except ImportError:
                logger.warning("sentence-transformers not installed, using LLM for similarity")
                return None
        return self._embedding_model
    
    def build_embeddings(self):
        """Build embeddings for all sales days."""
        model = self._get_embedding_model()
        if model is None:
            logger.warning("Cannot build embeddings without sentence-transformers")
            return
        
        texts = [day.to_text() for day in self._sales_days]
        self._embeddings = model.encode(texts, show_progress_bar=True)
        
        # Cache if path provided
        if self.embeddings_path:
            np.save(self.embeddings_path, self._embeddings)
            logger.info(f"Saved embeddings to {self.embeddings_path}")
    
    def load_embeddings(self) -> bool:
        """Load cached embeddings if available."""
        if self.embeddings_path and Path(self.embeddings_path).exists():
            self._embeddings = np.load(self.embeddings_path)
            logger.info(f"Loaded embeddings from {self.embeddings_path}")
            return True
        return False
    
    def search(self, query: str, top_k: int = 10) -> list[SalesDay]:
        """
        Search for days matching a natural language query.
        
        Args:
            query: Natural language query
            top_k: Number of results to return
        
        Returns:
            List of matching SalesDay objects
        """
        if not self._sales_days:
            raise ValueError("No sales data loaded. Call load_sales_data first.")
        
        model = self._get_embedding_model()
        
        if model is not None and self._embeddings is not None:
            # Semantic search with embeddings
            query_embedding = model.encode([query])[0]
            
            # Cosine similarity
            similarities = np.dot(self._embeddings, query_embedding) / (
                np.linalg.norm(self._embeddings, axis=1) * np.linalg.norm(query_embedding)
            )
            
            # Get top-k indices
            top_indices = np.argsort(similarities)[-top_k:][::-1]
            return [self._sales_days[i] for i in top_indices]
        
        else:
            # Fallback: Use LLM to filter
            return self._llm_search(query, top_k)
    
    def _llm_search(self, query: str, top_k: int) -> list[SalesDay]:
        """Use LLM to search when embeddings unavailable."""
        # Sample some days to send to LLM
        sample_size = min(100, len(self._sales_days))
        sample_indices = np.random.choice(len(self._sales_days), sample_size, replace=False)
        sample = [self._sales_days[i] for i in sample_indices]
        
        days_text = "\n".join(f"{i}: {day.to_text()}" for i, day in enumerate(sample))
        
        prompt = f"""Given this query about sales history:
"{query}"

Here are {sample_size} sample days. Return the indices of the {top_k} most relevant days.

{days_text}

Return a JSON object with a "indices" array of the {top_k} most relevant day indices."""

        try:
            result = self.client.complete_json(prompt)
            indices = result.get("indices", [])[:top_k]
            return [sample[i] for i in indices if i < len(sample)]
        except Exception as e:
            logger.error(f"LLM search failed: {e}")
            return sample[:top_k]
    
    def query(self, question: str, top_k: int = 10) -> str:
        """
        Answer a natural language question about sales history.
        
        Args:
            question: Natural language question
            top_k: Number of relevant days to retrieve
        
        Returns:
            Natural language answer
        """
        # Retrieve relevant days
        relevant_days = self.search(question, top_k)
        
        # Build context
        context = "\n\n".join(
            f"**{day.date}** ({day.day_of_week}): ${day.revenue:,.0f} revenue, "
            f"{day.transactions} transactions"
            + (f", {day.holiday_name}" if day.holiday_name else "")
            + (f" ({day.revenue_vs_avg:+.0f}% vs avg)" if abs(day.revenue_vs_avg) > 10 else "")
            for day in relevant_days
        )
        
        prompt = f"""Answer this question about sales history:

Question: {question}

Relevant historical data:
{context}

Provide a helpful, data-driven answer based on the historical records above."""

        return self.client.complete(prompt)
    
    def find_similar_days(
        self,
        target_date: date,
        top_k: int = 5,
    ) -> list[tuple[SalesDay, float]]:
        """
        Find days most similar to a target date.
        
        Args:
            target_date: The date to find similar days for
            top_k: Number of similar days to return
        
        Returns:
            List of (SalesDay, similarity_score) tuples
        """
        # Find the target day
        target_day = None
        for day in self._sales_days:
            if day.date == target_date:
                target_day = day
                break
        
        if target_day is None:
            # Build a synthetic target
            dow = target_date.strftime("%A")
            month = target_date.strftime("%B")
            query = f"A {dow} in {month}"
        else:
            query = target_day.to_text()
        
        # Search for similar days (excluding target)
        results = self.search(query, top_k + 1)
        
        # Remove target date if present
        results = [d for d in results if d.date != target_date][:top_k]
        
        # Add similarity scores (placeholder - would come from embeddings)
        return [(day, 1.0 - i * 0.1) for i, day in enumerate(results)]
    
    def get_summary_stats(self) -> dict:
        """Get summary statistics for loaded data."""
        if not self._sales_days:
            return {}
        
        revenues = [d.revenue for d in self._sales_days]
        return {
            "total_days": len(self._sales_days),
            "date_range": f"{self._sales_days[0].date} to {self._sales_days[-1].date}",
            "avg_revenue": np.mean(revenues),
            "std_revenue": np.std(revenues),
            "min_revenue": np.min(revenues),
            "max_revenue": np.max(revenues),
            "best_day": max(self._sales_days, key=lambda d: d.revenue).to_dict(),
            "worst_day": min(self._sales_days, key=lambda d: d.revenue).to_dict(),
        }
