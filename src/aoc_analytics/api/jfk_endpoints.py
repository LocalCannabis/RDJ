"""
JFK Integration API Endpoints

These endpoints expose AOC Brain insights to the JFK frontend.
They provide:
- Daily insights and recommendations
- Predictive analytics
- Category performance analysis
- Trend detection
- AI-generated summaries

These endpoints are designed to power the "AOC Insights" tab in JFK's UI.
"""

from __future__ import annotations

import os
import json
import logging
from datetime import datetime, date, timedelta
from typing import Any, Optional
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from aoc_analytics.core.db_adapter import get_connection

logger = logging.getLogger(__name__)

# Router for JFK integration
jfk_router = APIRouter(prefix="/api/v1/jfk", tags=["jfk-insights"])


# =============================================================================
# Models
# =============================================================================


class InsightItem(BaseModel):
    """A single insight from the brain."""
    id: str
    type: str = Field(..., description="recommendation, warning, observation, prediction")
    title: str
    content: str
    confidence: float = Field(..., ge=0.0, le=1.0)
    category: Optional[str] = None
    actionable: bool = True
    created_at: str


class InsightsResponse(BaseModel):
    """Response containing multiple insights."""
    date: str
    insights: list[InsightItem]
    total_count: int


class DailySummary(BaseModel):
    """Daily sales summary."""
    date: str
    total_revenue: float
    total_transactions: int
    total_items: int
    avg_transaction: float
    top_category: str
    top_product: str


class CategoryPerformance(BaseModel):
    """Category performance metrics."""
    category: str
    revenue: float
    units_sold: int
    transactions: int
    avg_price: float
    pct_of_total: float


class HourlyPattern(BaseModel):
    """Hourly sales pattern."""
    hour: int
    revenue: float
    transactions: int
    pct_of_daily: float


class PredictionItem(BaseModel):
    """A sales prediction."""
    period: str
    predicted_revenue: float
    confidence: float
    factors: list[str]


class TrendItem(BaseModel):
    """A detected trend."""
    category: str
    direction: str  # up, down, stable
    change_pct: float
    period_days: int


class AnalyticsOverview(BaseModel):
    """Complete analytics overview for dashboard."""
    summary: DailySummary
    categories: list[CategoryPerformance]
    hourly_pattern: list[HourlyPattern]
    insights: list[InsightItem]
    trends: list[TrendItem]
    ai_summary: str


# =============================================================================
# Database helpers
# =============================================================================


def get_jfk_db():
    """Get connection to JFK database."""
    db_path = os.environ.get(
        "JFK_DB_PATH",
        str(Path.home() / "Projects" / "JFK" / "backend" / "instance" / "cannabis_retail.db")
    )
    return get_connection(db_path)


# =============================================================================
# Endpoints
# =============================================================================


@jfk_router.get("/health")
def jfk_health():
    """Health check for JFK integration."""
    try:
        conn = get_jfk_db()
        result = conn.execute("SELECT COUNT(*) FROM sales").fetchone()
        conn.close()
        return {
            "status": "healthy",
            "sales_records": result[0],
            "integration": "active"
        }
    except Exception as e:
        return {
            "status": "unhealthy",
            "error": str(e),
            "integration": "inactive"
        }


@jfk_router.get("/summary", response_model=DailySummary)
def get_daily_summary(
    target_date: Optional[str] = Query(None, description="Date to summarize (YYYY-MM-DD)")
):
    """Get daily sales summary."""
    conn = get_jfk_db()
    
    if target_date:
        date_filter = target_date
    else:
        # Get most recent date
        result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Basic stats
        stats = conn.execute("""
            SELECT 
                COUNT(*) as transactions,
                SUM(quantity) as items,
                SUM(subtotal) as revenue,
                AVG(subtotal) as avg_txn
            FROM sales
            WHERE date = ?
        """, (date_filter,)).fetchone()
        
        # Top category
        top_cat = conn.execute("""
            SELECT category, SUM(subtotal) as rev
            FROM sales
            WHERE date = ? AND category IS NOT NULL
            GROUP BY category
            ORDER BY rev DESC
            LIMIT 1
        """, (date_filter,)).fetchone()
        
        # Top product
        top_prod = conn.execute("""
            SELECT product_name, SUM(subtotal) as rev
            FROM sales
            WHERE date = ? AND product_name IS NOT NULL
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 1
        """, (date_filter,)).fetchone()
        
        conn.close()
        
        return DailySummary(
            date=date_filter,
            total_revenue=stats[2] or 0,
            total_transactions=stats[0] or 0,
            total_items=int(stats[1] or 0),
            avg_transaction=stats[3] or 0,
            top_category=top_cat[0] if top_cat else "Unknown",
            top_product=top_prod[0] if top_prod else "Unknown"
        )
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@jfk_router.get("/categories", response_model=list[CategoryPerformance])
def get_category_performance(
    target_date: Optional[str] = Query(None),
    limit: int = Query(10, ge=1, le=50)
):
    """Get category performance breakdown."""
    conn = get_jfk_db()
    
    if target_date:
        date_filter = target_date
    else:
        result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Get total for percentage calculation
        total = conn.execute("""
            SELECT SUM(subtotal) FROM sales WHERE date = ?
        """, (date_filter,)).fetchone()[0] or 1
        
        # Get category stats
        rows = conn.execute("""
            SELECT 
                category,
                SUM(subtotal) as revenue,
                SUM(quantity) as units,
                COUNT(*) as transactions,
                AVG(unit_price) as avg_price
            FROM sales
            WHERE date = ? AND category IS NOT NULL
            GROUP BY category
            ORDER BY revenue DESC
            LIMIT ?
        """, (date_filter, limit)).fetchall()
        
        conn.close()
        
        return [
            CategoryPerformance(
                category=row[0],
                revenue=row[1],
                units_sold=int(row[2]),
                transactions=row[3],
                avg_price=row[4] or 0,
                pct_of_total=round(row[1] / total * 100, 1)
            )
            for row in rows
        ]
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@jfk_router.get("/hourly", response_model=list[HourlyPattern])
def get_hourly_pattern(target_date: Optional[str] = Query(None)):
    """Get hourly sales pattern."""
    conn = get_jfk_db()
    
    if target_date:
        date_filter = target_date
    else:
        result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Get total for percentage
        total = conn.execute("""
            SELECT SUM(subtotal) FROM sales WHERE date = ?
        """, (date_filter,)).fetchone()[0] or 1
        
        rows = conn.execute("""
            SELECT 
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(subtotal) as revenue,
                COUNT(*) as transactions
            FROM sales
            WHERE date = ?
            GROUP BY hour
            ORDER BY hour
        """, (date_filter,)).fetchall()
        
        conn.close()
        
        return [
            HourlyPattern(
                hour=row[0],
                revenue=row[1],
                transactions=row[2],
                pct_of_daily=round(row[1] / total * 100, 1)
            )
            for row in rows
        ]
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@jfk_router.get("/insights", response_model=InsightsResponse)
def get_insights(
    target_date: Optional[str] = Query(None),
    insight_type: Optional[str] = Query(None, description="Filter by type")
):
    """Get AI-generated insights for the day."""
    conn = get_jfk_db()
    
    if target_date:
        date_filter = target_date
    else:
        result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Generate insights based on data analysis
        insights = []
        
        # Insight 1: Top performer
        top = conn.execute("""
            SELECT product_name, SUM(quantity) as qty, SUM(subtotal) as rev
            FROM sales WHERE date = ?
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 1
        """, (date_filter,)).fetchone()
        
        if top:
            insights.append(InsightItem(
                id=f"top-{date_filter}",
                type="observation",
                title="Top Performer",
                content=f"{top[0]} led sales with {int(top[1])} units sold for ${top[2]:.2f}",
                confidence=1.0,
                category="performance",
                created_at=datetime.now().isoformat()
            ))
        
        # Insight 2: Category trend
        categories = conn.execute("""
            SELECT category, SUM(subtotal) as rev
            FROM sales WHERE date = ? AND category IS NOT NULL
            GROUP BY category
            ORDER BY rev DESC
            LIMIT 3
        """, (date_filter,)).fetchall()
        
        if categories:
            cat_names = [c[0].split(' > ')[-1] for c in categories]
            insights.append(InsightItem(
                id=f"cat-{date_filter}",
                type="observation",
                title="Category Leaders",
                content=f"Top categories today: {', '.join(cat_names)}",
                confidence=0.9,
                category="categories",
                created_at=datetime.now().isoformat()
            ))
        
        # Insight 3: Average transaction value
        avg = conn.execute("""
            SELECT AVG(subtotal), COUNT(*) FROM sales WHERE date = ?
        """, (date_filter,)).fetchone()
        
        if avg and avg[0]:
            insights.append(InsightItem(
                id=f"avg-{date_filter}",
                type="recommendation",
                title="Transaction Value",
                content=f"Average transaction is ${avg[0]:.2f}. Consider upselling to increase basket size.",
                confidence=0.85,
                category="revenue",
                actionable=True,
                created_at=datetime.now().isoformat()
            ))
        
        conn.close()
        
        # Filter by type if requested
        if insight_type:
            insights = [i for i in insights if i.type == insight_type]
        
        return InsightsResponse(
            date=date_filter,
            insights=insights,
            total_count=len(insights)
        )
        
    except Exception as e:
        conn.close()
        raise HTTPException(status_code=500, detail=str(e))


@jfk_router.get("/ai-summary")
def get_ai_summary(target_date: Optional[str] = Query(None)):
    """Get AI-generated executive summary."""
    # Load OpenAI key from JFK .env if not already set
    if not os.environ.get("OPENAI_API_KEY"):
        jfk_env = Path.home() / "Projects" / "JFK" / "backend" / ".env"
        if jfk_env.exists():
            for line in jfk_env.read_text().splitlines():
                if line.startswith("OPENAI_API_KEY="):
                    os.environ["OPENAI_API_KEY"] = line.split("=", 1)[1].strip().strip('"')
                    break
    
    from aoc_analytics.brain.llm_provider import get_llm_provider
    
    conn = get_jfk_db()
    
    if target_date:
        date_filter = target_date
    else:
        result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Gather data for summary
        stats = conn.execute("""
            SELECT 
                COUNT(*) as txns,
                SUM(quantity) as items,
                SUM(subtotal) as revenue,
                COUNT(DISTINCT category) as cats
            FROM sales WHERE date = ?
        """, (date_filter,)).fetchone()
        
        top_products = conn.execute("""
            SELECT product_name, SUM(quantity) as qty, SUM(subtotal) as rev
            FROM sales WHERE date = ?
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 5
        """, (date_filter,)).fetchall()
        
        conn.close()
        
        # Generate AI summary
        llm = get_llm_provider()
        
        prompt = f"""You are a cannabis retail analytics AI. Summarize today's performance in 2-3 sentences for a store manager.

Date: {date_filter}
Transactions: {stats[0]}
Items Sold: {stats[1]}
Revenue: ${stats[2]:,.2f}
Categories: {stats[3]}

Top Products:
{chr(10).join([f'- {p[0]}: {p[1]} units, ${p[2]:.2f}' for p in top_products])}

Be concise, data-driven, and include one actionable suggestion."""

        response = llm.generate(prompt, max_tokens=200, temperature=0.7)
        
        return {
            "date": date_filter,
            "summary": response.text if hasattr(response, 'text') else str(response),
            "generated_at": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"AI summary error: {e}")
        return {
            "date": date_filter,
            "summary": "Unable to generate AI summary at this time.",
            "error": str(e)
        }


@jfk_router.get("/overview", response_model=AnalyticsOverview)
def get_analytics_overview(target_date: Optional[str] = Query(None)):
    """Get complete analytics overview for dashboard."""
    # Combine all endpoints into one response
    summary = get_daily_summary(target_date)
    categories = get_category_performance(target_date, limit=5)
    hourly = get_hourly_pattern(target_date)
    insights_resp = get_insights(target_date)
    ai = get_ai_summary(target_date)
    
    return AnalyticsOverview(
        summary=summary,
        categories=categories,
        hourly_pattern=hourly,
        insights=insights_resp.insights,
        trends=[],  # TODO: Add trend detection
        ai_summary=ai.get("summary", "")
    )


# =============================================================================
# Brain-powered endpoints (more advanced analytics)
# =============================================================================


@jfk_router.get("/brain/status")
def get_brain_status():
    """Get the status of the brain analysis system."""
    return {
        "brain_active": True,
        "last_analysis": datetime.now().isoformat(),
        "modules": {
            "signal_detection": "active",
            "hypothesis_engine": "active",
            "llm_synthesis": "active",
            "memory": "active"
        },
        "insights_generated_today": 0,  # TODO: Track actual count
        "version": "2.0.0-brain"
    }


@jfk_router.post("/brain/analyze")
def trigger_brain_analysis():
    """Trigger a brain analysis cycle (for on-demand insights)."""
    # This would trigger the brain to run a think cycle
    # For now, return a placeholder
    return {
        "status": "analysis_queued",
        "message": "Brain analysis has been triggered",
        "estimated_completion": "30 seconds"
    }


# Export the router for inclusion in main app
__all__ = ["jfk_router"]
