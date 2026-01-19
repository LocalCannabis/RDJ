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


def get_aoc_db():
    """Get connection to AOC's own database (historical sales + weather)."""
    db_path = os.environ.get(
        "AOC_DB_PATH",
        str(Path(__file__).parent.parent.parent.parent.parent / "aoc_analytics.db")
    )
    return get_connection(db_path)


# =============================================================================
# Endpoints
# =============================================================================


@jfk_router.get("/health")
def jfk_health():
    """Health check for JFK integration."""
    try:
        conn = get_aoc_db()
        result = conn.execute("SELECT COUNT(*) FROM sales").fetchone()
        conn.close()
        return {
            "status": "healthy",
            "database": "aoc_analytics.db",
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
    target_date: Optional[str] = Query(None, description="Date to summarize (YYYY-MM-DD)"),
    store_id: Optional[str] = Query(None, description="Store ID to filter by (e.g., parksville, vancouver, burnaby)")
):
    """Get daily sales summary."""
    conn = get_aoc_db()
    
    # Build store filter
    store_filter = ""
    params = []
    if store_id:
        store_filter = " AND store_id = ?"
        params.append(store_id.lower())
    
    if target_date:
        date_filter = target_date
    else:
        # Get most recent date (for this store if specified)
        if store_id:
            result = conn.execute(f"SELECT MAX(date) FROM sales WHERE store_id = ?", (store_id.lower(),)).fetchone()
        else:
            result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Basic stats
        stats = conn.execute(f"""
            SELECT 
                COUNT(*) as transactions,
                SUM(quantity) as items,
                SUM(subtotal) as revenue,
                AVG(subtotal) as avg_txn
            FROM sales
            WHERE date = ?{store_filter}
        """, (date_filter, *params)).fetchone()
        
        # Top category
        top_cat = conn.execute(f"""
            SELECT category, SUM(subtotal) as rev
            FROM sales
            WHERE date = ? AND category IS NOT NULL{store_filter}
            GROUP BY category
            ORDER BY rev DESC
            LIMIT 1
        """, (date_filter, *params)).fetchone()
        
        # Top product
        top_prod = conn.execute(f"""
            SELECT product_name, SUM(subtotal) as rev
            FROM sales
            WHERE date = ? AND product_name IS NOT NULL{store_filter}
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 1
        """, (date_filter, *params)).fetchone()
        
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
    store_id: Optional[str] = Query(None, description="Store ID to filter by"),
    limit: int = Query(10, ge=1, le=50)
):
    """Get category performance breakdown."""
    conn = get_aoc_db()
    
    # Build store filter
    store_filter = ""
    params = []
    if store_id:
        store_filter = " AND store_id = ?"
        params.append(store_id.lower())
    
    if target_date:
        date_filter = target_date
    else:
        if store_id:
            result = conn.execute(f"SELECT MAX(date) FROM sales WHERE store_id = ?", (store_id.lower(),)).fetchone()
        else:
            result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Get total for percentage calculation
        total = conn.execute(f"""
            SELECT SUM(subtotal) FROM sales WHERE date = ?{store_filter}
        """, (date_filter, *params)).fetchone()[0] or 1
        
        # Get category stats
        rows = conn.execute(f"""
            SELECT 
                category,
                SUM(subtotal) as revenue,
                SUM(quantity) as units,
                COUNT(*) as transactions,
                AVG(unit_price) as avg_price
            FROM sales
            WHERE date = ? AND category IS NOT NULL{store_filter}
            GROUP BY category
            ORDER BY revenue DESC
            LIMIT ?
        """, (date_filter, *params, limit)).fetchall()
        
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
def get_hourly_pattern(
    target_date: Optional[str] = Query(None),
    store_id: Optional[str] = Query(None, description="Store ID to filter by")
):
    """Get hourly sales pattern."""
    conn = get_aoc_db()
    
    # Build store filter
    store_filter = ""
    params = []
    if store_id:
        store_filter = " AND store_id = ?"
        params.append(store_id.lower())
    
    if target_date:
        date_filter = target_date
    else:
        if store_id:
            result = conn.execute(f"SELECT MAX(date) FROM sales WHERE store_id = ?", (store_id.lower(),)).fetchone()
        else:
            result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Get total for percentage
        total = conn.execute(f"""
            SELECT SUM(subtotal) FROM sales WHERE date = ?{store_filter}
        """, (date_filter, *params)).fetchone()[0] or 1
        
        rows = conn.execute(f"""
            SELECT 
                CAST(strftime('%H', time) AS INTEGER) as hour,
                SUM(subtotal) as revenue,
                COUNT(*) as transactions
            FROM sales
            WHERE date = ?{store_filter}
            GROUP BY hour
            ORDER BY hour
        """, (date_filter, *params)).fetchall()
        
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
    store_id: Optional[str] = Query(None, description="Store ID to filter by"),
    insight_type: Optional[str] = Query(None, description="Filter by type")
):
    """Get AI-generated insights for the day."""
    conn = get_aoc_db()
    
    # Build store filter
    store_filter = ""
    params = []
    if store_id:
        store_filter = " AND store_id = ?"
        params.append(store_id.lower())
    
    if target_date:
        date_filter = target_date
    else:
        if store_id:
            result = conn.execute(f"SELECT MAX(date) FROM sales WHERE store_id = ?", (store_id.lower(),)).fetchone()
        else:
            result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Generate insights based on data analysis
        insights = []
        
        # Insight 1: Top performer
        top = conn.execute(f"""
            SELECT product_name, SUM(quantity) as qty, SUM(subtotal) as rev
            FROM sales WHERE date = ?{store_filter}
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 1
        """, (date_filter, *params)).fetchone()
        
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
        categories = conn.execute(f"""
            SELECT category, SUM(subtotal) as rev
            FROM sales WHERE date = ? AND category IS NOT NULL{store_filter}
            GROUP BY category
            ORDER BY rev DESC
            LIMIT 3
        """, (date_filter, *params)).fetchall()
        
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
        avg = conn.execute(f"""
            SELECT AVG(subtotal), COUNT(*) FROM sales WHERE date = ?{store_filter}
        """, (date_filter, *params)).fetchone()
        
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
def get_ai_summary(
    target_date: Optional[str] = Query(None),
    store_id: Optional[str] = Query(None, description="Store ID to filter by")
):
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
    
    conn = get_aoc_db()
    
    # Build store filter
    store_filter = ""
    params = []
    if store_id:
        store_filter = " AND store_id = ?"
        params.append(store_id.lower())
    
    if target_date:
        date_filter = target_date
    else:
        if store_id:
            result = conn.execute(f"SELECT MAX(date) FROM sales WHERE store_id = ?", (store_id.lower(),)).fetchone()
        else:
            result = conn.execute("SELECT MAX(date) FROM sales").fetchone()
        date_filter = result[0] if result[0] else str(date.today())
    
    try:
        # Gather data for summary
        stats = conn.execute(f"""
            SELECT 
                COUNT(*) as txns,
                SUM(quantity) as items,
                SUM(subtotal) as revenue,
                COUNT(DISTINCT category) as cats
            FROM sales WHERE date = ?{store_filter}
        """, (date_filter, *params)).fetchone()
        
        top_products = conn.execute(f"""
            SELECT product_name, SUM(quantity) as qty, SUM(subtotal) as rev
            FROM sales WHERE date = ?{store_filter}
            GROUP BY product_name
            ORDER BY rev DESC
            LIMIT 5
        """, (date_filter, *params)).fetchall()
        
        conn.close()
        
        # Generate AI summary
        llm = get_llm_provider()
        
        store_label = store_id.title() if store_id else "All Stores"
        prompt = f"""You are a cannabis retail analytics AI. Summarize today's performance in 2-3 sentences for a store manager.

Store: {store_label}
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


# =============================================================================
# Signage Recommendations (AOC Decision Router)
# =============================================================================


class SignageRecommendationItem(BaseModel):
    """A single product recommendation for signage."""
    rank: int
    sku: str
    product_name: str
    category: str
    subcategory: Optional[str] = None
    price: float
    score: float
    reasons: list[str]
    lens_scores: dict[str, float]


class SignageDecision(BaseModel):
    """The decision made by AOC for this recommendation set."""
    regime: str
    regime_drivers: list[str]
    selected_lenses: list[str]
    lens_weights: dict[str, float]
    category_boosts: list[str]
    category_demotes: list[str]
    confidence: float
    explanation: str


class SignageRecommendationResponse(BaseModel):
    """Full signage recommendation response."""
    store_id: str
    purpose: str
    generated_at: str
    valid_until: str
    total_items: int
    recommendations: list[SignageRecommendationItem]
    decision: SignageDecision
    categories_represented: list[str]


@jfk_router.get("/signage/recommend", response_model=SignageRecommendationResponse)
def get_signage_recommendations(
    store_id: str = Query(..., description="Store ID (e.g., parksville, kingsway, burnaby)"),
    purpose: str = Query("SIGNAGE", description="Screen purpose: SIGNAGE, ORDERING, PROMO, STAFF_PICKS"),
    max_items: int = Query(8, ge=1, le=20, description="Maximum number of recommendations"),
    categories: Optional[str] = Query(None, description="Comma-separated category filter"),
):
    """
    Get AOC-powered product recommendations for digital signage.
    
    AOC's Decision Router analyzes current conditions (weather, time, events)
    and selects the optimal analysis lenses to score products.
    
    Returns ranked products with explanations of why they were chosen.
    """
    from aoc_analytics.core.recommender import SignageRecommenderV1
    from aoc_analytics.core.decision_router import WeatherContext, TimeContext
    
    logger.info(f"Signage recommendation request: store={store_id}, purpose={purpose}, max={max_items}")
    
    try:
        # Initialize recommender with AOC database
        # Path: /src/aoc_analytics/api/jfk_endpoints.py -> /aoc_analytics.db
        db_path = os.environ.get(
            "AOC_DB_PATH",
            str(Path(__file__).parent.parent.parent.parent / "aoc_analytics.db")
        )
        logger.info(f"Using database: {db_path}")
        recommender = SignageRecommenderV1(db_path=db_path)
        
        # Build constraints
        constraints = {
            "max_items": max_items,
        }
        if categories:
            constraints["categories"] = categories.split(",")
        
        # Get current weather (optional - graceful fallback)
        weather = None
        try:
            from aoc_analytics.core.weather import WeatherClient, STORE_LOCATIONS
            if store_id.lower() in STORE_LOCATIONS:
                client = WeatherClient()
                lat, lon = STORE_LOCATIONS[store_id.lower()]
                weather_data = client.get_current(lat, lon)
                if weather_data:
                    weather = WeatherContext(
                        temp_c=weather_data.get("temp_c", 15),
                        feels_like_c=weather_data.get("feels_like_c", 15),
                        precip_mm=weather_data.get("precip_mm", 0),
                        precip_type=weather_data.get("precip_type", "none"),
                        cloud_cover_pct=weather_data.get("cloud_cover", 50),
                        humidity_pct=weather_data.get("humidity", 50),
                        wind_kph=weather_data.get("wind_kph", 10),
                        condition=weather_data.get("condition", "Clear"),
                    )
        except Exception as e:
            logger.warning(f"Weather fetch failed, using defaults: {e}")
        
        # Generate recommendations
        result = recommender.generate(
            store_id=store_id.lower(),
            purpose=purpose,
            constraints=constraints,
            weather=weather,
        )
        
        # Transform to response format
        recommendations = [
            SignageRecommendationItem(
                rank=item.rank,
                sku=item.product.sku,
                product_name=item.product.product_name,
                category=item.product.category,
                subcategory=item.product.subcategory,
                price=item.product.price,
                score=item.total_score,
                reasons=item.reasons,
                lens_scores=item.lens_scores,
            )
            for item in result.items
        ]
        
        decision = SignageDecision(
            regime=result.decision.regime.name,
            regime_drivers=result.decision.regime.drivers,
            selected_lenses=result.decision.selected_lenses,
            lens_weights=result.decision.lens_weights,
            category_boosts=result.decision.regime.category_boosts,
            category_demotes=result.decision.regime.category_demotes,
            confidence=result.decision.confidence,
            explanation=result.decision.explanation,
        )
        
        return SignageRecommendationResponse(
            store_id=store_id,
            purpose=purpose,
            generated_at=result.generated_at.isoformat(),
            valid_until=result.valid_until.isoformat(),
            total_items=len(recommendations),
            recommendations=recommendations,
            decision=decision,
            categories_represented=result.categories_represented,
        )
        
    except Exception as e:
        logger.error(f"Signage recommendation error: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


# Export the router for inclusion in main app
__all__ = ["jfk_router"]
