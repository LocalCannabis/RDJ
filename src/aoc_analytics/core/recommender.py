"""
Signage Recommender V1

Produces ranked SKU/category lists for digital signage screens.
Uses the Decision Router to determine lens weights, then scores
products accordingly with hard guards and diversity enforcement.
"""

from __future__ import annotations

import logging
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .decision_router import (
    AOCDecisionRouter,
    AnalysisLens,
    DecisionResult,
    RegimeConfig,
    TimeContext,
    WeatherContext,
    SignalContext,
)

logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION
# =============================================================================

@dataclass
class HardGuards:
    """Non-negotiable rules for recommendations."""
    exclude_archived: bool = True
    exclude_zero_stock: bool = True
    min_stock_threshold: int = 3
    max_same_category: int = 3
    max_hero_sku_frequency: float = 0.2  # Don't show same SKU > 20% of time
    require_recent_sale_days: int = 90   # Must have sold in last N days
    min_recommendation_count: int = 6
    max_recommendation_count: int = 12


@dataclass
class ProductData:
    """Product data for scoring."""
    sku: str
    product_name: str
    category: str
    subcategory: Optional[str] = None
    price: float = 0.0
    cost: float = 0.0
    gross_profit: float = 0.0
    stock: int = 0
    
    # Sales metrics
    sales_7d: float = 0.0       # Units sold last 7 days
    sales_30d: float = 0.0      # Units sold last 30 days
    revenue_7d: float = 0.0     # Revenue last 7 days
    revenue_30d: float = 0.0    # Revenue last 30 days
    
    # Derived metrics
    velocity_daily: float = 0.0      # Average units/day
    margin_pct: float = 0.0          # Gross margin percentage
    basket_affinity: float = 0.0     # How often in multi-item carts
    days_since_last_sale: int = 0
    
    # Volatility (for ABCXYZ)
    cv: float = 0.0  # Coefficient of variation
    
    # Flags
    is_archived: bool = False
    is_hero_sku: bool = False
    
    @property
    def abc_class(self) -> str:
        """ABC classification based on revenue contribution."""
        # This would be calculated across all products
        # A = top 80%, B = next 15%, C = bottom 5%
        return "A"  # Placeholder
    
    @property
    def xyz_class(self) -> str:
        """XYZ classification based on volatility."""
        if self.cv < 0.3:
            return "X"  # Stable
        elif self.cv < 0.7:
            return "Y"  # Variable
        return "Z"  # Unpredictable


@dataclass
class ScoredProduct:
    """Product with calculated score and reasons."""
    product: ProductData
    total_score: float
    lens_scores: Dict[str, float]
    reasons: List[str]
    rank: int = 0
    
    # Override tracking
    is_pinned: bool = False
    pinned_position: Optional[int] = None
    is_removed: bool = False
    override_reason: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "rank": self.rank,
            "sku": self.product.sku,
            "product_name": self.product.product_name,
            "category": self.product.category,
            "subcategory": self.product.subcategory,
            "price": self.product.price,
            "score": round(self.total_score, 3),
            "stock": self.product.stock,
            "reasons": self.reasons,
            "lens_scores": {k: round(v, 3) for k, v in self.lens_scores.items()},
            "is_pinned": self.is_pinned,
            "pinned_position": self.pinned_position,
        }


@dataclass
class RecommendationResult:
    """Full recommendation output."""
    items: List[ScoredProduct]
    decision: DecisionResult
    constraints_applied: Dict[str, Any]
    excluded_reasons: List[Dict[str, str]]
    categories_represented: List[str]
    generated_at: datetime
    valid_until: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "items": [item.to_dict() for item in self.items],
            "decision": self.decision.to_dict(),
            "constraints_applied": self.constraints_applied,
            "excluded_reasons": self.excluded_reasons,
            "categories_represented": self.categories_represented,
            "total_items": len(self.items),
            "generated_at": self.generated_at.isoformat(),
            "valid_until": self.valid_until.isoformat(),
        }


# =============================================================================
# RECOMMENDER ENGINE
# =============================================================================

class SignageRecommenderV1:
    """
    V1 Signage Recommender
    
    Produces ranked product recommendations for digital displays.
    
    Usage:
        recommender = SignageRecommenderV1(db_path="aoc_sales.db")
        
        result = recommender.generate(
            store_id="Parksville",
            purpose="SIGNAGE",
            constraints={"max_items": 12, "min_stock": 5}
        )
    """
    
    def __init__(
        self,
        db_path: Optional[str] = None,
        guards: Optional[HardGuards] = None,
    ):
        self.db_path = db_path or str(Path(__file__).parent.parent.parent.parent / "aoc_sales.db")
        self.guards = guards or HardGuards()
        self.router = AOCDecisionRouter()
    
    def generate(
        self,
        store_id: str,
        purpose: str = "SIGNAGE",
        time_horizon: str = "NOW",
        constraints: Optional[Dict[str, Any]] = None,
        overrides: Optional[List[Dict[str, Any]]] = None,
        weather: Optional[WeatherContext] = None,
        signals: Optional[SignalContext] = None,
    ) -> RecommendationResult:
        """
        Generate recommendations for a screen.
        
        Args:
            store_id: Store identifier
            purpose: Screen purpose (SIGNAGE, ORDERING, etc.)
            time_horizon: NOW, TODAY, NEXT_7_DAYS
            constraints: Screen constraints (max_items, min_stock, etc.)
            overrides: List of manual overrides (pins, removes)
            weather: Current weather context
            signals: Behavioral signals
        
        Returns:
            RecommendationResult with ranked products
        """
        constraints = constraints or {}
        overrides = overrides or []
        
        logger.info(f"Generating recommendations for {store_id} ({purpose})")
        
        # Step 1: Get decision (lens selection)
        decision = self.router.decide(
            purpose=purpose,
            time_horizon=time_horizon,
            weather=weather,
            time=TimeContext.now(),
            signals=signals,
        )
        
        logger.info(f"Decision: {decision.selected_lenses}, regime={decision.regime.name}")
        
        # Step 2: Load products with metrics
        products = self._load_products(store_id)
        logger.info(f"Loaded {len(products)} products")
        
        # Step 3: Apply hard guards (filter out ineligible)
        eligible, excluded = self._apply_hard_guards(products, constraints)
        logger.info(f"After guards: {len(eligible)} eligible, {len(excluded)} excluded")
        
        # Step 4: Score products
        scored = self._score_products(eligible, decision, constraints)
        
        # Step 5: Apply overrides
        scored = self._apply_overrides(scored, overrides)
        
        # Step 6: Rank and enforce diversity
        ranked = self._rank_with_diversity(scored, decision.regime, constraints)
        
        # Step 7: Build result
        max_items = constraints.get("max_items", self.guards.max_recommendation_count)
        final_items = ranked[:max_items]
        
        # Assign final ranks
        for i, item in enumerate(final_items, 1):
            item.rank = i
        
        categories = list(set(item.product.category for item in final_items))
        
        # Calculate validity window
        now = datetime.utcnow()
        refresh_hours = constraints.get("refresh_hours", 4)
        valid_until = now + timedelta(hours=refresh_hours)
        
        return RecommendationResult(
            items=final_items,
            decision=decision,
            constraints_applied=constraints,
            excluded_reasons=excluded,
            categories_represented=categories,
            generated_at=now,
            valid_until=valid_until,
        )
    
    def _load_products(self, store_id: str) -> List[ProductData]:
        """Load products with sales metrics from database."""
        products = []
        
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            
            # Get unique products with aggregated metrics
            query = """
                WITH product_metrics AS (
                    SELECT 
                        sku,
                        product_name,
                        category,
                        MAX(subtotal / NULLIF(quantity, 0)) as price,
                        
                        -- 7-day metrics
                        SUM(CASE WHEN date >= date('now', '-7 days') THEN quantity ELSE 0 END) as sales_7d,
                        SUM(CASE WHEN date >= date('now', '-7 days') THEN subtotal ELSE 0 END) as revenue_7d,
                        SUM(CASE WHEN date >= date('now', '-7 days') THEN gross_profit ELSE 0 END) as profit_7d,
                        
                        -- 30-day metrics
                        SUM(CASE WHEN date >= date('now', '-30 days') THEN quantity ELSE 0 END) as sales_30d,
                        SUM(CASE WHEN date >= date('now', '-30 days') THEN subtotal ELSE 0 END) as revenue_30d,
                        SUM(CASE WHEN date >= date('now', '-30 days') THEN gross_profit ELSE 0 END) as profit_30d,
                        
                        -- Recency
                        CAST(julianday('now') - julianday(MAX(date)) AS INTEGER) as days_since_sale,
                        
                        -- Transaction count (for basket affinity proxy)
                        COUNT(DISTINCT date || time) as transaction_count
                        
                    FROM sales
                    WHERE store_id = ?
                    GROUP BY sku, product_name, category
                )
                SELECT * FROM product_metrics
                WHERE sales_30d > 0
                ORDER BY revenue_30d DESC
            """
            
            cursor.execute(query, (store_id,))
            rows = cursor.fetchall()
            
            for row in rows:
                # Calculate derived metrics
                revenue_7d = row["revenue_7d"] or 0
                profit_7d = row["profit_7d"] or 0
                price = row["price"] or 0
                
                margin_pct = (profit_7d / revenue_7d) if revenue_7d > 0 else 0
                velocity = (row["sales_7d"] or 0) / 7
                
                # Basket affinity proxy: higher transaction count = more basket appearances
                basket_affinity = min((row["transaction_count"] or 0) / 100, 1.0)
                
                products.append(ProductData(
                    sku=row["sku"],
                    product_name=row["product_name"],
                    category=row["category"] or "uncategorized",
                    price=price,
                    gross_profit=profit_7d,
                    sales_7d=row["sales_7d"] or 0,
                    sales_30d=row["sales_30d"] or 0,
                    revenue_7d=revenue_7d,
                    revenue_30d=row["revenue_30d"] or 0,
                    velocity_daily=velocity,
                    margin_pct=margin_pct,
                    basket_affinity=basket_affinity,
                    days_since_last_sale=row["days_since_sale"] or 999,
                    stock=100,  # Placeholder - would come from inventory system
                ))
            
            conn.close()
            
        except Exception as e:
            logger.error(f"Error loading products: {e}")
        
        return products
    
    def _apply_hard_guards(
        self,
        products: List[ProductData],
        constraints: Dict[str, Any],
    ) -> Tuple[List[ProductData], List[Dict[str, str]]]:
        """Apply hard guards to filter out ineligible products."""
        eligible = []
        excluded = []
        
        min_stock = constraints.get("min_stock", self.guards.min_stock_threshold)
        max_days_since_sale = self.guards.require_recent_sale_days
        avoid_categories = constraints.get("avoid_categories", [])
        
        for product in products:
            # Check archived
            if self.guards.exclude_archived and product.is_archived:
                excluded.append({"sku": product.sku, "reason": "Archived product"})
                continue
            
            # Check zero stock
            if self.guards.exclude_zero_stock and product.stock == 0:
                excluded.append({"sku": product.sku, "reason": "Out of stock"})
                continue
            
            # Check minimum stock
            if product.stock < min_stock:
                excluded.append({"sku": product.sku, "reason": f"Low stock ({product.stock} < {min_stock})"})
                continue
            
            # Check recent sales
            if product.days_since_last_sale > max_days_since_sale:
                excluded.append({"sku": product.sku, "reason": f"No recent sales ({product.days_since_last_sale} days)"})
                continue
            
            # Check avoid categories
            if product.category.lower() in [c.lower() for c in avoid_categories]:
                excluded.append({"sku": product.sku, "reason": f"Category excluded: {product.category}"})
                continue
            
            # Check price band
            price_band = constraints.get("price_band", {})
            if price_band:
                min_price = price_band.get("min", 0)
                max_price = price_band.get("max", float("inf"))
                if not (min_price <= product.price <= max_price):
                    excluded.append({"sku": product.sku, "reason": f"Outside price band (${product.price})"})
                    continue
            
            eligible.append(product)
        
        return eligible, excluded
    
    def _score_products(
        self,
        products: List[ProductData],
        decision: DecisionResult,
        constraints: Dict[str, Any],
    ) -> List[ScoredProduct]:
        """Score products based on lens weights."""
        scored = []
        
        for product in products:
            lens_scores = {}
            reasons = []
            
            # FUNNEL lens: velocity-based
            if AnalysisLens.FUNNEL.value in decision.lens_weights:
                weight = decision.lens_weights[AnalysisLens.FUNNEL.value]
                # Normalize velocity (assume 10/day is exceptional)
                velocity_score = min(product.velocity_daily / 10, 1.0)
                lens_scores[AnalysisLens.FUNNEL.value] = velocity_score * weight
                if velocity_score > 0.5:
                    reasons.append(f"High velocity: {product.velocity_daily:.1f}/day")
            
            # MARGIN_MIX lens: profit-weighted
            if AnalysisLens.MARGIN_MIX.value in decision.lens_weights:
                weight = decision.lens_weights[AnalysisLens.MARGIN_MIX.value]
                # Normalize margin (assume 50% is exceptional)
                margin_score = min(product.margin_pct / 0.5, 1.0) if product.margin_pct > 0 else 0
                lens_scores[AnalysisLens.MARGIN_MIX.value] = margin_score * weight
                if margin_score > 0.5:
                    reasons.append(f"Strong margin: {product.margin_pct:.0%}")
            
            # BASKET lens: basket affinity
            if AnalysisLens.BASKET.value in decision.lens_weights:
                weight = decision.lens_weights[AnalysisLens.BASKET.value]
                basket_score = product.basket_affinity
                lens_scores[AnalysisLens.BASKET.value] = basket_score * weight
                if basket_score > 0.5:
                    reasons.append("Strong basket performer")
            
            # ABCXYZ lens: stability
            if AnalysisLens.ABCXYZ.value in decision.lens_weights:
                weight = decision.lens_weights[AnalysisLens.ABCXYZ.value]
                # Favor stable performers (low CV)
                stability_score = 1.0 - min(product.cv, 1.0)
                lens_scores[AnalysisLens.ABCXYZ.value] = stability_score * weight
            
            # Regime category boosts
            regime = decision.regime
            if product.category.lower() in [c.lower() for c in regime.category_boosts]:
                boost = 0.15
                lens_scores["REGIME_BOOST"] = boost
                reasons.append(f"Weather boost: {regime.name}")
            elif product.category.lower() in [c.lower() for c in regime.category_demotes]:
                lens_scores["REGIME_DEMOTE"] = -0.1
            
            # Must-have category boost
            must_have = constraints.get("must_have_categories", [])
            if product.category.lower() in [c.lower() for c in must_have]:
                lens_scores["MUST_HAVE"] = 0.1
                reasons.append("Required category")
            
            # Calculate total score
            total_score = sum(lens_scores.values())
            
            scored.append(ScoredProduct(
                product=product,
                total_score=total_score,
                lens_scores=lens_scores,
                reasons=reasons,
            ))
        
        return scored
    
    def _apply_overrides(
        self,
        scored: List[ScoredProduct],
        overrides: List[Dict[str, Any]],
    ) -> List[ScoredProduct]:
        """Apply manual overrides (pins, removes)."""
        override_map = {o.get("sku"): o for o in overrides}
        
        result = []
        for item in scored:
            override = override_map.get(item.product.sku)
            if override:
                if override.get("type") == "REMOVE":
                    item.is_removed = True
                    item.override_reason = override.get("reason", "Manually removed")
                    continue  # Skip this item
                elif override.get("type") == "PIN":
                    item.is_pinned = True
                    item.pinned_position = override.get("position", 1)
                    item.override_reason = override.get("reason", "Manually pinned")
                    item.total_score = 999  # Ensure it ranks high
                    item.reasons.append(f"📌 Pinned by {override.get('created_by', 'staff')}")
            
            result.append(item)
        
        return result
    
    def _rank_with_diversity(
        self,
        scored: List[ScoredProduct],
        regime: RegimeConfig,
        constraints: Dict[str, Any],
    ) -> List[ScoredProduct]:
        """Rank products while enforcing category diversity."""
        # Sort by score (descending)
        scored.sort(key=lambda x: (x.is_pinned, x.total_score), reverse=True)
        
        # Extract pinned items first
        pinned = [item for item in scored if item.is_pinned]
        unpinned = [item for item in scored if not item.is_pinned]
        
        # Sort pinned by their pinned position
        pinned.sort(key=lambda x: x.pinned_position or 999)
        
        # Apply diversity to unpinned
        category_caps = constraints.get("category_caps", {})
        max_per_category = constraints.get("max_per_category", self.guards.max_same_category)
        
        category_counts: Dict[str, int] = {}
        diverse_unpinned = []
        overflow = []
        
        for item in unpinned:
            cat = item.product.category.lower()
            cap = category_caps.get(cat, max_per_category)
            
            if category_counts.get(cat, 0) < cap:
                diverse_unpinned.append(item)
                category_counts[cat] = category_counts.get(cat, 0) + 1
            else:
                overflow.append(item)
        
        # Combine: pinned first (in position order), then diverse unpinned
        result = []
        
        # Insert pinned at their positions
        pinned_positions = {(item.pinned_position or 1): item for item in pinned}
        
        unpinned_idx = 0
        for position in range(1, len(scored) + 1):
            if position in pinned_positions:
                result.append(pinned_positions[position])
            elif unpinned_idx < len(diverse_unpinned):
                result.append(diverse_unpinned[unpinned_idx])
                unpinned_idx += 1
        
        # Add remaining if needed
        while unpinned_idx < len(diverse_unpinned):
            result.append(diverse_unpinned[unpinned_idx])
            unpinned_idx += 1
        
        return result


# =============================================================================
# CONVENIENCE FUNCTIONS
# =============================================================================

def generate_recommendations(
    store_id: str,
    purpose: str = "SIGNAGE",
    constraints: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> RecommendationResult:
    """
    Convenience function to generate recommendations.
    
    Usage:
        from aoc_analytics.core.recommender import generate_recommendations
        
        result = generate_recommendations(
            store_id="Parksville",
            purpose="SIGNAGE",
            constraints={"max_items": 12, "min_stock": 5}
        )
        
        for item in result.items:
            print(f"{item.rank}. {item.product.product_name} - {item.reasons}")
    """
    recommender = SignageRecommenderV1(db_path=db_path)
    return recommender.generate(
        store_id=store_id,
        purpose=purpose,
        constraints=constraints,
    )


def generate_with_weather(
    store_id: str,
    purpose: str = "SIGNAGE",
    constraints: Optional[Dict[str, Any]] = None,
    db_path: Optional[str] = None,
) -> RecommendationResult:
    """
    Generate recommendations with current weather from Open-Meteo.
    """
    from .weather import WeatherClient
    
    recommender = SignageRecommenderV1(db_path=db_path)
    
    # Get current weather
    try:
        client = WeatherClient(location=store_id)
        current = client.get_current()
        
        weather = WeatherContext(
            temp_c=current.temp_c,
            feels_like_c=current.feels_like_c,
            precip_mm=current.precip_mm,
            precip_type=current.precip_type,
            cloud_cover_pct=current.cloud_cover_pct,
            humidity_pct=current.humidity_pct,
            wind_kph=current.wind_kph,
            condition=current.condition,
        )
    except Exception as e:
        logger.warning(f"Could not fetch weather: {e}")
        weather = None
    
    return recommender.generate(
        store_id=store_id,
        purpose=purpose,
        constraints=constraints,
        weather=weather,
    )
