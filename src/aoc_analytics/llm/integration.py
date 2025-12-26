"""
Integration layer connecting LLM components to the forecast engine.

This module wires:
- Event extraction → similarity features
- Anomaly explanation → forecast feedback loop
- RAG queries → historical context
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class LLMFeatures:
    """Features extracted by LLM for a specific date."""
    
    date: date
    
    # From event extraction
    has_local_event: bool = False
    event_impact: float = 0.0  # -1 to 1
    event_names: list[str] = None
    
    # From news analysis
    sentiment_score: float = 0.0  # -1 to 1
    news_summary: Optional[str] = None
    
    def __post_init__(self):
        if self.event_names is None:
            self.event_names = []
    
    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "has_local_event": self.has_local_event,
            "event_impact": self.event_impact,
            "event_names": self.event_names,
            "sentiment_score": self.sentiment_score,
            "news_summary": self.news_summary,
        }


class LLMForecastEnhancer:
    """
    Enhances forecast predictions with LLM-derived features.
    
    This class sits between the raw forecast engine output and the final
    prediction, applying adjustments based on LLM analysis.
    """
    
    def __init__(self, llm_client=None, db_path: Optional[str] = None):
        self._client = llm_client
        self.db_path = db_path
        
        # Lazy-loaded components
        self._event_extractor = None
        self._rag = None
        self._explainer = None
        
        # Feature cache
        self._feature_cache: dict[str, LLMFeatures] = {}
    
    @property
    def event_extractor(self):
        if self._event_extractor is None:
            from aoc_analytics.llm.events import EventExtractor
            self._event_extractor = EventExtractor(llm_client=self._client)
        return self._event_extractor
    
    @property
    def rag(self):
        if self._rag is None and self.db_path:
            from aoc_analytics.llm.rag import SalesRAG
            self._rag = SalesRAG(db_path=self.db_path, llm_client=self._client)
        return self._rag
    
    @property
    def explainer(self):
        if self._explainer is None:
            from aoc_analytics.llm.explainer import AnomalyExplainer
            self._explainer = AnomalyExplainer(llm_client=self._client)
        return self._explainer
    
    def get_features_for_date(self, target_date: date) -> LLMFeatures:
        """
        Get LLM-derived features for a specific date.
        
        Uses caching to avoid repeated API calls.
        """
        cache_key = target_date.isoformat()
        
        if cache_key in self._feature_cache:
            return self._feature_cache[cache_key]
        
        # Extract features
        features = LLMFeatures(date=target_date)
        
        # Check for events
        try:
            impact, event_names = self.event_extractor.get_impact_for_date(target_date)
            features.has_local_event = len(event_names) > 0
            features.event_impact = impact
            features.event_names = event_names
        except Exception as e:
            logger.warning(f"Event extraction failed for {target_date}: {e}")
        
        # Cache and return
        self._feature_cache[cache_key] = features
        return features
    
    def adjust_forecast(
        self,
        base_prediction: float,
        target_date: date,
        confidence: float = 1.0,
    ) -> tuple[float, dict]:
        """
        Adjust a base forecast with LLM-derived insights.
        
        Args:
            base_prediction: The similarity-based prediction
            target_date: Date being forecast
            confidence: Base confidence level (0-1)
        
        Returns:
            (adjusted_prediction, metadata_dict)
        """
        features = self.get_features_for_date(target_date)
        
        adjustments = {}
        adjusted = base_prediction
        
        # Apply event impact
        if features.has_local_event and abs(features.event_impact) > 0.1:
            # Scale impact: 0.5 impact = 10% adjustment
            event_adjustment = features.event_impact * 0.2  # Max 20% adjustment
            adjusted *= (1 + event_adjustment)
            adjustments["event_adjustment"] = event_adjustment
            adjustments["event_names"] = features.event_names
        
        # Apply sentiment if significant
        if abs(features.sentiment_score) > 0.3:
            sentiment_adjustment = features.sentiment_score * 0.05  # Max 5% adjustment
            adjusted *= (1 + sentiment_adjustment)
            adjustments["sentiment_adjustment"] = sentiment_adjustment
        
        metadata = {
            "base_prediction": base_prediction,
            "adjusted_prediction": adjusted,
            "total_adjustment_pct": (adjusted - base_prediction) / base_prediction * 100,
            "features": features.to_dict(),
            "adjustments": adjustments,
        }
        
        return adjusted, metadata
    
    def enhance_forecast_results(
        self,
        results: list[dict],
        store: str,
    ) -> list[dict]:
        """
        Enhance a list of forecast results with LLM features.
        
        Args:
            results: List of forecast result dicts with 'date' and 'predicted_revenue'
            store: Store name
        
        Returns:
            Enhanced results with LLM features added
        """
        enhanced = []
        
        for result in results:
            target_date = date.fromisoformat(result["date"]) if isinstance(result["date"], str) else result["date"]
            base_prediction = result.get("predicted_revenue", result.get("revenue", 0))
            
            adjusted, metadata = self.adjust_forecast(base_prediction, target_date)
            
            enhanced_result = {**result}
            enhanced_result["llm_adjusted_revenue"] = adjusted
            enhanced_result["llm_metadata"] = metadata
            enhanced.append(enhanced_result)
        
        return enhanced
    
    def explain_forecast_miss(
        self,
        target_date: date,
        store: str,
        predicted: float,
        actual: float,
        conditions: Optional[dict] = None,
    ) -> dict:
        """
        Generate an explanation for a forecast miss using LLM.
        
        Args:
            target_date: The date of the forecast
            store: Store name
            predicted: Predicted revenue
            actual: Actual revenue
            conditions: Optional dict of conditions used for prediction
        
        Returns:
            Explanation dict
        """
        from aoc_analytics.llm.explainer import AnomalyContext
        
        error_pct = (actual - predicted) / actual * 100 if actual > 0 else 0
        
        # Build context
        context = AnomalyContext(
            date=target_date,
            store=store,
            predicted_revenue=predicted,
            actual_revenue=actual,
            error_pct=error_pct,
            day_of_week=target_date.strftime("%A"),
            is_holiday=conditions.get("is_holiday", False) if conditions else False,
            is_payday=conditions.get("is_payday_window", False) if conditions else False,
            weather_temp=conditions.get("temp_c") if conditions else None,
            weather_precip=conditions.get("precip_mm") if conditions else None,
            local_events=self.get_features_for_date(target_date).event_names,
        )
        
        explanation = self.explainer.explain(context)
        return explanation.to_dict()
    
    def scrape_and_update_events(self, days_ahead: int = 14) -> int:
        """
        Scrape news sources and update event cache.
        
        Args:
            days_ahead: How many days ahead to look for events
        
        Returns:
            Number of events extracted
        """
        from aoc_analytics.llm.events import scrape_local_events
        
        events = scrape_local_events(
            extractor=self.event_extractor,
            days_ahead=days_ahead,
        )
        
        logger.info(f"Extracted {len(events)} events for next {days_ahead} days")
        return len(events)
    
    def get_forecast_narrative(
        self,
        target_date: date,
        store: str,
        predicted_revenue: float,
    ) -> str:
        """
        Generate a natural language narrative for a forecast.
        
        Args:
            target_date: Date being forecast
            store: Store name
            predicted_revenue: Predicted revenue amount
        
        Returns:
            Natural language description of the forecast
        """
        features = self.get_features_for_date(target_date)
        
        parts = [
            f"For {target_date.strftime('%A, %B %d')}, we're forecasting **${predicted_revenue:,.0f}** in revenue for {store}."
        ]
        
        # Add context about why
        dow = target_date.strftime("%A")
        parts.append(f"\nThis is based on historical {dow}s with similar conditions.")
        
        # Add event context
        if features.has_local_event:
            direction = "boost" if features.event_impact > 0 else "reduction"
            parts.append(
                f"\n⚡ **Local events**: {', '.join(features.event_names)} "
                f"may cause a {direction} in sales."
            )
        
        # Add recommendations
        if target_date.weekday() in [4, 5]:  # Fri/Sat
            parts.append("\n💡 **Tip**: Weekend staffing and inventory should be at full capacity.")
        
        return "\n".join(parts)


# Convenience function for quick enhancement
def enhance_prediction(
    base_prediction: float,
    target_date: date,
    store: str = "Parksville",
    db_path: Optional[str] = None,
) -> tuple[float, str]:
    """
    Quick function to enhance a single prediction with LLM insights.
    
    Returns:
        (adjusted_prediction, narrative)
    """
    enhancer = LLMForecastEnhancer(db_path=db_path)
    adjusted, metadata = enhancer.adjust_forecast(base_prediction, target_date)
    narrative = enhancer.get_forecast_narrative(target_date, store, adjusted)
    return adjusted, narrative
