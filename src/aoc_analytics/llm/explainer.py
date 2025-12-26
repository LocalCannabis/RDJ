"""
Anomaly explainer - uses LLM to explain forecast misses.

When actual sales differ significantly from predicted, this module
analyzes available context to explain why.
"""

import logging
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class AnomalyContext:
    """Context for explaining an anomaly."""
    
    date: date
    store: str
    predicted_revenue: float
    actual_revenue: float
    error_pct: float  # (actual - predicted) / actual * 100
    
    # Conditions that were used for prediction
    day_of_week: str
    is_holiday: bool
    is_payday: bool
    weather_temp: Optional[float] = None
    weather_precip: Optional[float] = None
    
    # Additional context
    similar_days_used: int = 0
    top_similar_dates: list[str] = None
    local_events: list[str] = None
    news_context: Optional[str] = None
    
    def __post_init__(self):
        if self.top_similar_dates is None:
            self.top_similar_dates = []
        if self.local_events is None:
            self.local_events = []


@dataclass  
class AnomalyExplanation:
    """LLM-generated explanation for a forecast anomaly."""
    
    date: date
    store: str
    error_pct: float
    
    # LLM outputs
    primary_cause: str
    contributing_factors: list[str]
    confidence: float  # 0-1
    suggested_action: Optional[str] = None
    
    # For learning
    was_predictable: bool = False
    missing_data: list[str] = None
    
    def __post_init__(self):
        if self.missing_data is None:
            self.missing_data = []
    
    def to_dict(self) -> dict:
        return {
            "date": self.date.isoformat(),
            "store": self.store,
            "error_pct": self.error_pct,
            "primary_cause": self.primary_cause,
            "contributing_factors": self.contributing_factors,
            "confidence": self.confidence,
            "suggested_action": self.suggested_action,
            "was_predictable": self.was_predictable,
            "missing_data": self.missing_data,
        }


ANOMALY_SYSTEM_PROMPT = """You are a retail analytics expert for a cannabis dispensary on Vancouver Island, BC.
Your job is to explain why actual sales differed from predicted sales on a given day.

Consider these factors:
1. WEATHER: Rain, extreme heat/cold, storms can significantly impact foot traffic
2. LOCAL EVENTS: Festivals, concerts, sports events bring people to the area
3. HOLIDAYS: Long weekends, stat holidays change shopping patterns
4. PAYDAY: End of month / mid-month payday windows boost sales
5. COMPETITION: New store openings, competitor sales
6. SEASONALITY: Summer tourists, winter locals
7. RANDOM VARIATION: Sometimes sales just vary with no clear cause

Be honest when you don't know. If the variance seems like normal random variation (~15%), say so.
Only claim a specific cause if you have evidence from the context provided."""


ANOMALY_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "primary_cause": {
            "type": "string",
            "description": "The main reason for the forecast miss"
        },
        "contributing_factors": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Other factors that may have contributed"
        },
        "confidence": {
            "type": "number",
            "description": "Confidence in this explanation (0-1)"
        },
        "was_predictable": {
            "type": "boolean",
            "description": "Could this have been predicted with available data?"
        },
        "missing_data": {
            "type": "array",
            "items": {"type": "string"},
            "description": "What additional data would help predict this?"
        },
        "suggested_action": {
            "type": "string",
            "description": "What should the store do differently next time?"
        }
    },
    "required": ["primary_cause", "contributing_factors", "confidence", "was_predictable"]
}


class AnomalyExplainer:
    """Explains forecast anomalies using LLM analysis."""
    
    def __init__(self, llm_client=None):
        self._client = llm_client
    
    @property
    def client(self):
        if self._client is None:
            from aoc_analytics.llm.client import get_default_client
            self._client = get_default_client()
        return self._client
    
    def explain(self, context: AnomalyContext) -> AnomalyExplanation:
        """Generate an explanation for a forecast anomaly."""
        
        # Build the prompt with all available context
        direction = "higher" if context.error_pct > 0 else "lower"
        abs_error = abs(context.error_pct)
        
        prompt = f"""Explain why actual sales were {abs_error:.1f}% {direction} than predicted.

## Forecast Details
- Date: {context.date.strftime('%A, %B %d, %Y')}
- Store: {context.store}
- Predicted Revenue: ${context.predicted_revenue:,.0f}
- Actual Revenue: ${context.actual_revenue:,.0f}
- Error: {context.error_pct:+.1f}%

## Conditions Used for Prediction
- Day of Week: {context.day_of_week}
- Holiday: {'Yes' if context.is_holiday else 'No'}
- Payday Window: {'Yes' if context.is_payday else 'No'}
- Temperature: {f'{context.weather_temp:.0f}°C' if context.weather_temp else 'Unknown'}
- Precipitation: {f'{context.weather_precip:.1f}mm' if context.weather_precip else 'Unknown'}

## Prediction Method
- Used {context.similar_days_used} similar historical days
- Most similar dates: {', '.join(context.top_similar_dates[:5]) if context.top_similar_dates else 'N/A'}

## Additional Context
- Known local events: {', '.join(context.local_events) if context.local_events else 'None identified'}
- News context: {context.news_context or 'None available'}

Note: Normal day-to-day variance for this store is approximately 15-16%. 
If the error is within this range, it may just be random variation.

Analyze the situation and explain the likely cause of the forecast miss."""

        try:
            result = self.client.extract_structured(
                prompt=prompt,
                schema=ANOMALY_EXTRACTION_SCHEMA,
                system=ANOMALY_SYSTEM_PROMPT,
            )
            
            return AnomalyExplanation(
                date=context.date,
                store=context.store,
                error_pct=context.error_pct,
                primary_cause=result.get("primary_cause", "Unknown"),
                contributing_factors=result.get("contributing_factors", []),
                confidence=result.get("confidence", 0.5),
                was_predictable=result.get("was_predictable", False),
                missing_data=result.get("missing_data", []),
                suggested_action=result.get("suggested_action"),
            )
            
        except Exception as e:
            logger.error(f"Anomaly explanation failed: {e}")
            return AnomalyExplanation(
                date=context.date,
                store=context.store,
                error_pct=context.error_pct,
                primary_cause=f"Analysis failed: {e}",
                contributing_factors=[],
                confidence=0.0,
            )
    
    def explain_batch(
        self, 
        contexts: list[AnomalyContext],
        threshold_pct: float = 20.0,
    ) -> list[AnomalyExplanation]:
        """
        Explain multiple anomalies, filtering by significance.
        
        Args:
            contexts: List of anomaly contexts
            threshold_pct: Only explain errors above this threshold
        """
        explanations = []
        
        significant = [c for c in contexts if abs(c.error_pct) >= threshold_pct]
        logger.info(f"Explaining {len(significant)}/{len(contexts)} significant anomalies (>={threshold_pct}%)")
        
        for context in significant:
            explanation = self.explain(context)
            explanations.append(explanation)
        
        return explanations
    
    def summarize_patterns(self, explanations: list[AnomalyExplanation]) -> str:
        """Generate a summary of patterns across multiple anomalies."""
        
        if not explanations:
            return "No anomalies to summarize."
        
        # Count primary causes
        causes = {}
        for exp in explanations:
            cause = exp.primary_cause
            causes[cause] = causes.get(cause, 0) + 1
        
        # Count missing data requests
        missing = {}
        for exp in explanations:
            for item in exp.missing_data:
                missing[item] = missing.get(item, 0) + 1
        
        prompt = f"""Summarize patterns across {len(explanations)} forecast anomalies.

## Primary Causes (frequency)
{chr(10).join(f'- {cause}: {count}' for cause, count in sorted(causes.items(), key=lambda x: -x[1]))}

## Missing Data Requests (frequency)  
{chr(10).join(f'- {item}: {count}' for item, count in sorted(missing.items(), key=lambda x: -x[1])[:10])}

## Sample Anomalies
{chr(10).join(f'- {e.date}: {e.error_pct:+.0f}% - {e.primary_cause}' for e in explanations[:10])}

Provide a brief executive summary of:
1. The most common causes of forecast misses
2. What data sources would most improve accuracy
3. Actionable recommendations for the business"""

        try:
            return self.client.complete(prompt, system=ANOMALY_SYSTEM_PROMPT)
        except Exception as e:
            logger.error(f"Pattern summarization failed: {e}")
            return f"Summarization failed: {e}"
