#!/usr/bin/env python
"""
Demo: LLM-enhanced forecasting.

This script demonstrates the LLM module capabilities:
1. Event extraction from text
2. Anomaly explanation
3. RAG queries over sales history
4. Natural language chat interface

Requirements:
    pip install openai  # or anthropic, or use Ollama locally
    export OPENAI_API_KEY=your-key  # if using OpenAI
"""

import os
import sys
from datetime import date, timedelta

# Add src to path if running directly
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))


def demo_event_extraction():
    """Demo: Extract events from news text."""
    print("\n" + "="*60)
    print("DEMO 1: Event Extraction from News")
    print("="*60)
    
    from aoc_analytics.llm.events import EventExtractor
    
    sample_news = """
    The Parksville Beach Festival returns July 18-20, 2025! 
    Expect over 10,000 visitors for live music, food vendors, and family activities.
    
    Also, the Canucks are playing their season opener on October 9th - 
    BC Place will be packed and downtown Vancouver will be buzzing.
    
    Warning: Highway 19 will be closed for construction from July 25-27, 
    expect significant delays for travel between Nanaimo and Parksville.
    """
    
    print(f"\nSample news text:\n{sample_news}")
    print("\n[Would extract events if LLM API key configured]")
    print("Expected output:")
    print("  - Beach Festival (festival): Jul 18-20, POSITIVE impact +0.4")
    print("  - Canucks Game (sports): Oct 9, NEUTRAL impact")  
    print("  - Highway Closure (construction): Jul 25-27, NEGATIVE impact -0.3")


def demo_anomaly_explanation():
    """Demo: Explain forecast misses."""
    print("\n" + "="*60)
    print("DEMO 2: Anomaly Explanation")
    print("="*60)
    
    from aoc_analytics.llm.explainer import AnomalyContext, AnomalyExplainer
    
    context = AnomalyContext(
        date=date(2024, 7, 20),
        store="Parksville",
        predicted_revenue=4200.0,
        actual_revenue=5800.0,
        error_pct=38.1,  # Way above prediction
        day_of_week="Saturday",
        is_holiday=False,
        is_payday=False,
        weather_temp=28.0,
        weather_precip=0.0,
        similar_days_used=50,
        top_similar_dates=["2023-07-22", "2023-08-05", "2024-06-29"],
        local_events=["Parksville Beach Festival"],
    )
    
    print(f"\nForecast miss context:")
    print(f"  Date: {context.date} ({context.day_of_week})")
    print(f"  Predicted: ${context.predicted_revenue:,.0f}")
    print(f"  Actual: ${context.actual_revenue:,.0f}")
    print(f"  Error: {context.error_pct:+.1f}%")
    print(f"  Events: {context.local_events}")
    
    print("\n[Would generate explanation if LLM API key configured]")
    print("Expected output:")
    print("  Primary cause: Beach Festival bringing increased foot traffic")
    print("  Confidence: 0.85")
    print("  Was predictable: Yes, if event data was integrated")


def demo_rag_query():
    """Demo: RAG queries over sales history."""
    print("\n" + "="*60)
    print("DEMO 3: RAG Query over Sales History")
    print("="*60)
    
    from aoc_analytics.llm.rag import SalesRAG
    
    db_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "aoc_sales.db")
    
    if os.path.exists(db_path):
        rag = SalesRAG(db_path=db_path)
        rag.load_sales_data("Parksville", start_date=date(2024, 1, 1))
        
        stats = rag.get_summary_stats()
        print(f"\nLoaded data: {stats.get('total_days', 0)} days")
        print(f"Date range: {stats.get('date_range', 'N/A')}")
        print(f"Avg revenue: ${stats.get('avg_revenue', 0):,.0f}")
        
        print("\nSample queries you could ask:")
        print('  - "What were the best sales days in July?"')
        print('  - "Find rainy Saturdays similar to tomorrow"')
        print('  - "What happened on the last BC Day long weekend?"')
    else:
        print(f"\nDatabase not found at: {db_path}")
        print("Run with actual database to see RAG in action.")


def demo_chat_interface():
    """Demo: Natural language chat interface."""
    print("\n" + "="*60)
    print("DEMO 4: Natural Language Chat Interface")
    print("="*60)
    
    print("\nThe chat interface allows managers to ask:")
    print('  - "What should I order for the long weekend?"')
    print('  - "Why were sales down last Tuesday?"')
    print('  - "What\'s the forecast for New Year\'s Eve?"')
    print('  - "Show me similar days to this Saturday"')
    
    print("\nTo try it interactively:")
    print("  python -m aoc_analytics.llm.chat --store Parksville --db aoc_sales.db")


def demo_integration():
    """Demo: Integration with forecast engine."""
    print("\n" + "="*60)
    print("DEMO 5: Forecast Engine Integration")
    print("="*60)
    
    from aoc_analytics.llm.integration import LLMForecastEnhancer
    
    enhancer = LLMForecastEnhancer()
    
    # Simulate a forecast enhancement
    target = date(2025, 7, 19)  # Saturday during beach festival
    base_prediction = 4200.0
    
    # Manually add an event to the extractor cache for demo
    from aoc_analytics.llm.events import LocalEvent, EventType, EventImpact
    event = LocalEvent(
        name="Parksville Beach Festival",
        event_type=EventType.FESTIVAL,
        start_date=date(2025, 7, 18),
        end_date=date(2025, 7, 20),
        impact=EventImpact.POSITIVE,
        impact_magnitude=0.4,
        confidence=0.8,
    )
    enhancer.event_extractor.cache_events([event])
    
    adjusted, metadata = enhancer.adjust_forecast(base_prediction, target)
    
    print(f"\nForecast for {target.strftime('%A, %B %d, %Y')}:")
    print(f"  Base prediction: ${base_prediction:,.0f}")
    print(f"  LLM-adjusted:    ${adjusted:,.0f}")
    print(f"  Adjustment:      {metadata['total_adjustment_pct']:+.1f}%")
    print(f"  Events detected: {metadata['features']['event_names']}")
    
    narrative = enhancer.get_forecast_narrative(target, "Parksville", adjusted)
    print(f"\nNarrative:\n{narrative}")


def main():
    """Run all demos."""
    print("🌿 AOC Analytics - LLM Integration Demo")
    print("="*60)
    
    demo_event_extraction()
    demo_anomaly_explanation()
    demo_rag_query()
    demo_chat_interface()
    demo_integration()
    
    print("\n" + "="*60)
    print("SETUP INSTRUCTIONS")
    print("="*60)
    print("""
To use the LLM features:

1. Install dependencies:
   pip install openai anthropic sentence-transformers beautifulsoup4
   
   Or: pip install -e ".[llm]"

2. Set API key (choose one):
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   
   Or use Ollama locally (no API key needed):
   ollama pull llama3.2

3. Run the chat interface:
   python -m aoc_analytics.llm.chat --store Parksville

4. Use in code:
   from aoc_analytics.llm import ForecastChat, LLMForecastEnhancer
   
   chat = ForecastChat(store="Parksville", db_path="aoc_sales.db")
   response = chat.chat("What's the forecast for this Saturday?")
""")


if __name__ == "__main__":
    main()
