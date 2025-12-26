"""
Natural language chat interface for forecasting.

Allows managers to ask questions like:
- "What should I order for the long weekend?"
- "Why were sales down last Tuesday?"
- "What's the forecast for New Year's Eve?"
- "Show me similar days to this Saturday"
"""

import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ChatMessage:
    """A message in the chat history."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class ChatContext:
    """Context for the chat session."""
    store: str
    current_date: date = field(default_factory=date.today)
    history: list[ChatMessage] = field(default_factory=list)
    
    # Loaded tools/data
    has_forecast_engine: bool = False
    has_rag: bool = False
    has_events: bool = False


CHAT_SYSTEM_PROMPT = """You are an AI assistant for a cannabis retail store on Vancouver Island, BC.
You help store managers with:
- Sales forecasting and planning
- Understanding historical sales patterns  
- Inventory ordering recommendations
- Explaining why sales varied from expectations

You have access to:
1. **Forecast Engine**: Predicts daily revenue based on similar historical days
2. **Sales History**: Years of historical sales data you can query
3. **Event Detection**: Local events that may impact sales

Current store: {store}
Today's date: {current_date}

Be helpful, concise, and data-driven. When you're uncertain, say so.
Format numbers nicely (e.g., $4,200 not 4200.0).
If asked about specific dates, check if they're in the past (use history) or future (use forecasts)."""


INTENT_CLASSIFICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "intent": {
            "type": "string",
            "enum": [
                "forecast_single_day",    # "What's the forecast for Saturday?"
                "forecast_range",          # "Forecast for next week"
                "explain_past",            # "Why were sales low yesterday?"
                "find_similar",            # "Find days like this"
                "historical_query",        # "What were sales last July?"
                "ordering_advice",         # "What should I order?"
                "general_question",        # Anything else
            ],
            "description": "The user's intent"
        },
        "target_dates": {
            "type": "array",
            "items": {"type": "string"},
            "description": "ISO dates mentioned or implied (YYYY-MM-DD)"
        },
        "needs_forecast": {"type": "boolean"},
        "needs_history": {"type": "boolean"},
        "needs_events": {"type": "boolean"},
    },
    "required": ["intent", "needs_forecast", "needs_history", "needs_events"]
}


class ForecastChat:
    """Interactive chat interface for forecasting."""
    
    def __init__(
        self,
        store: str,
        db_path: Optional[str] = None,
        llm_client=None,
    ):
        """
        Initialize the chat interface.
        
        Args:
            store: Store name (e.g., "Parksville")
            db_path: Path to sales database
            llm_client: Optional LLM client
        """
        self.context = ChatContext(store=store)
        self.db_path = db_path
        self._client = llm_client
        
        # Lazy-loaded tools
        self._forecast_engine = None
        self._rag = None
        self._event_extractor = None
    
    @property
    def client(self):
        if self._client is None:
            from aoc_analytics.llm.client import get_default_client
            self._client = get_default_client()
        return self._client
    
    @property
    def forecast_engine(self):
        if self._forecast_engine is None and self.db_path:
            try:
                from aoc_analytics.core.forecast_engine import ForecastEngine
                self._forecast_engine = ForecastEngine(db_path=self.db_path)
                self.context.has_forecast_engine = True
            except Exception as e:
                logger.warning(f"Could not initialize forecast engine: {e}")
        return self._forecast_engine
    
    @property
    def rag(self):
        if self._rag is None and self.db_path:
            try:
                from aoc_analytics.llm.rag import SalesRAG
                self._rag = SalesRAG(db_path=self.db_path, llm_client=self._client)
                self._rag.load_sales_data(self.context.store)
                self.context.has_rag = True
            except Exception as e:
                logger.warning(f"Could not initialize RAG: {e}")
        return self._rag
    
    @property
    def event_extractor(self):
        if self._event_extractor is None:
            try:
                from aoc_analytics.llm.events import EventExtractor
                self._event_extractor = EventExtractor(llm_client=self._client)
                self.context.has_events = True
            except Exception as e:
                logger.warning(f"Could not initialize event extractor: {e}")
        return self._event_extractor
    
    def _get_system_prompt(self) -> str:
        """Build the system prompt with current context."""
        return CHAT_SYSTEM_PROMPT.format(
            store=self.context.store,
            current_date=self.context.current_date.strftime("%A, %B %d, %Y"),
        )
    
    def _classify_intent(self, message: str) -> dict:
        """Classify the user's intent to route to appropriate tools."""
        prompt = f"""Classify this user message and extract relevant information.

User message: "{message}"

Today is {self.context.current_date.isoformat()}. 
Convert any relative dates (like "tomorrow", "next Saturday") to ISO format."""

        try:
            return self.client.extract_structured(
                prompt=prompt,
                schema=INTENT_CLASSIFICATION_SCHEMA,
            )
        except Exception as e:
            logger.error(f"Intent classification failed: {e}")
            return {
                "intent": "general_question",
                "target_dates": [],
                "needs_forecast": False,
                "needs_history": True,
                "needs_events": False,
            }
    
    def _gather_context(self, intent: dict) -> str:
        """Gather relevant context based on classified intent."""
        context_parts = []
        
        # Get target dates
        target_dates = []
        for date_str in intent.get("target_dates", []):
            try:
                target_dates.append(date.fromisoformat(date_str))
            except ValueError:
                continue
        
        # Forecast for future dates
        if intent.get("needs_forecast") and self.forecast_engine:
            for target_date in target_dates:
                if target_date > self.context.current_date:
                    try:
                        # TODO: Actually run forecast
                        context_parts.append(
                            f"Forecast for {target_date}: [Would run forecast engine here]"
                        )
                    except Exception as e:
                        logger.error(f"Forecast failed: {e}")
        
        # Historical data
        if intent.get("needs_history") and self.rag:
            try:
                # Search for relevant historical context
                for target_date in target_dates:
                    if target_date <= self.context.current_date:
                        similar = self.rag.search(
                            f"Sales on {target_date.strftime('%A, %B %d')}",
                            top_k=5
                        )
                        if similar:
                            context_parts.append(
                                f"Historical data for similar days:\n" +
                                "\n".join(f"- {d.to_text()}" for d in similar)
                            )
            except Exception as e:
                logger.error(f"RAG search failed: {e}")
        
        # Local events
        if intent.get("needs_events") and self.event_extractor:
            for target_date in target_dates:
                events = self.event_extractor.get_events_for_date(target_date)
                if events:
                    context_parts.append(
                        f"Events on {target_date}:\n" +
                        "\n".join(f"- {e.name} ({e.event_type.value})" for e in events)
                    )
        
        return "\n\n".join(context_parts) if context_parts else ""
    
    def _build_history_prompt(self) -> str:
        """Build prompt from conversation history."""
        if not self.context.history:
            return ""
        
        # Keep last 10 messages
        recent = self.context.history[-10:]
        return "\n\n".join(
            f"{'User' if m.role == 'user' else 'Assistant'}: {m.content}"
            for m in recent
        )
    
    def chat(self, message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: User's natural language message
        
        Returns:
            Assistant's response
        """
        # Add to history
        self.context.history.append(ChatMessage(role="user", content=message))
        
        # Classify intent
        intent = self._classify_intent(message)
        logger.debug(f"Classified intent: {intent}")
        
        # Gather relevant context
        additional_context = self._gather_context(intent)
        
        # Build the prompt
        history_prompt = self._build_history_prompt()
        
        full_prompt = f"""Previous conversation:
{history_prompt}

Additional context from data sources:
{additional_context if additional_context else "(No additional context needed)"}

User's new message: {message}

Respond helpfully based on the conversation and context above."""

        # Generate response
        try:
            response = self.client.complete(full_prompt, system=self._get_system_prompt())
        except Exception as e:
            logger.error(f"Chat completion failed: {e}")
            response = f"I'm sorry, I encountered an error: {e}"
        
        # Add response to history
        self.context.history.append(ChatMessage(role="assistant", content=response))
        
        return response
    
    def reset(self):
        """Reset conversation history."""
        self.context.history = []
    
    def get_quick_forecast(self, target_date: date) -> str:
        """Quick method to get a forecast summary for a specific date."""
        return self.chat(f"What's the forecast for {target_date.strftime('%A, %B %d, %Y')}?")
    
    def explain_day(self, target_date: date) -> str:
        """Quick method to explain a past day's performance."""
        return self.chat(
            f"Explain the sales performance on {target_date.strftime('%A, %B %d, %Y')}. "
            f"What factors contributed to the results?"
        )


# CLI interface
def main():
    """Interactive CLI for the forecast chat."""
    import argparse
    import os
    
    parser = argparse.ArgumentParser(description="Chat with the forecasting system")
    parser.add_argument("--store", default="Parksville", help="Store name")
    parser.add_argument("--db", default="aoc_sales.db", help="Database path")
    args = parser.parse_args()
    
    # Find database
    db_path = args.db
    if not os.path.exists(db_path):
        db_path = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..", db_path)
    
    print(f"🌿 AOC Forecast Chat - {args.store}")
    print("Type 'quit' to exit, 'reset' to clear history\n")
    
    chat = ForecastChat(store=args.store, db_path=db_path)
    
    while True:
        try:
            user_input = input("You: ").strip()
            
            if not user_input:
                continue
            if user_input.lower() == "quit":
                break
            if user_input.lower() == "reset":
                chat.reset()
                print("History cleared.\n")
                continue
            
            response = chat.chat(user_input)
            print(f"\nAssistant: {response}\n")
            
        except KeyboardInterrupt:
            break
        except EOFError:
            break
    
    print("\nGoodbye! 🌿")


if __name__ == "__main__":
    main()
