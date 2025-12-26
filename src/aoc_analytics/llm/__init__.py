"""
LLM-powered analytics layer for AOC.

This module provides:
- Event extraction from local news/calendars
- Anomaly explanation for forecast misses
- RAG over sales history for semantic queries
- Natural language chat interface for forecasting
- Reddit signal extraction for local events

NOTE: This module requires optional dependencies:
    pip install aoc-analytics[llm]

On machines without LLM dependencies, this module will be empty
but won't crash the rest of the application.
"""

import logging

logger = logging.getLogger(__name__)

# Track what's available
LLM_FEATURES = {
    "client": False,
    "events": False,
    "explainer": False,
    "rag": False,
    "chat": False,
    "reddit": False,
}

__all__ = ["LLM_FEATURES", "get_available_features"]


def get_available_features() -> dict:
    """Return dict of available LLM features."""
    return LLM_FEATURES.copy()


# === LLM Client ===
try:
    from aoc_analytics.llm.client import LLMClient, get_llm_client
    LLM_FEATURES["client"] = True
    __all__.extend(["LLMClient", "get_llm_client"])
except ImportError as e:
    logger.debug(f"LLM client unavailable: {e}")
    LLMClient = None
    get_llm_client = None

# === Event Extraction ===
try:
    from aoc_analytics.llm.events import EventExtractor, LocalEvent
    LLM_FEATURES["events"] = True
    __all__.extend(["EventExtractor", "LocalEvent"])
except ImportError as e:
    logger.debug(f"Event extraction unavailable: {e}")
    EventExtractor = None
    LocalEvent = None

# === Anomaly Explainer ===
try:
    from aoc_analytics.llm.explainer import AnomalyExplainer
    LLM_FEATURES["explainer"] = True
    __all__.append("AnomalyExplainer")
except ImportError as e:
    logger.debug(f"Anomaly explainer unavailable: {e}")
    AnomalyExplainer = None

# === RAG ===
try:
    from aoc_analytics.llm.rag import SalesRAG
    LLM_FEATURES["rag"] = True
    __all__.append("SalesRAG")
except ImportError as e:
    logger.debug(f"RAG unavailable: {e}")
    SalesRAG = None

# === Chat ===
try:
    from aoc_analytics.llm.chat import ForecastChat
    LLM_FEATURES["chat"] = True
    __all__.append("ForecastChat")
except ImportError as e:
    logger.debug(f"Chat unavailable: {e}")
    ForecastChat = None

# === Reddit ===
try:
    from aoc_analytics.llm.reddit import (
        RedditClient, 
        RedditPost, 
        RedditConfig,
        RedditSignalExtractor,
        check_reddit_credentials,
    )
    LLM_FEATURES["reddit"] = True
    __all__.extend([
        "RedditClient",
        "RedditPost",
        "RedditConfig",
        "RedditSignalExtractor",
        "check_reddit_credentials",
    ])
except ImportError as e:
    logger.debug(f"Reddit client unavailable: {e}")
    RedditClient = None
    RedditPost = None
    RedditConfig = None
    RedditSignalExtractor = None
    check_reddit_credentials = None
