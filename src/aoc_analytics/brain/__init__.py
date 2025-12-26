"""
AOC Brain - Autonomous Learning Agent

A continuously running AI that:
1. Monitors sales data for patterns and anomalies
2. Learns retail marketing best practices
3. Forms hypotheses about what drives sales
4. Tests predictions against outcomes
5. Refines its understanding over time

This is NOT a chatbot - it's a thinking machine that
develops its own theories and validates them.

IMPORTANT: This module piggybacks on existing signal infrastructure:
- weather_daily table for weather signals
- vibe_signals.py for sports, events, calendar
- predictor.py for similarity-based forecasting

NOTE: GPU/LLM features require optional dependencies:
    pip install aoc-analytics[brain]  # For GPU embeddings
    pip install aoc-analytics[ai]     # For full AI stack

On machines without GPU, the brain still works for:
- Signal analysis (pure Python/numpy)
- Hypothesis generation (rule-based)
- Data correlation analysis

LLM features (explanation synthesis) require Ollama running locally.
"""

import logging
import warnings

logger = logging.getLogger(__name__)

# Track what's available
BRAIN_FEATURES = {
    "signal_analysis": True,     # Always available (numpy only)
    "hypothesis_engine": True,   # Always available (pure Python)
    "memory": True,              # Always available (SQLite)
    "gpu_embeddings": False,     # Requires sentence-transformers + torch
    "llm_synthesis": False,      # Requires Ollama running
}

# === ALWAYS AVAILABLE: Signal analysis (numpy-based) ===
try:
    from aoc_analytics.brain.signal_integration import (
        SignalAnalyzer,
        SignalSnapshot,
        analyze_signals_demo,
    )
    BRAIN_FEATURES["signal_analysis"] = True
except ImportError as e:
    logger.warning(f"Signal analysis unavailable: {e}")
    SignalAnalyzer = None
    SignalSnapshot = None
    analyze_signals_demo = None
    BRAIN_FEATURES["signal_analysis"] = False

# === CORE BRAIN: Memory and learning (SQLite-based) ===
try:
    from aoc_analytics.brain.memory import BrainMemory, MemoryEntry, Hypothesis
    from aoc_analytics.brain.learner import KnowledgeLearner
    from aoc_analytics.brain.hypothesis import HypothesisEngine
    BRAIN_FEATURES["memory"] = True
    BRAIN_FEATURES["hypothesis_engine"] = True
except ImportError as e:
    logger.warning(f"Brain memory/learning unavailable: {e}")
    BrainMemory = None
    MemoryEntry = None
    Hypothesis = None
    KnowledgeLearner = None
    HypothesisEngine = None
    BRAIN_FEATURES["memory"] = False
    BRAIN_FEATURES["hypothesis_engine"] = False

# === GPU FEATURES: Embeddings (requires torch + sentence-transformers) ===
try:
    import torch
    from sentence_transformers import SentenceTransformer
    BRAIN_FEATURES["gpu_embeddings"] = torch.cuda.is_available()
    if not BRAIN_FEATURES["gpu_embeddings"]:
        logger.info("PyTorch available but no GPU detected - embeddings will use CPU")
except ImportError:
    BRAIN_FEATURES["gpu_embeddings"] = False

# === FULL AGENT: Requires memory + optional LLM ===
try:
    from aoc_analytics.brain.agent import BrainAgent
except ImportError as e:
    logger.warning(f"BrainAgent unavailable: {e}")
    BrainAgent = None

# === CHECK FOR OLLAMA (LLM synthesis) ===
def check_ollama_available() -> bool:
    """Check if Ollama is running and accessible."""
    try:
        import httpx
        response = httpx.get("http://localhost:11434/api/tags", timeout=2.0)
        return response.status_code == 200
    except Exception:
        return False

# Don't check at import time - too slow
# BRAIN_FEATURES["llm_synthesis"] = check_ollama_available()


def get_available_features() -> dict:
    """Return dict of available brain features."""
    return BRAIN_FEATURES.copy()


def require_feature(feature: str) -> None:
    """Raise ImportError if feature isn't available."""
    if not BRAIN_FEATURES.get(feature, False):
        raise ImportError(
            f"Brain feature '{feature}' not available. "
            f"Install with: pip install aoc-analytics[brain]"
        )


# Build __all__ based on what's available
__all__ = [
    "BRAIN_FEATURES",
    "get_available_features",
    "require_feature",
    "check_ollama_available",
]

if SignalAnalyzer is not None:
    __all__.extend(["SignalAnalyzer", "SignalSnapshot", "analyze_signals_demo"])

if BrainMemory is not None:
    __all__.extend(["BrainMemory", "MemoryEntry", "Hypothesis", "KnowledgeLearner", "HypothesisEngine"])

if BrainAgent is not None:
    __all__.append("BrainAgent")
