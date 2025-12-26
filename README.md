# AOC Analytics

**Weather-aware retail analytics for cannabis retail.**

## The Philosophy

> **"Weather is not an insight — it's background radiation."**

Traditional retail analytics treat weather as just another variable to correlate with sales. AOC treats weather fundamentally differently: it's an **a priori factor** that must be normalized out BEFORE any other analysis.

This means:
1. Weather effects are removed FIRST (Tier 0: Context Normalization)
2. ALL other analytics operate on **context-normalized demand**
3. The residual after weather correction is the TRUE signal

## Installation

```bash
pip install aoc-analytics
```

Or with ML dependencies:

```bash
pip install aoc-analytics[ml]
```

### Optional Dependencies

The package uses optional dependency groups to support machines with different capabilities:

| Install Command | What You Get | Requirements |
|-----------------|--------------|--------------|
| `pip install aoc-analytics` | Core analytics, signals, forecasting | Python 3.10+ |
| `pip install aoc-analytics[ml]` | + scikit-learn, advanced forecasting | ~500MB |
| `pip install aoc-analytics[llm]` | + LLM integrations (OpenAI, Anthropic, Ollama) | API keys |
| `pip install aoc-analytics[brain]` | + Learning agent, hypothesis engine | LLM deps |
| `pip install aoc-analytics[gpu]` | + PyTorch, sentence-transformers | CUDA GPU |
| `pip install aoc-analytics[all]` | Everything | GPU + all deps |

**For machines WITHOUT a GPU:**
```bash
pip install aoc-analytics[ml]  # Full forecasting, no GPU required
```

**For machines WITH an NVIDIA GPU:**
```bash
pip install aoc-analytics[brain,gpu]  # Full AI brain features
```

### Runtime Feature Detection

The brain and LLM modules gracefully degrade on machines without required dependencies:

```python
from aoc_analytics import brain, llm

# Check what's available
print(brain.BRAIN_FEATURES)
# {'memory': True, 'learner': True, 'hypothesis': True, 'agent': True, 'signal_integration': True}

print(llm.LLM_FEATURES)
# {'client': True, 'events': True, 'explainer': True, 'rag': True, 'chat': True}

# On a machine without GPU/torch:
# {'memory': False, 'learner': False, ...}  # All False, but no import errors
```

This allows the same codebase to run everywhere - GPU features simply won't be available on machines without CUDA.

## Quick Start

### As a Library

```python
from aoc_analytics import AOCClient

# Connect to AOC service
aoc = AOCClient(base_url="http://localhost:8001")

# Get current behavioral signals (weather-normalized context)
signals = aoc.get_signals("central")
print(f"At-home propensity: {signals['at_home']}")
print(f"Payday boost: {signals['payday']}")

# Get hero products for digital signage
heroes = aoc.get_heroes(
    store="central",
    lens="margin_mix",
    limit=12
)

# Forecast demand for a SKU
forecast = aoc.forecast_demand(
    sku="ABC123",
    store="central",
    days=7
)
print(f"7-day forecast: {forecast['total_forecast']} units")
```

### As a Service

```bash
# Start the API server
aoc-server --host 0.0.0.0 --port 8001

# Or with uvicorn directly
uvicorn aoc_analytics.api.server:app --host 0.0.0.0 --port 8001
```

## Architecture

```
aoc-analytics/
├── src/aoc_analytics/
│   ├── core/                    # THE KEYSTONE - analytics engine
│   │   ├── signals/
│   │   │   ├── builder.py       # Behavioral signals (weather → propensity)
│   │   │   └── payday_index.py  # Payday proximity scoring
│   │   ├── predictor.py         # Demand forecasting
│   │   ├── hero_signals.py      # Hero product ranking
│   │   └── anomaly_registry.py  # Anomaly detection
│   │
│   ├── brain/                   # 🧠 Learning agent (optional, GPU)
│   │   ├── memory.py            # Vector-based memory store
│   │   ├── learner.py           # Pattern learning from sales data
│   │   ├── hypothesis.py        # Testable hypothesis generation
│   │   ├── agent.py             # Autonomous learning agent
│   │   └── signal_integration.py # Piggyback on existing signals
│   │
│   ├── llm/                     # 🤖 LLM integrations (optional)
│   │   ├── client.py            # Ollama/OpenAI client
│   │   ├── events.py            # Event extraction from text
│   │   ├── explainer.py         # Human-readable explanations
│   │   ├── rag.py               # Retrieval-augmented generation
│   │   └── chat.py              # Conversational interface
│   │
│   ├── api/                     # FastAPI REST interface
│   │   ├── server.py            # Application factory
│   │   ├── routes/
│   │   │   ├── signals.py       # /signals/* endpoints
│   │   │   ├── forecast.py      # /forecast/* endpoints
│   │   │   ├── heroes.py        # /heroes/* endpoints
│   │   │   └── anomalies.py     # /anomalies/* endpoints
│   │   └── middleware.py        # Auth, rate limiting
│   │
│   ├── client/                  # Python SDK for consumers
│   │   └── client.py            # AOCClient class
│   │
│   └── db/                      # Database layer
│       ├── connection.py        # SQLite connection management
│       └── migrations/          # Schema migrations
```

## The Keystone: `_score_at_home()`

This function is the mathematical embodiment of weather-as-a-priori:

```python
def _score_at_home(weather, mood=None):
    """
    Convert weather conditions into behavioral propensity.
    Applied BEFORE any sales correlation.
    """
    if not weather:
        base = 0.35
    else:
        rain_factor = min(weather["precip_mm"] / 6.0, 1.0)
        cold_factor = max(0, (12 - weather["feels_like"]) / 15.0)
        wind_factor = min(weather["wind_kph"] / 40.0, 1.0)
        snow_factor = min(weather["snow_share"] * 1.25, 1.0)
        
        base = 0.1 + 0.45*rain + 0.3*cold + 0.1*wind + 0.15*snow
    
    return blend_with_mood(base, mood, weight=0.35)
```

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Service health check |
| `/signals/current` | GET | Current behavioral context |
| `/signals/range` | GET | Historical signals |
| `/forecast/demand` | POST | SKU demand forecast |
| `/heroes/ranked` | GET | Ranked hero products |
| `/anomalies/active` | GET | Active anomalies |

See [API Contract](docs/API_CONTRACT.md) for full specification.

## Configuration

Environment variables:

```bash
# Database
AOC_DATABASE_URL=sqlite:///path/to/weather_sales.db

# API
AOC_HOST=0.0.0.0
AOC_PORT=8001
AOC_API_KEY=your-secret-key

# Optional: External services
WEATHER_API_KEY=open-meteo-key
```

## Integration with LocalBot/JFK

```python
# In your Flask backend
from aoc_analytics import AOCClient

aoc = AOCClient(
    base_url=os.getenv("AOC_API_URL"),
    api_key=os.getenv("AOC_API_KEY")
)

@app.route("/api/signage/heroes")
def get_signage_heroes():
    signals = aoc.get_signals(store="central")
    heroes = aoc.get_heroes(
        store="central",
        lens="margin_mix" if signals["at_home"] > 0.5 else "velocity"
    )
    return jsonify(heroes)
```

## License

Proprietary - Local Cannabis Co. All Rights Reserved.
