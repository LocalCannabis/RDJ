#!/bin/bash
# =============================================================================
# AOC Brain API Deployment Script
# =============================================================================
# This script starts the AOC Brain API server that powers JFK's insights tab.
#
# The brain API provides:
# - AI-generated sales insights (GPT-4o-mini)
# - Category performance analysis
# - Daily/hourly sales patterns
# - Executive summaries
#
# Environment Variables Required:
# - JFK_DB_PATH: Path to JFK's SQLite database (local) or set for PostgreSQL
# - OPENAI_API_KEY: For AI summary generation (or loads from JFK .env)
# - AOC_BRAIN_PORT: Port to run on (default: 8081)
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
AOC_ROOT="${SCRIPT_DIR}"

# Configuration
PORT="${AOC_BRAIN_PORT:-8081}"
HOST="${AOC_BRAIN_HOST:-127.0.0.1}"

echo "=========================================="
echo "  AOC Brain API Server"
echo "=========================================="
echo "  Port: $PORT"
echo "  Host: $HOST"
echo "=========================================="

# Activate virtual environment
if [ -d "${AOC_ROOT}/.venv" ]; then
    source "${AOC_ROOT}/.venv/bin/activate"
elif [ -d "${AOC_ROOT}/venv" ]; then
    source "${AOC_ROOT}/venv/bin/activate"
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv "${AOC_ROOT}/.venv"
    source "${AOC_ROOT}/.venv/bin/activate"
    pip install --upgrade pip
    pip install -e "${AOC_ROOT}"
fi

# Set up Python path
export PYTHONPATH="${AOC_ROOT}/src:${PYTHONPATH}"

# AOC's own database (historical sales + weather)
if [ -z "$AOC_DB_PATH" ]; then
    AOC_DB="${AOC_ROOT}/aoc_analytics.db"
    if [ -f "$AOC_DB" ]; then
        export AOC_DB_PATH="$AOC_DB"
        echo "Using AOC database: $AOC_DB_PATH"
    else
        echo "WARNING: AOC database not found at $AOC_DB"
        echo "Run: python scripts/create_aoc_database.py to create it"
    fi
fi

# Load OpenAI key from JFK .env if not set
if [ -z "$OPENAI_API_KEY" ]; then
    JFK_ENV="${HOME}/Projects/JFK/backend/.env"
    if [ -f "$JFK_ENV" ]; then
        export OPENAI_API_KEY=$(grep "^OPENAI_API_KEY=" "$JFK_ENV" | cut -d'=' -f2 | tr -d '"')
        if [ -n "$OPENAI_API_KEY" ]; then
            echo "Loaded OpenAI key from JFK .env"
        fi
    fi
fi

# Start the server
echo ""
echo "Starting AOC Brain API..."
echo ""

python3 -c "
import uvicorn
from aoc_analytics.api.server import create_app

app = create_app()
uvicorn.run(
    app, 
    host='${HOST}', 
    port=${PORT},
    log_level='info'
)
"
