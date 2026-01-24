# AOC Analytics API Server
# Provides AI-powered analytics for JFK retail management
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc \
    libpq-dev \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy pyproject.toml and README first for better caching
COPY pyproject.toml README.md ./

# Install the package (non-editable for Docker)
RUN pip install --no-cache-dir ".[llm]"

# Install psycopg2-binary for PostgreSQL support
RUN pip install --no-cache-dir psycopg2-binary

# Copy source code
COPY src/ src/

# Set up Python path
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

# Cloud Run provides PORT, default to 8080
ENV PORT=8080

# Run the FastAPI server with uvicorn
CMD exec uvicorn aoc_analytics.api.server:app --host 0.0.0.0 --port $PORT
