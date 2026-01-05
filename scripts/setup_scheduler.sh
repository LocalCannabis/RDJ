#!/bin/bash
# Setup Cloud Scheduler jobs for AOC Analytics mood pipeline
#
# These jobs keep the mood/vibe data fresh:
# - Hourly: Weather sync  
# - Daily: Mood features generation
# - Weekly: Adaptive weights recalculation
#
# Jobs hit JFK API endpoints which run the AOC pipeline internally.

set -e

# Configuration
PROJECT_ID="${PROJECT_ID:-sheets-writer-465009}"
REGION="${REGION:-us-central1}"

echo "🧠 Setting up AOC Analytics scheduled jobs..."
echo "   Project: $PROJECT_ID"
echo "   Region: $REGION"
echo ""

# Get JFK API URL
echo "🔍 Looking for JFK API service..."
JFK_URL=$(gcloud run services describe jfk-api \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --format 'value(status.url)' 2>/dev/null || true)

if [ -z "$JFK_URL" ]; then
    echo "❌ JFK API service not found!"
    exit 1
fi

echo "📡 JFK API URL: $JFK_URL"
echo ""

# Get or generate scheduler key
if [ -z "$SCHEDULER_KEY" ]; then
    echo "⚠️  SCHEDULER_KEY not set."
    echo "   Fetching from Cloud Run service..."
    
    # Try to get existing key from service
    SCHEDULER_KEY=$(gcloud run services describe jfk-api \
        --platform managed \
        --region $REGION \
        --project $PROJECT_ID \
        --format 'value(spec.template.spec.containers[0].env)' 2>/dev/null | grep -oP 'SCHEDULER_KEY=\K[^,]+' || true)
    
    if [ -z "$SCHEDULER_KEY" ]; then
        echo "   Generating new SCHEDULER_KEY..."
        SCHEDULER_KEY=$(openssl rand -hex 32)
        echo ""
        echo "   Generated SCHEDULER_KEY: $SCHEDULER_KEY"
        echo ""
        echo "   Set it on Cloud Run:"
        echo "   gcloud run services update jfk-api \\"
        echo "       --region $REGION \\"
        echo "       --project $PROJECT_ID \\"
        echo "       --set-env-vars SCHEDULER_KEY=$SCHEDULER_KEY"
        echo ""
        read -p "Press Enter after setting the env var, or Ctrl+C to abort..."
    fi
fi

echo ""

# 1. Hourly: Weather sync (at :05)
echo "📅 Creating hourly weather job..."
gcloud scheduler jobs delete aoc-weather-hourly \
    --project=$PROJECT_ID \
    --location=$REGION \
    --quiet 2>/dev/null || true

gcloud scheduler jobs create http aoc-weather-hourly \
    --project=$PROJECT_ID \
    --location=$REGION \
    --schedule="5 * * * *" \
    --time-zone="America/Vancouver" \
    --uri="$JFK_URL/api/analytics/jobs/weather" \
    --http-method=POST \
    --headers="X-Scheduler-Key=$SCHEDULER_KEY" \
    --attempt-deadline="300s" \
    --description="Fetch weather for all stores (hourly)"

# 2. Daily: Mood features (at 5am)
echo "📅 Creating daily mood job..."
gcloud scheduler jobs delete aoc-mood-daily \
    --project=$PROJECT_ID \
    --location=$REGION \
    --quiet 2>/dev/null || true

gcloud scheduler jobs create http aoc-mood-daily \
    --project=$PROJECT_ID \
    --location=$REGION \
    --schedule="0 5 * * *" \
    --time-zone="America/Vancouver" \
    --uri="$JFK_URL/api/analytics/jobs/mood" \
    --http-method=POST \
    --headers="X-Scheduler-Key=$SCHEDULER_KEY" \
    --attempt-deadline="600s" \
    --description="Generate daily mood/vibe features"

# 3. Weekly: Weights recalculation (Sunday 3am)
echo "📅 Creating weekly weights job..."
gcloud scheduler jobs delete aoc-weights-weekly \
    --project=$PROJECT_ID \
    --location=$REGION \
    --quiet 2>/dev/null || true

gcloud scheduler jobs create http aoc-weights-weekly \
    --project=$PROJECT_ID \
    --location=$REGION \
    --schedule="0 3 * * 0" \
    --time-zone="America/Vancouver" \
    --uri="$JFK_URL/api/analytics/jobs/weights" \
    --http-method=POST \
    --headers="X-Scheduler-Key=$SCHEDULER_KEY" \
    --attempt-deadline="900s" \
    --description="Recalculate adaptive weights (weekly)"

echo ""
echo "✅ Scheduler jobs created!"
echo ""
gcloud scheduler jobs list --project=$PROJECT_ID --location=$REGION
echo ""
echo "To backfill mood data for the last 30 days:"
echo "  curl -X POST '$JFK_URL/api/analytics/jobs/mood?backfill_days=30' \\"
echo "       -H 'X-Scheduler-Key: $SCHEDULER_KEY'"
echo ""
echo "To run a job immediately:"
echo "  gcloud scheduler jobs run aoc-mood-daily --project=$PROJECT_ID --location=$REGION"
