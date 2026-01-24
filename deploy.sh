#!/bin/bash
set -e

# =============================================================================
# AOC Analytics Cloud Run Deployment Script
# =============================================================================
# Usage:
#   ./deploy.sh          # Full build + deploy
#   ./deploy.sh --quick  # Deploy without rebuilding (config changes only)
#
# This script:
#   1. Builds AOC Analytics Docker image
#   2. Pushes to Google Artifact Registry
#   3. Deploys to Cloud Run with Cloud SQL access
#
# Environment:
#   - Reads JFK's DATABASE_URL for sales data
#   - Uses OPENAI_API_KEY for AI summaries
# =============================================================================

echo "🧠 Deploying AOC Analytics to Cloud Run..."

PROJECT_ID="sheets-writer-465009"
REGION="us-central1"
SERVICE_NAME="aoc-analytics"
IMAGE_URL="us-central1-docker.pkg.dev/$PROJECT_ID/docker-repo/$SERVICE_NAME"
CLOUD_SQL_INSTANCE="sheets-writer-465009:us-central1:jfk-db"

SKIP_BUILD=false
if [[ "${1:-}" == "--quick" ]] || [[ "${1:-}" == "-q" ]]; then
    SKIP_BUILD=true
    echo "⚡ Quick deploy mode (skipping build)"
fi

echo "📦 Project: $PROJECT_ID"
echo "🌍 Region: $REGION"
echo "🔧 Service: $SERVICE_NAME"
echo ""

# Confirm deployment
read -p "Deploy AOC to production? (y/N) " -n 1 -r
echo
if [[ ! $REPLY =~ ^[Yy]$ ]]; then
    echo "❌ Cancelled"
    exit 1
fi

# Build and push (unless --quick)
if [[ "$SKIP_BUILD" == false ]]; then
    echo "🏗️  Building AOC Analytics container..."
    gcloud builds submit --tag $IMAGE_URL --project $PROJECT_ID
fi

# Deploy to Cloud Run with Cloud SQL
echo "🚢 Deploying to Cloud Run..."
gcloud run deploy $SERVICE_NAME \
    --image $IMAGE_URL \
    --platform managed \
    --region $REGION \
    --project $PROJECT_ID \
    --allow-unauthenticated \
    --set-env-vars "PYTHONUNBUFFERED=1,CLOUD_SQL_INSTANCE=$CLOUD_SQL_INSTANCE" \
    --set-secrets "DATABASE_URL=database-url:latest,OPENAI_API_KEY=openai-api-key:latest" \
    --add-cloudsql-instances $CLOUD_SQL_INSTANCE \
    --memory 1Gi \
    --cpu 1 \
    --max-instances 5 \
    --min-instances 0 \
    --timeout 300

# Get the service URL
SERVICE_URL=$(gcloud run services describe $SERVICE_NAME --region $REGION --project $PROJECT_ID --format 'value(status.url)')

echo ""
echo "✅ AOC Analytics deployed!"
echo "🌐 Service URL: $SERVICE_URL"
echo ""
echo "📝 Next step: Update JFK's deploy-prod.sh to include:"
echo "   AOC_API_URL=$SERVICE_URL"
echo ""
echo "Test: curl $SERVICE_URL/health"
