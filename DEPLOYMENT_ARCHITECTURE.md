# AOC Brain Integration - Deployment Architecture

## Overview

The AOC Brain is an AI-powered analytics engine that analyzes JFK sales data to generate insights, predictions, and recommendations. It runs as a separate microservice alongside the JFK backend.

```
┌─────────────────────────────────────────────────────────────────────┐
│                         JFK Frontend (React)                        │
│                                                                     │
│  ┌─────────────┐  ┌─────────────┐  ┌──────────────────────────┐   │
│  │ Inventory   │  │  Signage    │  │   AOC Insights Tab       │   │
│  │    Tab      │  │    Tab      │  │   (Weather + Brain)      │   │
│  └─────────────┘  └─────────────┘  └──────────────────────────┘   │
└─────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────────┐
│                      JFK Backend (Flask)                            │
│                       Port 5000 (production)                        │
│                                                                     │
│  ┌──────────────────────────────────────────────────────────────┐  │
│  │                 /api/aoc/* endpoints                          │  │
│  │                                                               │  │
│  │  /api/aoc/status        → Weather + Regime (direct AOC call) │  │
│  │  /api/aoc/data-health   → Database status                    │  │
│  │  /api/aoc/stores        → Available stores                   │  │
│  │                                                               │  │
│  │  /api/aoc/brain/*       → Proxied to Brain API ────────────┐ │  │
│  │    /brain/summary       → Daily summary                    │ │  │
│  │    /brain/categories    → Category breakdown               │ │  │
│  │    /brain/insights      → AI insights                      │ │  │
│  │    /brain/ai-summary    → GPT-4o-mini summary              │ │  │
│  │    /brain/overview      → Complete dashboard data          │ │  │
│  │    /brain/status        → Brain service status             │ │  │
│  └──────────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────┘
                              │
             ┌────────────────┴────────────────┐
             ▼                                 ▼
┌─────────────────────────┐     ┌─────────────────────────────────────┐
│   JFK Database          │     │      AOC Brain API (FastAPI)        │
│   (cannabis_retail.db)  │     │         Port 8081                   │
│                         │     │                                     │
│  - cova_sales table     │◀────│  - Reads sales via 'sales' view     │
│  - products, stores     │     │  - Generates AI insights (GPT)      │
│  - etc.                 │     │  - Runs analysis modules            │
│                         │     │  - Learns patterns over time        │
└─────────────────────────┘     └─────────────────────────────────────┘
```

## Deployment Options

### Option 1: Local Development (Current)

Both services run locally:

```bash
# Terminal 1: JFK Backend
cd ~/Projects/JFK/backend
./run.py  # Runs on port 5000

# Terminal 2: AOC Brain API  
cd ~/Projects/aoc-analytics
./run-brain-api.sh  # Runs on port 8081
```

### Option 2: Production (GCP Cloud Run)

Deploy as two separate Cloud Run services:

```yaml
# JFK Backend Service
- Service: jfk-backend
- Port: 5000
- Memory: 1Gi
- CPU: 1
- Environment:
  - AOC_BRAIN_API_URL: https://aoc-brain-xxx.run.app

# AOC Brain API Service  
- Service: aoc-brain
- Port: 8081
- Memory: 512Mi
- CPU: 1
- Environment:
  - JFK_DB_PATH: /cloudsql/project:region:instance/cannabis_retail.db
  - OPENAI_API_KEY: (from Secret Manager)
  - DATABASE_URL: postgresql://...
```

### Option 3: Single Container (Simpler)

Run brain API as a subprocess within JFK container:

```dockerfile
# JFK Dockerfile addition
RUN pip install aoc-analytics
CMD ["sh", "-c", "python3 -m aoc_analytics.api & ./run.py"]
```

## Database Setup

The brain requires a `sales` view that maps JFK's `cova_sales` table:

```sql
CREATE VIEW IF NOT EXISTS sales AS
SELECT 
    id,
    transaction_date as date,
    transaction_time as time,
    transaction_date || ' ' || transaction_time as datetime_local,
    store_id as location,
    product_sku as sku,
    product_name,
    category,
    quantity,
    unit_price,
    total_price as subtotal,
    transaction_id as invoice_id,
    source,
    ingested_at as created_at
FROM cova_sales
WHERE total_price > 0;
```

Run this via:
```bash
cd ~/Projects/aoc-analytics
python3 -m aoc_analytics.integrations.jfk_bridge
```

## Environment Variables

### Required for Brain API:
```bash
JFK_DB_PATH=/path/to/cannabis_retail.db    # Path to JFK database
OPENAI_API_KEY=sk-...                       # For AI summary generation
```

### Required for JFK Backend:
```bash
AOC_BRAIN_API_URL=http://127.0.0.1:8081    # Brain API URL
```

## API Endpoints

### Brain Summary
```bash
GET /api/aoc/brain/summary?date=2025-12-23

Response:
{
    "date": "2025-12-23",
    "total_revenue": 7099.07,
    "total_transactions": 250,
    "total_items": 304,
    "avg_transaction": 28.40,
    "top_category": "Flower > Dried Flower",
    "top_product": "SLURMMM: LEMON ROYALE LIVE HASH ROSIN (1g)"
}
```

### Brain AI Summary
```bash
GET /api/aoc/brain/ai-summary?date=2025-12-23

Response:
{
    "date": "2025-12-23",
    "summary": "Today's performance shows a total of 250 transactions...",
    "generated_at": "2026-01-15T20:43:04.948605"
}
```

### Brain Overview (Dashboard)
```bash
GET /api/aoc/brain/overview?date=2025-12-23

Response:
{
    "summary": {...},
    "categories": [...],
    "hourly_pattern": [...],
    "insights": [...],
    "trends": [...],
    "ai_summary": "..."
}
```

## Frontend Integration

The AOC Insights Tab can be enhanced to show brain insights:

```tsx
// Add to AOCInsightsTab.tsx
const [brainOverview, setBrainOverview] = useState(null);

useEffect(() => {
    fetchBrainOverview().then(setBrainOverview);
}, []);

async function fetchBrainOverview() {
    const response = await fetch('/api/aoc/brain/overview');
    return response.json();
}
```

## Security Considerations

1. **OpenAI Key**: Store in Secret Manager, not in code
2. **Database Access**: Brain reads JFK data - ensure proper DB permissions
3. **API Authentication**: Brain endpoints should be protected by same auth as JFK
4. **Rate Limiting**: AI summaries can be expensive - consider caching

## Monitoring

Key metrics to track:
- Brain API response times
- OpenAI token usage
- Insights generation rate
- Database query performance

## Future Enhancements

1. **Scheduled Learning**: Run brain's learn cycle daily at 2 AM
2. **Hypothesis Testing**: Auto-test predictions against actual sales
3. **Memory Consolidation**: Archive old insights, keep learnings
4. **Multi-store Support**: Separate insights per store location
5. **Predictive Alerts**: Proactive notifications for anomalies
