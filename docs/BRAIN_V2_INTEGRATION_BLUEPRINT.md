# Brain v2 Integration Blueprint

**Created:** 2026-01-15  
**Branch:** `brain-v2-advanced`  
**Status:** Planning → Ready for Work Integration

---

## Executive Summary

We have two divergent codebases:
- **`main` (production):** PostgreSQL-compatible, Cloud Scheduler jobs, stable
- **`brain-v2-advanced` (local):** ~12,700 lines of advanced AI/ML, SQLite-only

This document is the roadmap for merging brain-v2's capabilities into production.

---

## Current State

### Production (`origin/main`)
```
✅ PostgreSQL via db_adapter.py
✅ Cloud Scheduler integration
✅ Error handling with rollback
✅ Schema auto-creation
✅ Weather jobs running
✅ Mood jobs running
```

### Brain v2 (`brain-v2-advanced`)
```
❌ SQLite only (hardcoded sqlite3.connect())
❌ Cron-based daemon (not Cloud Scheduler)
❌ No error handling for missing tables
❌ Hardcoded business assumptions
✅ Curiosity Engine (autonomous hypothesis testing)
✅ Backtesting framework (prove accuracy)
✅ ROI Tracker (dollar value calculations)
✅ Signal Magnitude Learning
✅ Predictive Calendar
✅ 40+ hypothesis templates
```

---

## Integration Phases

### Phase 1: Database Compatibility (Priority: CRITICAL)

**Goal:** Make brain modules work with PostgreSQL

**Files to modify:**
| File | Lines | Changes Needed |
|------|-------|----------------|
| `backtesting.py` | 673 | Replace `sqlite3.connect()` with `db_adapter` |
| `curiosity_engine.py` | 757 | Replace `sqlite3.connect()` with `db_adapter` |
| `signal_magnitude.py` | 662 | Replace `sqlite3.connect()` with `db_adapter` |
| `roi_tracker.py` | 532 | Replace `sqlite3.connect()` with `db_adapter` |
| `cross_store.py` | 496 | Replace `sqlite3.connect()` with `db_adapter` |
| `event_correlation.py` | 474 | Replace `sqlite3.connect()` with `db_adapter` |
| `agent.py` | 477 | Replace `sqlite3.connect()` with `db_adapter` |
| + 8 more modules | ~2000 | Same pattern |

**Implementation pattern:**
```python
# BEFORE (current brain code)
import sqlite3
def get_connection(self):
    return sqlite3.connect(self.db_path)

# AFTER (production-compatible)
from aoc_analytics.core.db_adapter import get_connection

def get_connection(self):
    return get_connection()  # Uses DATABASE_URL env var
```

**Verification:**
```bash
# Test locally with SQLite
DATABASE_URL=sqlite:///aoc_sales.db python -c "from aoc_analytics.brain import *"

# Test with prod snapshot
DATABASE_URL=$PROD_DATABASE_URL python -c "from aoc_analytics.brain import *"
```

---

### Phase 2: Cloud Scheduler Integration (Priority: HIGH)

**Goal:** Replace cron daemon with Cloud Scheduler jobs

**Current daemon structure:**
```python
# daemon.py - runs everything sequentially via cron
class LearningDaemon:
    def run_all(self):
        self.run_signal_magnitude()    # ~2 min
        self.run_product_weather()     # ~1 min
        self.run_time_of_day()         # ~1 min
        self.run_cross_store()         # ~3 min
        self.run_backtesting()         # ~5 min
        self.run_roi_calculation()     # ~1 min
```

**New Cloud Scheduler approach:**

| Job Name | Schedule | Endpoint | Timeout |
|----------|----------|----------|---------|
| `brain-signal-magnitude` | `0 3 * * *` | `/jobs/brain/signal-magnitude` | 5min |
| `brain-product-weather` | `0 3 * * *` | `/jobs/brain/product-weather` | 3min |
| `brain-backtesting` | `0 4 * * 0` | `/jobs/brain/backtesting` | 15min |
| `brain-roi-report` | `0 5 1 * *` | `/jobs/brain/roi-report` | 5min |
| `brain-curiosity` | `0 2 * * *` | `/jobs/brain/curiosity` | 30min |

**New file needed:** `src/aoc_analytics/jobs/brain_jobs.py`
```python
"""
Cloud Scheduler endpoints for brain learning jobs.

Each job is idempotent and can be retried safely.
"""
from flask import Blueprint, jsonify
import logging

brain_jobs = Blueprint('brain_jobs', __name__)
logger = logging.getLogger(__name__)

@brain_jobs.route('/jobs/brain/signal-magnitude', methods=['POST'])
def run_signal_magnitude():
    """Learn signal magnitudes from historical data."""
    try:
        from aoc_analytics.brain.signal_magnitude import SignalMagnitudeLearner
        learner = SignalMagnitudeLearner()
        results = learner.learn_all_signal_impacts()
        return jsonify({"status": "success", "signals_learned": len(results)})
    except Exception as e:
        logger.exception("Signal magnitude job failed")
        return jsonify({"status": "error", "error": str(e)}), 500

@brain_jobs.route('/jobs/brain/backtesting', methods=['POST'])
def run_backtesting():
    """Weekly backtest to measure prediction accuracy."""
    try:
        from aoc_analytics.brain.backtesting import Backtester
        tester = Backtester()
        results = tester.run_full_backtest(days=180)
        return jsonify({
            "status": "success",
            "mape": results.overall_mape,
            "directional_accuracy": results.overall_directional_accuracy
        })
    except Exception as e:
        logger.exception("Backtesting job failed")
        return jsonify({"status": "error", "error": str(e)}), 500

# ... more endpoints
```

**Cloud Build deployment update:**
```yaml
# Add to cloudbuild-jobs.yaml
- id: 'deploy-brain-jobs'
  name: 'gcr.io/cloud-builders/gcloud'
  args:
    - 'scheduler'
    - 'jobs'
    - 'create'
    - 'http'
    - 'brain-signal-magnitude'
    - '--schedule=0 3 * * *'
    - '--uri=${_SERVICE_URL}/jobs/brain/signal-magnitude'
    - '--http-method=POST'
    - '--oidc-service-account-email=${_SERVICE_ACCOUNT}'
```

---

### Phase 3: Fix Known Signal Issues (Priority: MEDIUM)

**Canucks signal is inverted:**
```python
# Current (WRONG)
"canucks_home": -0.027  # Predicts decrease

# Should be (based on backtest)
"canucks_home": +0.067  # Actually increases sales
```

**Location:** `backtesting.py` line ~155, signal defaults dict

**Academic signal over-predicts:**
```python
# Current
"academic_finals": 0.30  # Predicts +30%

# Should be
"academic_finals": -0.10  # Actually -6 to -15%
```

---

### Phase 4: Error Handling & Resilience (Priority: MEDIUM)

**Add to each brain module:**
```python
def get_connection(self):
    """Get database connection with error handling."""
    try:
        conn = get_connection()
        # Verify required tables exist
        self._verify_schema(conn)
        return conn
    except Exception as e:
        logger.error(f"Database connection failed: {e}")
        raise BrainDatabaseError(f"Cannot connect to database: {e}")

def _verify_schema(self, conn):
    """Check that required tables exist."""
    required_tables = ['sales', 'weather_hourly']
    # ... validation logic
```

**Add retry decorator for transient failures:**
```python
from tenacity import retry, stop_after_attempt, wait_exponential

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
def run_with_retry(self):
    # ... job logic
```

---

### Phase 5: Output Storage (Priority: LOW)

**Current:** Brain outputs to local JSON files in `brain/data/`
**Problem:** Cloud Run is stateless - files disappear

**Options:**
1. **Cloud Storage bucket** - Store JSON outputs in GCS
2. **Database tables** - Store in PostgreSQL 
3. **BigQuery** - For analytics/historical tracking

**Recommended:** Database tables for active data, GCS for archives

**New tables needed:**
```sql
CREATE TABLE brain_backtest_results (
    id SERIAL PRIMARY KEY,
    run_date TIMESTAMP DEFAULT NOW(),
    period_start DATE,
    period_end DATE,
    mape FLOAT,
    directional_accuracy FLOAT,
    correlation FLOAT,
    results_json JSONB
);

CREATE TABLE brain_signal_magnitudes (
    id SERIAL PRIMARY KEY,
    updated_at TIMESTAMP DEFAULT NOW(),
    signal_name VARCHAR(100),
    signal_type VARCHAR(50),
    avg_lift FLOAT,
    confidence FLOAT,
    sample_size INTEGER
);

CREATE TABLE brain_predictions (
    id SERIAL PRIMARY KEY,
    prediction_date DATE,
    created_at TIMESTAMP DEFAULT NOW(),
    predicted_lift FLOAT,
    events JSONB,
    actual_lift FLOAT,  -- Filled in after the fact
    was_accurate BOOLEAN
);
```

---

## Testing Strategy

### Local Testing (SQLite)
```bash
# 1. Get a recent prod data dump
pg_dump $PROD_DATABASE_URL --data-only -t sales -t weather_hourly > prod_data.sql

# 2. Load into local SQLite (use sqlite-utils or manual conversion)
# 3. Run brain modules
python -m aoc_analytics.brain.daemon --dry-run
```

### Staging Testing
```bash
# Deploy to staging Cloud Run
gcloud run deploy aoc-analytics-staging \
  --source . \
  --set-env-vars="DATABASE_URL=$STAGING_DB"

# Trigger brain jobs manually
curl -X POST https://aoc-analytics-staging-xxx.run.app/jobs/brain/backtesting
```

### Production Rollout
1. Deploy with brain endpoints disabled (feature flag)
2. Enable one job at a time (signal-magnitude first)
3. Monitor logs for 48 hours
4. Enable next job
5. Full rollout after 1 week stable

---

## File Checklist

### Must Merge from `main`
- [ ] `src/aoc_analytics/core/db_adapter.py` (161 lines)
- [ ] Updated `src/aoc_analytics/core/weather.py` (PostgreSQL functions)
- [ ] Updated `src/aoc_analytics/jobs/scheduler.py` (error handling)

### Must Create New
- [ ] `src/aoc_analytics/jobs/brain_jobs.py` (Cloud Scheduler endpoints)
- [ ] `deploy/cloudbuild-brain.yaml` (scheduler job definitions)
- [ ] `migrations/brain_tables.sql` (new schema)

### Must Modify in brain-v2
- [ ] All 16 brain modules: `sqlite3` → `db_adapter`
- [ ] `daemon.py`: Convert to job endpoints
- [ ] `backtesting.py`: Fix Canucks/Academic signals
- [ ] `__init__.py`: Add Cloud Run compatibility

---

## Quick Start (At Work)

```bash
# 1. Fetch the brain branch
cd ~/Projects/aoc-analytics
git fetch origin
git checkout brain-v2-advanced

# 2. Get db_adapter from main
git checkout main -- src/aoc_analytics/core/db_adapter.py

# 3. Create the brain_jobs.py file (copy from this doc or generate)

# 4. Start converting modules (pick one, test, repeat)
# Example: backtesting.py
```

---

## Success Metrics

| Metric | Target | Measurement |
|--------|--------|-------------|
| Directional accuracy | >75% | Weekly backtest job |
| MAPE | <10% | Weekly backtest job |
| Rush day recall | >80% | Monthly review |
| Job success rate | >99% | Cloud Monitoring |
| ROI generated | >$1500/mo | Monthly ROI report |

---

## Timeline Estimate

| Phase | Effort | Dependencies |
|-------|--------|--------------|
| Phase 1: DB Compat | 4-6 hours | None |
| Phase 2: Scheduler | 2-3 hours | Phase 1 |
| Phase 3: Signal Fixes | 1 hour | Phase 1 |
| Phase 4: Error Handling | 2-3 hours | Phase 1 |
| Phase 5: Output Storage | 4-6 hours | Phase 2 |

**Total:** ~2-3 days of focused work

---

## Open Questions

1. **Curiosity Engine:** Run daily or weekly? (Currently assumes nightly)
2. **LLM Synthesis:** Do we need Ollama in Cloud Run? (Heavy dependency)
3. **Multi-store:** Should brain learn per-store or aggregate?
4. **Alerting:** Where should brain alerts go? (Slack? Email? JFK dashboard?)

---

## Notes

- Brain outputs JSON files - need to decide: GCS, DB, or both
- LLM features (`llm_synthesis.py`) may need to be disabled in Cloud Run
- Consider feature flags for gradual rollout
- Backtest weekly, ROI monthly, signals nightly

---

*This document lives in `docs/BRAIN_V2_INTEGRATION_BLUEPRINT.md` on the `brain-v2-advanced` branch.*
