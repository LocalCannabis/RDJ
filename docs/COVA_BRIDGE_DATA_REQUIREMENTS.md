# Cova-Bridge Data Requirements

## Overview

The brain needs regular data exports from Cova to:
1. Track daily sales patterns
2. Learn product relationships
3. Detect anomalies in real-time
4. Improve prediction accuracy

---

## Required Reports

### 1. **Daily Sales Export** (CRITICAL)
**Frequency:** Daily, after close (~11pm)
**Format:** CSV or JSON

| Field | Required | Notes |
|-------|----------|-------|
| `transaction_id` | ✅ | Unique ID |
| `datetime` | ✅ | Full timestamp with time zone |
| `sku` | ✅ | Product SKU |
| `product_name` | ✅ | Full product name |
| `category` | ✅ | Cova category |
| `quantity` | ✅ | Units sold |
| `unit_price` | ✅ | Sale price |
| `subtotal` | ✅ | Line total |
| `cost` | ⭕ | Product cost (for margin) |
| `discount_applied` | ⭕ | Any discount |
| `location` | ✅ | Store location |
| `register_id` | ⭕ | Which register |
| `employee_id` | ⭕ | For staffing analysis |

**Why daily?** This is the practical limit until API access. Brain will process overnight and have predictions ready by morning.

---

### 2. **Inventory Snapshot** (HIGH PRIORITY)
**Frequency:** Daily at close, plus on-demand
**Format:** CSV or JSON

| Field | Required | Notes |
|-------|----------|-------|
| `sku` | ✅ | Product SKU |
| `product_name` | ✅ | Full name |
| `category` | ✅ | Category |
| `quantity_on_hand` | ✅ | Current stock |
| `quantity_reserved` | ⭕ | Reserved/held |
| `quantity_on_order` | ⭕ | Incoming |
| `last_received_date` | ⭕ | When last restocked |
| `cost` | ⭕ | Unit cost |
| `location` | ✅ | Store |

**Why?** Powers inventory simulation, stockout prediction, reorder recommendations.

---

### 3. **Product Catalog** (WEEKLY)
**Frequency:** Weekly, or when products added/removed
**Format:** CSV or JSON

| Field | Required | Notes |
|-------|----------|-------|
| `sku` | ✅ | Product SKU |
| `product_name` | ✅ | Full name |
| `brand` | ✅ | Brand name |
| `category` | ✅ | Cova category |
| `subcategory` | ⭕ | If available |
| `thc_min` | ⭕ | THC range |
| `thc_max` | ⭕ | THC range |
| `cbd_min` | ⭕ | CBD range |
| `cbd_max` | ⭕ | CBD range |
| `unit_size` | ⭕ | Package size |
| `unit_price` | ✅ | Current price |
| `cost` | ⭕ | Cost |
| `active` | ✅ | Is product active? |
| `created_date` | ⭕ | When added |

**Why?** Powers product relationship analysis, new product detection, category trends.

---

### 4. **Daily Summary** (DAILY)
**Frequency:** Daily at close (after last transaction)
**Format:** JSON preferred

```json
{
  "date": "2025-12-27",
  "location": "Victoria Drive",
  "summary": {
    "total_transactions": 245,
    "total_units": 456,
    "total_revenue": 10234.56,
    "total_cost": 6543.21,
    "gross_profit": 3691.35,
    "avg_basket_size": 1.86,
    "avg_basket_value": 41.77
  },
  "by_hour": [
    {"hour": 9, "transactions": 12, "units": 23, "revenue": 456.78},
    {"hour": 10, "transactions": 18, "units": 34, "revenue": 789.01},
    // ... etc
  ],
  "by_category": [
    {"category": "Pre-Rolls", "units": 123, "revenue": 2345.67},
    {"category": "Flower", "units": 89, "revenue": 3456.78},
    // ... etc
  ],
  "top_products": [
    {"sku": "ABC123", "name": "Product X", "units": 15, "revenue": 234.56},
    // ... top 20
  ]
}
```

**Why?** Quick daily health check, trend analysis, immediate anomaly detection.

---

## Export Schedule

| Report | Frequency | Time | Priority |
|--------|-----------|------|----------|
| Daily Sales Export | Daily | ~11:00 PM | CRITICAL |
| Inventory Snapshot | Daily | ~11:15 PM | HIGH |
| Daily Summary | Daily | ~11:30 PM | HIGH |
| Product Catalog | Weekly | Sunday 11 PM | MEDIUM |

---

## Realistic Daily Workflow

```
11:00 PM - Cova exports daily sales CSV
11:15 PM - Cova exports inventory snapshot  
11:30 PM - Brain daemon detects new files
11:35 PM - Brain imports and processes data
11:45 PM - Brain runs hypothesis tests
12:00 AM - Brain generates next-day predictions
 6:00 AM - Morning report ready for staff
```

---

## Recommended Cova Bridge Implementation

### Option A: Scheduled Manual Export (Current Reality)

Staff exports at close:
1. Run "Daily Sales Report" in Cova
2. Export as CSV to shared folder
3. Brain picks it up overnight

### Option B: Scheduled Automatic Export (Ideal)

```bash
# crontab entry (if Cova supports scheduled exports)
0 23 * * * /path/to/cova-bridge/export_daily.sh
```

### Option C: Cova API (Future - When Available)

Once API access granted:
1. Real-time transaction webhooks
2. Intraday anomaly detection
3. Live inventory tracking

---

## File Locations

Place exports in:
```
/home/macklemoron/Projects/aoc-analytics/data/imports/
├── sales/
│   ├── sales_2025-12-27_0915.csv
│   ├── sales_2025-12-27_0930.csv
│   └── ...
├── inventory/
│   ├── inventory_2025-12-27.csv
│   └── ...
├── catalog/
│   └── catalog_2025-12-22.csv
└── summaries/
    ├── summary_2025-12-27.json
    └── ...
```

---

## Brain Processing

The brain will:

1. **On new sales file:**
   - Parse and validate
   - Insert into SQLite
   - Check for anomalies
   - Update running metrics

2. **On inventory file:**
   - Update stock levels
   - Check for low stock alerts
   - Update reorder recommendations

3. **On daily summary:**
   - Compare to prediction
   - Update accuracy metrics
   - Generate next-day prediction
   - Log to discovery journal

---

## Minimum Viable Export

If resources are limited, prioritize:

1. **Daily transaction dump** (end of day, all transactions)
2. **Weekly inventory** 

This is enough to:
- Run backtests
- Generate predictions
- Track patterns

But NOT enough for:
- Real-time alerts
- Intraday anomaly detection
- Rush hour predictions

---

## Data Quality Requirements

### Must Have:
- Consistent timestamps (same timezone)
- Consistent SKUs (no duplicates, no typos)
- Complete transactions (no missing fields)

### Nice to Have:
- Cost data (enables margin analysis)
- Employee IDs (enables staffing optimization)
- Customer IDs (enables loyalty analysis - if compliant)

---

## Questions for Cova Setup

1. What export formats does Cova support? (CSV, JSON, XML?)
2. Can we schedule automatic exports?
3. Is there an SFTP/S3 push option?
4. What's the data retention period?
5. Can we get historical backfill? (6-12 months ideal)
6. Are webhooks available for real-time?
7. When will API access be ready?

---

## Contact

For brain integration questions:
- Check `/src/aoc_analytics/brain/` for import scripts
- Data validation in `/src/aoc_analytics/core/validators.py`
- Import daemon in `/src/aoc_analytics/brain/daemon.py`
