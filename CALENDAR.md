Absolutely. Below is a Copilot-ready blueprint markdown you can paste directly into your repo or into GitHub Copilot Chat.
It’s written as an implementation spec, not prose, so Copilot can scaffold code from it cleanly.

# Blueprint: Regional Calendar Aggregation Service
## BC / Federal Statutory Holidays + Vancouver Events (25km)

### Purpose
Provide a single normalized calendar feed combining:
- Canadian **federal + BC statutory holidays**
- **Local events within 25km of Vancouver**
Returned as normalized objects for forecasting, staffing, and analytics.

This service is **read-only**, deterministic, and cache-friendly.

---

## Inputs
- `center_lat`: float (default: 49.2827)
- `center_lon`: float (default: -123.1207)
- `radius_km`: int (default: 25)
- `start_date`: ISO date (YYYY-MM-DD)
- `end_date`: ISO date (YYYY-MM-DD)
- `region`: string (default: "CA-BC")
- `timezone`: string (default: "America/Vancouver")

---

## External Data Sources

### Holidays
**Provider**: Nager.Date Public Holiday API  
**Endpoint pattern**:


GET https://date.nager.at/api/v3/publicholidays/{year}/CA


**Filtering rules**:
- Include holidays where:
  - `global == true` (federal)
  - OR `counties` contains `"CA-BC"`

---

### Events
**Primary Provider**: PredictHQ Events API  
**Radius query format**:


within={radius_km}km@{center_lat},{center_lon}


**Required filters**:
- Active dates overlap `[start_date, end_date]`
- Location within radius
- Category not null

---

## Normalized Event Object (Canonical Schema)

```json
{
  "id": "string",
  "type": "holiday | event",
  "title": "string",
  "start": "ISO-8601 datetime",
  "end": "ISO-8601 datetime",
  "all_day": boolean,
  "timezone": "America/Vancouver",
  "region": {
    "country": "CA",
    "subdivisions": ["CA-BC"],
    "is_federal": boolean
  },
  "location": {
    "name": "string | null",
    "lat": number | null,
    "lon": number | null,
    "radius_km": number | null
  },
  "categories": ["string"],
  "source": {
    "provider": "nager | predicthq",
    "provider_id": "string",
    "url": "string | null"
  },
  "metadata": {
    "confidence": "high | medium | low",
    "attendance": number | null,
    "rank": number | null
  }
}

Normalization Rules
Holidays → Normalized

type = "holiday"

all_day = true

start = holiday.date + "T00:00:00"

end = start + 1 day

is_federal = holiday.global

location = null

categories = ["public_holiday"]

provider = "nager"

Events → Normalized

type = "event"

all_day = false unless explicitly marked

start/end from provider

location.lat/lon required

categories mapped from provider taxonomy

confidence derived from event rank / category

provider = "predicthq"

Deduplication Strategy

Primary key:

hash(title + start + provider)


Prefer higher-confidence provider if collision occurs

Output

JSON array of normalized objects

Sorted by start ASC

Stable ordering for identical timestamps

Caching Strategy

Cache per:

region + start_date + end_date + radius_km


TTL:

Holidays: 30 days

Events: 6–12 hours

Example Internal Endpoint
GET /calendar/external
  ?start=2026-01-01
  &end=2026-12-31
  &center=49.2827,-123.1207
  &radius_km=25

Intended Use

Demand forecasting regressors

Staffing forecasts

Sunday volatility explainability

Holiday uplift modeling

Non-Goals

No user calendars

No write/update operations

No personalization

No ticketing or RSVP logic


---

If you want, next steps I can:
- Convert this into **FastAPI / Flask scaffolding**
- Add **feature-flagged providers** (Ticketmaster fallback)
- Extend schema for **holiday uplift coefficients** (forecast-ready)

Just say the word.
