"""
Local event extraction from news and calendar sources.

Scrapes Vancouver Island news sources and extracts structured event data
that can impact cannabis sales (festivals, concerts, sports, etc).
"""

import re
import logging
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Optional
from enum import Enum

logger = logging.getLogger(__name__)


class EventType(str, Enum):
    """Types of events that can impact sales."""
    FESTIVAL = "festival"
    CONCERT = "concert"
    SPORTS = "sports"
    HOLIDAY = "holiday"
    COMMUNITY = "community"
    WEATHER = "weather"
    CONSTRUCTION = "construction"
    COMPETITOR = "competitor"
    OTHER = "other"


class EventImpact(str, Enum):
    """Expected impact direction on sales."""
    POSITIVE = "positive"
    NEGATIVE = "negative"
    NEUTRAL = "neutral"


@dataclass
class LocalEvent:
    """A local event that may impact sales."""
    
    name: str
    event_type: EventType
    start_date: date
    end_date: Optional[date] = None
    location: Optional[str] = None
    expected_attendance: Optional[int] = None
    impact: EventImpact = EventImpact.NEUTRAL
    impact_magnitude: float = 0.0  # -1.0 to 1.0
    confidence: float = 0.5
    source: Optional[str] = None
    raw_text: Optional[str] = None
    
    def __post_init__(self):
        if self.end_date is None:
            self.end_date = self.start_date
    
    def affects_date(self, check_date: date) -> bool:
        """Check if this event affects the given date."""
        return self.start_date <= check_date <= self.end_date
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "event_type": self.event_type.value,
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "location": self.location,
            "expected_attendance": self.expected_attendance,
            "impact": self.impact.value,
            "impact_magnitude": self.impact_magnitude,
            "confidence": self.confidence,
            "source": self.source,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "LocalEvent":
        return cls(
            name=data["name"],
            event_type=EventType(data["event_type"]),
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]) if data.get("end_date") else None,
            location=data.get("location"),
            expected_attendance=data.get("expected_attendance"),
            impact=EventImpact(data.get("impact", "neutral")),
            impact_magnitude=data.get("impact_magnitude", 0.0),
            confidence=data.get("confidence", 0.5),
            source=data.get("source"),
        )


# Event extraction JSON schema for LLM
EVENT_EXTRACTION_SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "name": {"type": "string", "description": "Name of the event"},
                    "event_type": {
                        "type": "string",
                        "enum": ["festival", "concert", "sports", "holiday", "community", "weather", "construction", "competitor", "other"]
                    },
                    "start_date": {"type": "string", "description": "ISO date YYYY-MM-DD"},
                    "end_date": {"type": "string", "description": "ISO date YYYY-MM-DD or null"},
                    "location": {"type": "string", "description": "Location name"},
                    "expected_attendance": {"type": "integer", "description": "Estimated attendance or null"},
                    "impact_on_cannabis_sales": {
                        "type": "string",
                        "enum": ["positive", "negative", "neutral"],
                        "description": "Expected impact on cannabis retail sales"
                    },
                    "impact_magnitude": {
                        "type": "number",
                        "description": "Impact magnitude from -1.0 (very negative) to 1.0 (very positive)"
                    },
                    "reasoning": {"type": "string", "description": "Why this event would impact sales"}
                },
                "required": ["name", "event_type", "start_date", "impact_on_cannabis_sales"]
            }
        }
    },
    "required": ["events"]
}


EVENT_EXTRACTION_SYSTEM = """You are an analyst for a cannabis retail business on Vancouver Island, BC, Canada.
Your job is to extract events from news articles and calendar listings that could impact cannabis sales.

Events that INCREASE cannabis sales:
- Music festivals, concerts (especially outdoor)
- Beach parties, camping events
- Long weekends, holidays
- Sports events (especially if local team wins)
- Good weather forecasts (sunny weekends)
- Community celebrations

Events that DECREASE cannabis sales:
- Severe weather warnings
- Road closures, construction blocking access
- New competitor openings nearby
- Large events that draw people AWAY from the area
- Power outages, emergencies

Be specific with dates. If only a month is mentioned, estimate the likely date.
For impact_magnitude: 0.1-0.3 = minor, 0.4-0.6 = moderate, 0.7-1.0 = major impact."""


class EventExtractor:
    """Extracts structured events from text using LLM."""
    
    def __init__(self, llm_client=None):
        """Initialize with optional LLM client."""
        self._client = llm_client
        self._event_cache: dict[str, list[LocalEvent]] = {}  # date -> events
    
    @property
    def client(self):
        if self._client is None:
            from aoc_analytics.llm.client import get_default_client
            self._client = get_default_client()
        return self._client
    
    def extract_from_text(
        self, 
        text: str, 
        reference_date: Optional[date] = None,
        source: Optional[str] = None,
    ) -> list[LocalEvent]:
        """Extract events from raw text."""
        if reference_date is None:
            reference_date = date.today()
        
        prompt = f"""Extract all events from the following text that could impact cannabis retail sales in the Parksville/Nanaimo area of Vancouver Island.

Reference date for relative dates: {reference_date.isoformat()}

Text:
---
{text}
---

Extract events with their dates, type, location, and expected impact on cannabis sales."""

        try:
            result = self.client.extract_structured(
                prompt=prompt,
                schema=EVENT_EXTRACTION_SCHEMA,
                system=EVENT_EXTRACTION_SYSTEM,
            )
            
            events = []
            for event_data in result.get("events", []):
                try:
                    event = LocalEvent(
                        name=event_data["name"],
                        event_type=EventType(event_data.get("event_type", "other")),
                        start_date=date.fromisoformat(event_data["start_date"]),
                        end_date=date.fromisoformat(event_data["end_date"]) if event_data.get("end_date") else None,
                        location=event_data.get("location"),
                        expected_attendance=event_data.get("expected_attendance"),
                        impact=EventImpact(event_data.get("impact_on_cannabis_sales", "neutral")),
                        impact_magnitude=event_data.get("impact_magnitude", 0.0),
                        confidence=0.7,  # LLM extraction confidence
                        source=source,
                        raw_text=text[:500],  # Store first 500 chars for reference
                    )
                    events.append(event)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Failed to parse event: {e}")
                    continue
            
            return events
            
        except Exception as e:
            logger.error(f"Event extraction failed: {e}")
            return []
    
    def get_events_for_date(self, target_date: date) -> list[LocalEvent]:
        """Get all cached events affecting a specific date."""
        date_key = target_date.isoformat()
        if date_key in self._event_cache:
            return self._event_cache[date_key]
        
        # Check all cached events
        matching = []
        for events in self._event_cache.values():
            for event in events:
                if event.affects_date(target_date):
                    matching.append(event)
        
        return matching
    
    def cache_events(self, events: list[LocalEvent]):
        """Add events to the cache."""
        for event in events:
            date_key = event.start_date.isoformat()
            if date_key not in self._event_cache:
                self._event_cache[date_key] = []
            self._event_cache[date_key].append(event)
    
    def get_impact_for_date(self, target_date: date) -> tuple[float, list[str]]:
        """
        Get the aggregate impact magnitude for a date.
        
        Returns:
            (impact_magnitude, list of event names)
        """
        events = self.get_events_for_date(target_date)
        if not events:
            return 0.0, []
        
        # Combine impacts (weighted by confidence)
        total_impact = 0.0
        total_weight = 0.0
        names = []
        
        for event in events:
            weight = event.confidence
            impact = event.impact_magnitude
            if event.impact == EventImpact.NEGATIVE:
                impact = -abs(impact)
            elif event.impact == EventImpact.POSITIVE:
                impact = abs(impact)
            
            total_impact += impact * weight
            total_weight += weight
            names.append(event.name)
        
        if total_weight > 0:
            return total_impact / total_weight, names
        return 0.0, names


# News source scrapers
class NewsSource:
    """Base class for news scrapers."""
    
    name: str = "unknown"
    url: str = ""
    
    def fetch_articles(self, days_ahead: int = 14) -> list[dict]:
        """Fetch recent articles. Returns list of {title, text, date, url}."""
        raise NotImplementedError


class PQBNewsSource(NewsSource):
    """Parksville Qualicum Beach News scraper."""
    
    name = "PQB News"
    url = "https://www.pqbnews.com"
    
    def fetch_articles(self, days_ahead: int = 14) -> list[dict]:
        """Fetch articles from PQB News."""
        try:
            import httpx
            from bs4 import BeautifulSoup
        except ImportError:
            logger.warning("httpx and beautifulsoup4 required for news scraping")
            return []
        
        articles = []
        try:
            # Fetch events/community pages
            for section in ["/local-news/", "/community/"]:
                response = httpx.get(
                    f"{self.url}{section}",
                    timeout=10.0,
                    headers={"User-Agent": "AOC-Analytics/1.0"}
                )
                if response.status_code == 200:
                    soup = BeautifulSoup(response.text, "html.parser")
                    for article in soup.select("article")[:10]:
                        title_el = article.select_one("h2, h3")
                        link_el = article.select_one("a")
                        if title_el and link_el:
                            articles.append({
                                "title": title_el.get_text(strip=True),
                                "url": link_el.get("href", ""),
                                "text": article.get_text(" ", strip=True)[:1000],
                                "source": self.name,
                            })
        except Exception as e:
            logger.error(f"Failed to fetch {self.name}: {e}")
        
        return articles


class EventbriteSource(NewsSource):
    """Eventbrite Vancouver Island events."""
    
    name = "Eventbrite"
    url = "https://www.eventbrite.ca"
    
    def fetch_articles(self, days_ahead: int = 14) -> list[dict]:
        """Fetch events from Eventbrite."""
        # Would need Eventbrite API key for proper integration
        # This is a placeholder for the structure
        return []


def scrape_local_events(
    extractor: EventExtractor,
    days_ahead: int = 14,
    sources: Optional[list[NewsSource]] = None,
) -> list[LocalEvent]:
    """
    Scrape local news sources and extract events.
    
    Args:
        extractor: EventExtractor instance
        days_ahead: How many days ahead to look
        sources: List of NewsSource instances (defaults to all)
    
    Returns:
        List of extracted LocalEvent objects
    """
    if sources is None:
        sources = [PQBNewsSource()]
    
    all_events = []
    
    for source in sources:
        logger.info(f"Scraping {source.name}...")
        articles = source.fetch_articles(days_ahead)
        
        for article in articles:
            events = extractor.extract_from_text(
                text=f"{article.get('title', '')}\n\n{article.get('text', '')}",
                source=f"{source.name}: {article.get('url', '')}",
            )
            all_events.extend(events)
    
    # Cache all events
    extractor.cache_events(all_events)
    
    return all_events
