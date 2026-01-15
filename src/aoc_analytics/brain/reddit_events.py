"""
Reddit Local Events Detection

Scrapes r/vancouver and related subreddits for local events that might
affect retail traffic:
- Concerts and festivals
- Road closures and construction
- Major protests or gatherings
- Community events
- Weather warnings
- Transit disruptions

Uses Reddit's JSON API (no auth required for public subreddits).
"""

import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
from collections import defaultdict
from enum import Enum
import time

try:
    import httpx
    HAS_HTTPX = True
except ImportError:
    HAS_HTTPX = False


class EventType(Enum):
    """Types of events we care about."""
    CONCERT = "concert"
    FESTIVAL = "festival"
    SPORTS = "sports"
    PROTEST = "protest"
    ROAD_CLOSURE = "road_closure"
    TRANSIT = "transit"
    WEATHER = "weather"
    COMMUNITY = "community"
    OTHER = "other"


@dataclass
class LocalEvent:
    """A detected local event."""
    title: str
    event_type: EventType
    source: str  # subreddit
    url: str
    detected_date: str  # When we found it
    event_date: Optional[str]  # When it happens (if we can parse it)
    location: Optional[str]  # Where (if mentioned)
    score: int  # Reddit score (upvotes)
    num_comments: int
    keywords: List[str]  # Which keywords matched
    summary: str  # First ~200 chars of post
    
    def __str__(self) -> str:
        date_str = f" on {self.event_date}" if self.event_date else ""
        loc_str = f" @ {self.location}" if self.location else ""
        return f"[{self.event_type.value}] {self.title}{date_str}{loc_str} (score: {self.score})"


class RedditEventScanner:
    """
    Scans Reddit for local events in Vancouver area.
    """
    
    # Subreddits to monitor
    SUBREDDITS = [
        "vancouver",
        "britishcolumbia", 
        "VancouverEvents",
        "VancouverConcerts",
    ]
    
    # Keywords that indicate relevant events
    KEYWORDS = {
        EventType.CONCERT: [
            "concert", "show", "live music", "tour", "performance",
            "rogers arena", "bc place", "commodore", "orpheum",
            "vogue theatre", "queen elizabeth", "pne",
        ],
        EventType.FESTIVAL: [
            "festival", "4/20", "420", "celebration of light",
            "pride", "fireworks", "parade", "fair", "expo",
            "car free day", "khatsahlano",
        ],
        EventType.SPORTS: [
            "canucks", "whitecaps", "lions", "game day", "playoff",
            "hockey", "soccer", "football", "marathon", "run",
            "triathlon", "seawall",
        ],
        EventType.PROTEST: [
            "protest", "rally", "march", "demonstration", "strike",
            "blockade", "occupation",
        ],
        EventType.ROAD_CLOSURE: [
            "road closure", "closed", "detour", "construction",
            "lane closure", "bridge", "tunnel", "traffic",
        ],
        EventType.TRANSIT: [
            "skytrain", "bus", "translink", "delay", "disruption",
            "shutdown", "canada line", "expo line", "millennium",
        ],
        EventType.WEATHER: [
            "storm", "snow", "flooding", "heat wave", "warning",
            "advisory", "evacuation", "power outage",
        ],
        EventType.COMMUNITY: [
            "farmer's market", "market", "meetup", "event",
            "free", "outdoor", "beach", "park",
        ],
    }
    
    # Location keywords for Vancouver area
    LOCATIONS = [
        "downtown", "gastown", "yaletown", "kitsilano", "kits",
        "commercial drive", "main street", "broadway", "granville",
        "robson", "davie", "west end", "east van", "burnaby",
        "richmond", "north van", "surrey", "coquitlam", "new west",
        "stanley park", "english bay", "science world", "waterfront",
    ]
    
    # Date patterns to extract event dates
    DATE_PATTERNS = [
        r"(\d{1,2})[/-](\d{1,2})",  # MM/DD or DD/MM
        r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\w*\s+(\d{1,2})",
        r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
        r"this\s+(weekend|week)",
        r"tomorrow",
        r"tonight",
        r"today",
    ]
    
    USER_AGENT = "AOC-Analytics/1.0 (Local event detection for retail analytics)"
    
    def __init__(self, db_path: str = None):
        if not HAS_HTTPX:
            raise ImportError("httpx is required for Reddit scanning. Install with: pip install httpx")
        
        if db_path is None:
            possible_paths = [
                Path.cwd() / "aoc_sales.db",
                Path(__file__).resolve().parent.parent.parent.parent.parent / "aoc_sales.db",
            ]
            for p in possible_paths:
                if p.exists() and p.stat().st_size > 0:
                    db_path = str(p)
                    break
        self.db_path = str(db_path) if db_path else None
        
        self.client = httpx.Client(
            timeout=10.0,
            headers={"User-Agent": self.USER_AGENT}
        )
    
    def _fetch_subreddit(self, subreddit: str, limit: int = 50, sort: str = "hot") -> List[Dict]:
        """Fetch posts from a subreddit."""
        
        url = f"https://www.reddit.com/r/{subreddit}/{sort}.json"
        params = {"limit": limit, "raw_json": 1}
        
        try:
            response = self.client.get(url, params=params)
            response.raise_for_status()
            data = response.json()
            
            posts = []
            for child in data.get("data", {}).get("children", []):
                post = child.get("data", {})
                posts.append({
                    "title": post.get("title", ""),
                    "selftext": post.get("selftext", ""),
                    "url": f"https://reddit.com{post.get('permalink', '')}",
                    "score": post.get("score", 0),
                    "num_comments": post.get("num_comments", 0),
                    "created_utc": post.get("created_utc", 0),
                    "subreddit": subreddit,
                })
            
            return posts
            
        except Exception as e:
            print(f"Error fetching r/{subreddit}: {e}")
            return []
    
    def _classify_event(self, title: str, text: str) -> Tuple[Optional[EventType], List[str]]:
        """Classify a post into an event type based on keywords."""
        
        combined = f"{title} {text}".lower()
        
        matched_keywords = []
        best_type = None
        best_count = 0
        
        for event_type, keywords in self.KEYWORDS.items():
            type_matches = []
            for kw in keywords:
                if kw.lower() in combined:
                    type_matches.append(kw)
            
            if len(type_matches) > best_count:
                best_count = len(type_matches)
                best_type = event_type
                matched_keywords = type_matches
        
        if best_count == 0:
            return None, []
        
        return best_type, matched_keywords
    
    def _extract_location(self, title: str, text: str) -> Optional[str]:
        """Try to extract a location from the post."""
        
        combined = f"{title} {text}".lower()
        
        for loc in self.LOCATIONS:
            if loc.lower() in combined:
                return loc.title()
        
        return None
    
    def _extract_date(self, title: str, text: str) -> Optional[str]:
        """Try to extract an event date from the post."""
        
        combined = f"{title} {text}".lower()
        today = date.today()
        
        # Check for relative dates
        if "tomorrow" in combined:
            return (today + timedelta(days=1)).strftime("%Y-%m-%d")
        if "tonight" in combined or "today" in combined:
            return today.strftime("%Y-%m-%d")
        if "this weekend" in combined:
            # Find next Saturday
            days_until_sat = (5 - today.weekday()) % 7
            if days_until_sat == 0:
                days_until_sat = 7
            return (today + timedelta(days=days_until_sat)).strftime("%Y-%m-%d")
        
        # Check for day names
        day_names = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
        for i, day in enumerate(day_names):
            if day in combined:
                # Find next occurrence of this day
                days_until = (i - today.weekday()) % 7
                if days_until == 0:
                    days_until = 7
                return (today + timedelta(days=days_until)).strftime("%Y-%m-%d")
        
        # TODO: More sophisticated date parsing
        return None
    
    def scan_all_subreddits(self, limit_per_sub: int = 50) -> List[LocalEvent]:
        """Scan all monitored subreddits for events."""
        
        all_events = []
        
        for subreddit in self.SUBREDDITS:
            print(f"  Scanning r/{subreddit}...")
            posts = self._fetch_subreddit(subreddit, limit=limit_per_sub)
            
            for post in posts:
                event_type, keywords = self._classify_event(post["title"], post["selftext"])
                
                if event_type is None:
                    continue
                
                # Only include posts with decent engagement
                if post["score"] < 5 and post["num_comments"] < 3:
                    continue
                
                location = self._extract_location(post["title"], post["selftext"])
                event_date = self._extract_date(post["title"], post["selftext"])
                
                # Create summary
                summary = post["selftext"][:200] if post["selftext"] else post["title"]
                
                event = LocalEvent(
                    title=post["title"],
                    event_type=event_type,
                    source=f"r/{subreddit}",
                    url=post["url"],
                    detected_date=datetime.now().strftime("%Y-%m-%d"),
                    event_date=event_date,
                    location=location,
                    score=post["score"],
                    num_comments=post["num_comments"],
                    keywords=keywords,
                    summary=summary,
                )
                
                all_events.append(event)
            
            # Be nice to Reddit
            time.sleep(1)
        
        # Sort by score
        all_events.sort(key=lambda x: x.score, reverse=True)
        
        return all_events
    
    def save_events(self, events: List[LocalEvent]) -> str:
        """Save detected events to JSON."""
        
        output = {
            "scanned_at": str(datetime.now()),
            "subreddits": self.SUBREDDITS,
            "events": [
                {
                    "title": e.title,
                    "type": e.event_type.value,
                    "source": e.source,
                    "url": e.url,
                    "detected_date": e.detected_date,
                    "event_date": e.event_date,
                    "location": e.location,
                    "score": e.score,
                    "num_comments": e.num_comments,
                    "keywords": e.keywords,
                    "summary": e.summary,
                }
                for e in events
            ],
        }
        
        # Save
        brain_dir = Path(__file__).parent / "data"
        brain_dir.mkdir(exist_ok=True)
        
        output_file = brain_dir / "reddit_events.json"
        with open(output_file, "w") as f:
            json.dump(output, f, indent=2)
        
        return str(output_file)
    
    def get_upcoming_events(self, events: List[LocalEvent], days_ahead: int = 7) -> List[LocalEvent]:
        """Filter to events happening in the next N days."""
        
        today = date.today()
        cutoff = today + timedelta(days=days_ahead)
        
        upcoming = []
        for event in events:
            if event.event_date:
                try:
                    event_dt = datetime.strptime(event.event_date, "%Y-%m-%d").date()
                    if today <= event_dt <= cutoff:
                        upcoming.append(event)
                except ValueError:
                    pass
        
        return upcoming
    
    def get_high_impact_events(self, events: List[LocalEvent], min_score: int = 100) -> List[LocalEvent]:
        """Filter to high-engagement events."""
        
        return [e for e in events if e.score >= min_score]


def demo():
    """Demonstrate Reddit event scanning."""
    
    print("=" * 70)
    print("📱 REDDIT LOCAL EVENTS SCANNER")
    print("   Detecting events from Vancouver subreddits")
    print("=" * 70)
    print()
    
    if not HAS_HTTPX:
        print("❌ httpx not installed. Install with: pip install httpx")
        return
    
    scanner = RedditEventScanner()
    
    print("Scanning subreddits...")
    events = scanner.scan_all_subreddits(limit_per_sub=25)
    
    print(f"\n📊 Found {len(events)} relevant events\n")
    
    if not events:
        print("No events detected. This might mean:")
        print("  - Quiet news day")
        print("  - Keywords need tuning")
        print("  - Rate limited by Reddit")
        return
    
    # Group by type
    by_type = defaultdict(list)
    for event in events:
        by_type[event.event_type].append(event)
    
    print("=" * 70)
    print("📋 EVENTS BY TYPE")
    print("=" * 70)
    
    for event_type in EventType:
        type_events = by_type.get(event_type, [])
        if type_events:
            print(f"\n{event_type.value.upper()} ({len(type_events)} events):")
            for e in type_events[:5]:  # Top 5 by score
                loc = f" @ {e.location}" if e.location else ""
                date_str = f" [{e.event_date}]" if e.event_date else ""
                print(f"  • {e.title[:60]}...{loc}{date_str}")
                print(f"    Score: {e.score}, Comments: {e.num_comments}")
    
    # Upcoming events
    upcoming = scanner.get_upcoming_events(events, days_ahead=7)
    if upcoming:
        print("\n" + "=" * 70)
        print("📅 UPCOMING EVENTS (next 7 days)")
        print("=" * 70 + "\n")
        
        for e in upcoming[:10]:
            print(f"  {e.event_date}: [{e.event_type.value}] {e.title[:50]}...")
    
    # High impact
    high_impact = scanner.get_high_impact_events(events, min_score=50)
    if high_impact:
        print("\n" + "=" * 70)
        print("🔥 HIGH IMPACT EVENTS (score >= 50)")
        print("=" * 70 + "\n")
        
        for e in high_impact[:10]:
            print(f"  • [{e.event_type.value}] {e.title[:50]}... (score: {e.score})")
    
    # Save
    output_file = scanner.save_events(events)
    print(f"\n💾 Saved to: {output_file}")


if __name__ == "__main__":
    demo()
