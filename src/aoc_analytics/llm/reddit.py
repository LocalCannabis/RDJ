"""
Reddit API client for extracting local event/discussion signals.

Uses Reddit's official OAuth API - no scraping required.
Free tier: 100 queries/minute.

Setup:
1. Go to https://www.reddit.com/prefs/apps
2. Create a "script" type application
3. Note your client_id (under app name) and client_secret
4. Set environment variables:
   - REDDIT_CLIENT_ID
   - REDDIT_CLIENT_SECRET
   - REDDIT_USER_AGENT (optional, defaults to AOC-Analytics)
"""

import os
import time
import logging
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from typing import Optional
import json

logger = logging.getLogger(__name__)


@dataclass
class RedditPost:
    """A Reddit post with relevant metadata."""
    
    id: str
    subreddit: str
    title: str
    selftext: str
    score: int
    num_comments: int
    created_utc: datetime
    url: str
    permalink: str
    
    # Extracted signals
    is_event: bool = False
    event_date: Optional[date] = None
    sentiment: float = 0.0  # -1 to 1
    relevance_score: float = 0.0
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "subreddit": self.subreddit,
            "title": self.title,
            "selftext": self.selftext[:500] if self.selftext else "",
            "score": self.score,
            "num_comments": self.num_comments,
            "created_utc": self.created_utc.isoformat(),
            "url": self.url,
            "permalink": self.permalink,
            "is_event": self.is_event,
            "event_date": self.event_date.isoformat() if self.event_date else None,
            "sentiment": self.sentiment,
            "relevance_score": self.relevance_score,
        }


@dataclass
class RedditConfig:
    """Configuration for Reddit API client."""
    
    client_id: Optional[str] = None
    client_secret: Optional[str] = None
    user_agent: str = "AOC-Analytics:v1.0 (by /u/LocalCannabis)"
    
    # Target subreddits for Vancouver Island cannabis retail signals
    subreddits: list[str] = field(default_factory=lambda: [
        "VancouverIsland",
        "nanaimo", 
        "Parksville",
        "VictoriaBC",
        "britishcolumbia",
        "canadients",  # Canadian cannabis community
        "TheOCS",      # Ontario but has BC discussion
    ])
    
    # Keywords that might indicate events or sales-impacting news
    event_keywords: list[str] = field(default_factory=lambda: [
        "festival", "concert", "event", "show", "music",
        "beach", "summer", "party", "celebration",
        "market", "fair", "parade", "fireworks",
        "traffic", "closed", "construction", "accident",
        "weather", "storm", "heat", "rain",
        "cannabis", "dispensary", "weed", "420",
        "new store", "opening", "closing",
    ])
    
    def __post_init__(self):
        # Auto-load from environment
        if self.client_id is None:
            self.client_id = os.getenv("REDDIT_CLIENT_ID")
        if self.client_secret is None:
            self.client_secret = os.getenv("REDDIT_CLIENT_SECRET")
        if os.getenv("REDDIT_USER_AGENT"):
            self.user_agent = os.getenv("REDDIT_USER_AGENT")


class RedditClient:
    """
    Reddit API client using official OAuth.
    
    Rate limited to 100 requests/minute on free tier.
    """
    
    def __init__(self, config: Optional[RedditConfig] = None):
        self.config = config or RedditConfig()
        self._access_token: Optional[str] = None
        self._token_expires: float = 0
        self._request_count = 0
        self._request_window_start = time.time()
        
        # Check for httpx
        try:
            import httpx
            self._httpx = httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")
    
    def _get_access_token(self) -> str:
        """Get or refresh OAuth access token."""
        if self._access_token and time.time() < self._token_expires:
            return self._access_token
        
        if not self.config.client_id or not self.config.client_secret:
            raise ValueError(
                "Reddit API credentials required. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET"
            )
        
        # Request token using client credentials flow
        auth = (self.config.client_id, self.config.client_secret)
        
        response = self._httpx.post(
            "https://www.reddit.com/api/v1/access_token",
            auth=auth,
            data={"grant_type": "client_credentials"},
            headers={"User-Agent": self.config.user_agent},
            timeout=10.0,
        )
        response.raise_for_status()
        
        data = response.json()
        self._access_token = data["access_token"]
        self._token_expires = time.time() + data.get("expires_in", 3600) - 60
        
        logger.info("Reddit OAuth token acquired")
        return self._access_token
    
    def _rate_limit(self):
        """Enforce rate limiting (100 req/min)."""
        now = time.time()
        
        # Reset window if needed
        if now - self._request_window_start > 60:
            self._request_count = 0
            self._request_window_start = now
        
        # Wait if at limit
        if self._request_count >= 95:  # Leave buffer
            sleep_time = 60 - (now - self._request_window_start)
            if sleep_time > 0:
                logger.info(f"Rate limit reached, sleeping {sleep_time:.1f}s")
                time.sleep(sleep_time)
                self._request_count = 0
                self._request_window_start = time.time()
        
        self._request_count += 1
    
    def _api_request(self, endpoint: str, params: Optional[dict] = None) -> dict:
        """Make an authenticated API request."""
        self._rate_limit()
        token = self._get_access_token()
        
        url = f"https://oauth.reddit.com{endpoint}"
        headers = {
            "Authorization": f"Bearer {token}",
            "User-Agent": self.config.user_agent,
        }
        
        response = self._httpx.get(
            url,
            headers=headers,
            params=params,
            timeout=10.0,
        )
        response.raise_for_status()
        return response.json()
    
    def get_subreddit_posts(
        self,
        subreddit: str,
        sort: str = "new",
        limit: int = 25,
        time_filter: str = "week",
    ) -> list[RedditPost]:
        """
        Get posts from a subreddit.
        
        Args:
            subreddit: Subreddit name (without r/)
            sort: One of 'hot', 'new', 'top', 'rising'
            limit: Max posts to return (max 100)
            time_filter: For 'top', one of 'hour', 'day', 'week', 'month', 'year', 'all'
        """
        endpoint = f"/r/{subreddit}/{sort}"
        params = {"limit": min(limit, 100)}
        if sort == "top":
            params["t"] = time_filter
        
        try:
            data = self._api_request(endpoint, params)
        except Exception as e:
            logger.warning(f"Failed to fetch r/{subreddit}: {e}")
            return []
        
        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            
            created = datetime.fromtimestamp(post_data.get("created_utc", 0))
            
            post = RedditPost(
                id=post_data.get("id", ""),
                subreddit=post_data.get("subreddit", subreddit),
                title=post_data.get("title", ""),
                selftext=post_data.get("selftext", ""),
                score=post_data.get("score", 0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=created,
                url=post_data.get("url", ""),
                permalink=f"https://reddit.com{post_data.get('permalink', '')}",
            )
            posts.append(post)
        
        return posts
    
    def search_subreddit(
        self,
        subreddit: str,
        query: str,
        sort: str = "relevance",
        limit: int = 25,
        time_filter: str = "month",
    ) -> list[RedditPost]:
        """
        Search within a subreddit.
        
        Args:
            subreddit: Subreddit name
            query: Search query
            sort: One of 'relevance', 'hot', 'top', 'new', 'comments'
            limit: Max results
            time_filter: Time filter for results
        """
        endpoint = f"/r/{subreddit}/search"
        params = {
            "q": query,
            "restrict_sr": "true",
            "sort": sort,
            "limit": min(limit, 100),
            "t": time_filter,
        }
        
        try:
            data = self._api_request(endpoint, params)
        except Exception as e:
            logger.warning(f"Search failed for r/{subreddit}: {e}")
            return []
        
        posts = []
        for child in data.get("data", {}).get("children", []):
            post_data = child.get("data", {})
            
            created = datetime.fromtimestamp(post_data.get("created_utc", 0))
            
            post = RedditPost(
                id=post_data.get("id", ""),
                subreddit=post_data.get("subreddit", subreddit),
                title=post_data.get("title", ""),
                selftext=post_data.get("selftext", ""),
                score=post_data.get("score", 0),
                num_comments=post_data.get("num_comments", 0),
                created_utc=created,
                url=post_data.get("url", ""),
                permalink=f"https://reddit.com{post_data.get('permalink', '')}",
            )
            posts.append(post)
        
        return posts
    
    def search_all(
        self,
        query: str,
        subreddits: Optional[list[str]] = None,
        limit: int = 10,
    ) -> list[RedditPost]:
        """
        Search across multiple subreddits.
        
        Args:
            query: Search query
            subreddits: List of subreddits to search (defaults to config)
            limit: Max results per subreddit
        """
        if subreddits is None:
            subreddits = self.config.subreddits
        
        all_posts = []
        for subreddit in subreddits:
            posts = self.search_subreddit(subreddit, query, limit=limit)
            all_posts.extend(posts)
            
        # Sort by score
        all_posts.sort(key=lambda p: p.score, reverse=True)
        return all_posts
    
    def get_recent_events(
        self,
        days_back: int = 7,
        min_score: int = 5,
    ) -> list[RedditPost]:
        """
        Find recent posts that might be events.
        
        Searches for event-related keywords across target subreddits.
        """
        all_posts = []
        cutoff = datetime.now() - timedelta(days=days_back)
        
        # Search for event keywords
        for keyword in self.config.event_keywords[:5]:  # Limit API calls
            posts = self.search_all(keyword, limit=10)
            for post in posts:
                if post.created_utc >= cutoff and post.score >= min_score:
                    post.is_event = True
                    all_posts.append(post)
        
        # Deduplicate by ID
        seen = set()
        unique = []
        for post in all_posts:
            if post.id not in seen:
                seen.add(post.id)
                unique.append(post)
        
        return unique


class RedditSignalExtractor:
    """
    Extracts actionable signals from Reddit posts using LLM.
    
    Combines Reddit API data with LLM analysis for:
    - Event detection and date extraction
    - Sentiment analysis
    - Sales impact prediction
    """
    
    def __init__(
        self,
        reddit_client: Optional[RedditClient] = None,
        llm_client=None,
    ):
        self.reddit = reddit_client or RedditClient()
        self._llm_client = llm_client
    
    @property
    def llm_client(self):
        if self._llm_client is None:
            from aoc_analytics.llm.client import get_default_client
            self._llm_client = get_default_client()
        return self._llm_client
    
    def analyze_post(self, post: RedditPost) -> RedditPost:
        """
        Analyze a post for event/sales signals using LLM.
        
        Returns the post with updated signal fields.
        """
        prompt = f"""Analyze this Reddit post for cannabis retail impact:

Subreddit: r/{post.subreddit}
Title: {post.title}
Content: {post.selftext[:500] if post.selftext else "(no content)"}
Score: {post.score}, Comments: {post.num_comments}

Questions:
1. Is this about an event (festival, concert, gathering, etc)?
2. If yes, what date(s) does it occur? (format: YYYY-MM-DD or null)
3. Sentiment: positive, negative, or neutral about the local area?
4. Relevance to cannabis retail (0-1 scale)?
5. Expected sales impact: positive, negative, or neutral?

Respond with JSON only."""

        schema = {
            "type": "object",
            "properties": {
                "is_event": {"type": "boolean"},
                "event_date": {"type": "string", "nullable": True},
                "sentiment": {"type": "string", "enum": ["positive", "negative", "neutral"]},
                "relevance": {"type": "number"},
                "sales_impact": {"type": "string", "enum": ["positive", "negative", "neutral"]},
            }
        }
        
        try:
            result = self.llm_client.extract_structured(prompt, schema)
            
            post.is_event = result.get("is_event", False)
            if result.get("event_date"):
                try:
                    post.event_date = date.fromisoformat(result["event_date"])
                except ValueError:
                    pass
            
            sentiment_map = {"positive": 0.5, "neutral": 0.0, "negative": -0.5}
            post.sentiment = sentiment_map.get(result.get("sentiment", "neutral"), 0.0)
            post.relevance_score = result.get("relevance", 0.0)
            
        except Exception as e:
            logger.warning(f"LLM analysis failed for post {post.id}: {e}")
        
        return post
    
    def get_signals_for_date(
        self,
        target_date: date,
        days_context: int = 7,
    ) -> dict:
        """
        Get Reddit-derived signals for a specific date.
        
        Args:
            target_date: The date to get signals for
            days_context: Days of Reddit history to consider
        
        Returns:
            Dict with aggregated signals
        """
        # Get recent posts
        posts = self.reddit.get_recent_events(days_back=days_context)
        
        # Analyze each post
        analyzed = []
        for post in posts[:20]:  # Limit LLM calls
            analyzed.append(self.analyze_post(post))
        
        # Filter to events affecting target date
        relevant_events = [
            p for p in analyzed 
            if p.is_event and p.event_date == target_date
        ]
        
        # Aggregate signals
        avg_sentiment = 0.0
        if analyzed:
            avg_sentiment = sum(p.sentiment for p in analyzed) / len(analyzed)
        
        return {
            "date": target_date.isoformat(),
            "reddit_sentiment": avg_sentiment,
            "event_count": len(relevant_events),
            "events": [p.title for p in relevant_events],
            "total_posts_analyzed": len(analyzed),
            "high_engagement_posts": [
                p.to_dict() for p in analyzed 
                if p.score > 50 or p.num_comments > 20
            ],
        }


# Convenience function
def get_reddit_client() -> RedditClient:
    """Get a configured Reddit client."""
    return RedditClient()


def check_reddit_credentials() -> bool:
    """Check if Reddit API credentials are configured."""
    return bool(os.getenv("REDDIT_CLIENT_ID") and os.getenv("REDDIT_CLIENT_SECRET"))
