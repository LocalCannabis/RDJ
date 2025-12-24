"""
Google Trends Mood Client

Fetches search trend data from Google Trends to derive mood buckets
(stress, cozy, party, money pressure, cannabis interest).

Uses pytrends (unofficial API) with fallback handling.

Note: pytrends is rate-limited and may be flaky. Design for graceful degradation.
"""

from __future__ import annotations

import logging
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# Mood bucket definitions
MOOD_BUCKETS = {
    "stress": {
        "terms": ["stress", "anxiety", "can't sleep", "panic attack", "insomnia"],
        "description": "Stress and anxiety indicators",
    },
    "cozy": {
        "terms": ["cozy", "comfort food", "movie night", "relax", "stay home"],
        "description": "At-home and comfort indicators",
    },
    "party": {
        "terms": ["bar near me", "club", "party", "events tonight", "concert"],
        "description": "Going out and social indicators",
    },
    "money_pressure": {
        "terms": ["rent", "payday loan", "unemployment", "debt", "budget"],
        "description": "Financial stress indicators",
    },
    "cannabis_interest": {
        "terms": ["dispensary near me", "edibles", "vape pen", "weed", "cannabis"],
        "description": "Cannabis industry interest",
    },
}

# Default geo settings
DEFAULT_GEO = "CA"  # Canada
BC_GEO = "CA-BC"    # British Columbia


@dataclass
class TermInterest:
    """Interest value for a single term on a date."""
    date: str
    geo: str
    term: str
    value: int  # 0-100 Google scale
    
    
@dataclass
class BucketScores:
    """Daily bucket scores for a geo."""
    date: str
    geo: str
    stress_score: float
    cozy_score: float
    party_score: float
    money_pressure_score: float
    cannabis_interest_score: float
    term_coverage: float  # % of terms with data


class GoogleTrendsMoodClient:
    """
    Client for fetching mood bucket data from Google Trends.
    
    Uses pytrends for data access with built-in rate limiting
    and graceful fallback when data is unavailable.
    """
    
    def __init__(
        self,
        geo: str = DEFAULT_GEO,
        rate_limit_delay: float = 2.0,
    ):
        """
        Initialize Google Trends client.
        
        Args:
            geo: Geographic region (CA, CA-BC, etc.)
            rate_limit_delay: Seconds to wait between API calls
        """
        self.geo = geo
        self.rate_limit_delay = rate_limit_delay
        self._pytrends = None
        self._last_request_time = 0
    
    def _get_pytrends(self):
        """Lazy-load pytrends client."""
        if self._pytrends is None:
            try:
                from pytrends.request import TrendReq
                self._pytrends = TrendReq(hl='en-US', tz=480)  # Pacific time
                logger.debug("pytrends client initialized")
            except ImportError:
                raise ImportError(
                    "pytrends not installed. Install with: pip install pytrends"
                )
        return self._pytrends
    
    def _rate_limit(self):
        """Apply rate limiting between requests."""
        elapsed = time.time() - self._last_request_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)
        self._last_request_time = time.time()
    
    def fetch_term_interest(
        self,
        term: str,
        timeframe: str = "today 3-m",
        geo: Optional[str] = None,
    ) -> List[TermInterest]:
        """
        Fetch interest over time for a single term.
        
        Args:
            term: Search term
            timeframe: pytrends timeframe string
            geo: Geographic region (defaults to self.geo)
            
        Returns:
            List of TermInterest values
        """
        geo = geo or self.geo
        pytrends = self._get_pytrends()
        
        self._rate_limit()
        
        try:
            pytrends.build_payload(
                kw_list=[term],
                cat=0,
                timeframe=timeframe,
                geo=geo,
            )
            
            df = pytrends.interest_over_time()
            
            if df.empty:
                logger.warning(f"No data for term '{term}' in {geo}")
                return []
            
            results = []
            for idx, row in df.iterrows():
                if term in row:
                    results.append(TermInterest(
                        date=idx.strftime("%Y-%m-%d"),
                        geo=geo,
                        term=term,
                        value=int(row[term]),
                    ))
            
            logger.debug(f"Fetched {len(results)} data points for '{term}'")
            return results
            
        except Exception as e:
            logger.error(f"Error fetching term '{term}': {e}")
            return []
    
    def fetch_bucket_terms(
        self,
        bucket_name: str,
        timeframe: str = "today 3-m",
        geo: Optional[str] = None,
    ) -> Dict[str, List[TermInterest]]:
        """
        Fetch interest for all terms in a mood bucket.
        
        Args:
            bucket_name: Name of the mood bucket
            timeframe: pytrends timeframe string
            geo: Geographic region
            
        Returns:
            Dict of term -> list of TermInterest
        """
        if bucket_name not in MOOD_BUCKETS:
            raise ValueError(f"Unknown bucket: {bucket_name}")
        
        geo = geo or self.geo
        bucket = MOOD_BUCKETS[bucket_name]
        
        results = {}
        for term in bucket["terms"]:
            interests = self.fetch_term_interest(term, timeframe, geo)
            results[term] = interests
        
        return results
    
    def compute_bucket_scores(
        self,
        target_date: str,
        geo: Optional[str] = None,
        lookback_days: int = 90,
    ) -> BucketScores:
        """
        Compute mood bucket scores for a specific date.
        
        Uses rolling z-scores to normalize term values.
        
        Args:
            target_date: Date to compute scores for (YYYY-MM-DD)
            geo: Geographic region
            lookback_days: Days to use for baseline calculation
            
        Returns:
            BucketScores for the date
        """
        geo = geo or self.geo
        
        # Calculate timeframe
        end_date = datetime.strptime(target_date, "%Y-%m-%d")
        start_date = end_date - timedelta(days=lookback_days)
        timeframe = f"{start_date.strftime('%Y-%m-%d')} {end_date.strftime('%Y-%m-%d')}"
        
        # Fetch all bucket data
        all_bucket_data: Dict[str, Dict[str, List[TermInterest]]] = {}
        for bucket_name in MOOD_BUCKETS:
            all_bucket_data[bucket_name] = self.fetch_bucket_terms(
                bucket_name, timeframe, geo
            )
        
        # Compute z-scored bucket values for target date
        def compute_z_score_for_date(
            interests: List[TermInterest],
            target: str,
        ) -> Optional[float]:
            """Compute z-score for target date using all values as baseline."""
            if not interests:
                return None
            
            values = [i.value for i in interests]
            target_value = None
            
            for i in interests:
                if i.date == target:
                    target_value = i.value
                    break
            
            if target_value is None:
                return None
            
            if len(values) < 2:
                return 0.0
            
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            std = variance ** 0.5 if variance > 0 else 1.0
            
            return (target_value - mean) / std if std > 0 else 0.0
        
        # Compute bucket scores
        bucket_scores = {}
        total_terms = 0
        terms_with_data = 0
        
        for bucket_name, term_data in all_bucket_data.items():
            z_scores = []
            
            for term, interests in term_data.items():
                total_terms += 1
                z = compute_z_score_for_date(interests, target_date)
                if z is not None:
                    z_scores.append(z)
                    terms_with_data += 1
            
            # Bucket score = mean of term z-scores
            if z_scores:
                bucket_scores[bucket_name] = sum(z_scores) / len(z_scores)
            else:
                bucket_scores[bucket_name] = 0.0
        
        term_coverage = terms_with_data / total_terms if total_terms > 0 else 0.0
        
        return BucketScores(
            date=target_date,
            geo=geo,
            stress_score=bucket_scores.get("stress", 0.0),
            cozy_score=bucket_scores.get("cozy", 0.0),
            party_score=bucket_scores.get("party", 0.0),
            money_pressure_score=bucket_scores.get("money_pressure", 0.0),
            cannabis_interest_score=bucket_scores.get("cannabis_interest", 0.0),
            term_coverage=term_coverage,
        )
    
    def save_raw_terms_to_db(
        self,
        db_path: str | Path,
        timeframe: str = "today 3-m",
        geo: Optional[str] = None,
    ) -> int:
        """
        Fetch and save all raw term data to database.
        
        Args:
            db_path: Path to SQLite database
            timeframe: pytrends timeframe string
            geo: Geographic region
            
        Returns:
            Number of rows saved
        """
        geo = geo or self.geo
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        rows_saved = 0
        
        for bucket_name, bucket in MOOD_BUCKETS.items():
            for term in bucket["terms"]:
                interests = self.fetch_term_interest(term, timeframe, geo)
                
                for interest in interests:
                    cursor.execute("""
                        INSERT OR REPLACE INTO mood_google_terms_raw
                        (date, geo, term, value_0_100, source)
                        VALUES (?, ?, ?, ?, 'pytrends')
                    """, (
                        interest.date,
                        interest.geo,
                        interest.term,
                        interest.value,
                    ))
                    rows_saved += 1
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved {rows_saved} raw term values to database")
        return rows_saved
    
    def save_bucket_scores_to_db(
        self,
        db_path: str | Path,
        target_date: str,
        geo: Optional[str] = None,
    ) -> BucketScores:
        """
        Compute and save bucket scores to database.
        
        Args:
            db_path: Path to SQLite database
            target_date: Date to compute scores for
            geo: Geographic region
            
        Returns:
            BucketScores
        """
        scores = self.compute_bucket_scores(target_date, geo)
        
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO mood_google_buckets_daily
            (date, geo, stress_score, cozy_score, party_score,
             money_pressure_score, cannabis_interest_score, term_coverage)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            scores.date,
            scores.geo,
            scores.stress_score,
            scores.cozy_score,
            scores.party_score,
            scores.money_pressure_score,
            scores.cannabis_interest_score,
            scores.term_coverage,
        ))
        
        conn.commit()
        conn.close()
        
        logger.info(f"Saved bucket scores for {scores.date} ({scores.geo})")
        return scores


def check_pytrends_available() -> bool:
    """Check if pytrends is installed and working."""
    try:
        from pytrends.request import TrendReq
        return True
    except ImportError:
        return False


if __name__ == "__main__":
    # Quick test
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    if not check_pytrends_available():
        print("ERROR: pytrends not installed.")
        print("Install with: pip install pytrends")
        sys.exit(1)
    
    client = GoogleTrendsMoodClient(geo="CA")
    
    # Test single term
    print("\nFetching 'stress' interest...")
    interests = client.fetch_term_interest("stress", "today 1-m")
    
    if interests:
        print(f"  Got {len(interests)} data points")
        print(f"  Latest: {interests[-1].date} = {interests[-1].value}")
    
    # Test bucket scores
    target_date = date.today().isoformat()
    print(f"\nComputing bucket scores for {target_date}...")
    
    try:
        scores = client.compute_bucket_scores(target_date)
        print(f"\nBucket Scores:")
        print(f"  Stress: {scores.stress_score:.3f}")
        print(f"  Cozy: {scores.cozy_score:.3f}")
        print(f"  Party: {scores.party_score:.3f}")
        print(f"  Money Pressure: {scores.money_pressure_score:.3f}")
        print(f"  Cannabis Interest: {scores.cannabis_interest_score:.3f}")
        print(f"  Term Coverage: {scores.term_coverage:.1%}")
    except Exception as e:
        print(f"Error: {e}")
