"""
AOC Analytics Python Client

Provides a clean interface for consuming AOC analytics from other services
(e.g., LocalBot/JFK inventory management).

Copyright (c) 2024-2025 Tim Kaye / Local Cannabis Co.
All Rights Reserved. Proprietary and Confidential.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Optional

import httpx


@dataclass
class AOCConfig:
    """Configuration for AOC client."""
    
    base_url: str = field(
        default_factory=lambda: os.getenv("AOC_API_URL", "http://localhost:8001/api/v1")
    )
    api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("AOC_API_KEY")
    )
    timeout: float = 30.0
    retry_attempts: int = 3


class AOCError(Exception):
    """Base exception for AOC client errors."""
    
    def __init__(self, message: str, code: str = "UNKNOWN", details: dict | None = None):
        super().__init__(message)
        self.code = code
        self.details = details or {}


class AOCConnectionError(AOCError):
    """Raised when unable to connect to AOC service."""
    pass


class AOCNotFoundError(AOCError):
    """Raised when requested resource not found."""
    pass


class AOCClient:
    """
    Client for AOC Analytics API.
    
    Provides methods to retrieve behavioral signals, demand forecasts,
    hero rankings, and anomaly information from the AOC service.
    
    Example:
        >>> aoc = AOCClient()
        >>> signals = aoc.get_signals("central")
        >>> print(f"At-home index: {signals['at_home']}")
        
        >>> heroes = aoc.get_heroes("central", lens="margin_mix")
        >>> for hero in heroes[:5]:
        ...     print(f"{hero['product_name']}: {hero['hero_score']}")
    """
    
    def __init__(self, config: AOCConfig | None = None):
        """
        Initialize AOC client.
        
        Args:
            config: Optional configuration. Uses environment variables if not provided.
        """
        self.config = config or AOCConfig()
        self._client: httpx.Client | None = None
    
    @property
    def client(self) -> httpx.Client:
        """Lazy-initialize HTTP client."""
        if self._client is None:
            headers = {}
            if self.config.api_key:
                headers["X-API-Key"] = self.config.api_key
            
            self._client = httpx.Client(
                base_url=self.config.base_url,
                timeout=self.config.timeout,
                headers=headers,
            )
        return self._client
    
    def close(self) -> None:
        """Close the HTTP client."""
        if self._client:
            self._client.close()
            self._client = None
    
    def __enter__(self) -> "AOCClient":
        return self
    
    def __exit__(self, *args: Any) -> None:
        self.close()
    
    def _handle_response(self, response: httpx.Response) -> dict:
        """Handle API response and raise appropriate errors."""
        try:
            response.raise_for_status()
            return response.json()
        except httpx.HTTPStatusError as e:
            try:
                error_data = response.json().get("error", {})
            except Exception:
                error_data = {}
            
            code = error_data.get("code", "HTTP_ERROR")
            message = error_data.get("message", str(e))
            details = error_data.get("details", {})
            
            if response.status_code == 404:
                raise AOCNotFoundError(message, code, details) from e
            raise AOCError(message, code, details) from e
        except httpx.ConnectError as e:
            raise AOCConnectionError(
                f"Unable to connect to AOC service at {self.config.base_url}",
                "CONNECTION_ERROR"
            ) from e
    
    # =========================================================================
    # Health & Status
    # =========================================================================
    
    def health(self) -> dict:
        """
        Check AOC service health.
        
        Returns:
            Health status including database connectivity and signal coverage.
        """
        response = self.client.get("/health")
        return self._handle_response(response)
    
    def is_available(self) -> bool:
        """
        Quick check if AOC service is available.
        
        Returns:
            True if service is healthy, False otherwise.
        """
        try:
            health = self.health()
            return health.get("status") == "healthy"
        except Exception:
            return False
    
    # =========================================================================
    # Behavioral Signals
    # =========================================================================
    
    def get_signals(
        self,
        store: str,
        date: str | date | None = None,
        hour: int | None = None,
    ) -> dict:
        """
        Get current behavioral signals (weather-normalized context).
        
        These signals represent the A PRIORI context - weather effects have
        already been transformed into behavioral propensities.
        
        Args:
            store: Store identifier (e.g., "central", "west")
            date: Optional date (defaults to today)
            hour: Optional hour of day 0-23 (defaults to current)
        
        Returns:
            Dictionary containing:
            - signals: at_home, out_and_about, holiday, payday, local_vibe, etc.
            - mood_components: music, anxiety, party_energy, coziness, etc.
            - weather_context: Raw weather data that generated these signals
        
        Example:
            >>> signals = aoc.get_signals("central")
            >>> if signals["signals"]["at_home"] > 0.6:
            ...     print("Cozy day - feature indoor products")
        """
        params: dict[str, Any] = {"store": store}
        if date:
            params["date"] = date.isoformat() if isinstance(date, date) else date
        if hour is not None:
            params["hour"] = hour
        
        response = self.client.get("/signals/current", params=params)
        return self._handle_response(response)
    
    def get_signals_range(
        self,
        store: str,
        start_date: str | date,
        end_date: str | date,
        granularity: str = "daily",
    ) -> dict:
        """
        Get signals for a date range.
        
        Args:
            store: Store identifier
            start_date: Start of range
            end_date: End of range
            granularity: "daily" or "hourly"
        
        Returns:
            Dictionary with list of signal records for the period.
        """
        params = {
            "store": store,
            "start_date": start_date.isoformat() if isinstance(start_date, date) else start_date,
            "end_date": end_date.isoformat() if isinstance(end_date, date) else end_date,
            "granularity": granularity,
        }
        response = self.client.get("/signals/range", params=params)
        return self._handle_response(response)
    
    # =========================================================================
    # Demand Forecasting
    # =========================================================================
    
    def forecast_demand(
        self,
        sku: str,
        store: str,
        days: int = 7,
        context: dict | None = None,
    ) -> dict:
        """
        Get demand forecast for a specific SKU.
        
        Forecasts are based on similar historical contexts (weather, day-of-week,
        payday proximity, etc.) - NOT simple time series extrapolation.
        
        Args:
            sku: Product SKU
            store: Store identifier
            days: Number of days to forecast (1-30)
            context: Optional context override (weather, date)
        
        Returns:
            Dictionary containing:
            - daily_forecast: List of {date, units, confidence}
            - total_forecast: Sum of units over period
            - context_applied: Which signals affected the forecast
            - similar_periods_used: Historical periods used for prediction
        
        Example:
            >>> forecast = aoc.forecast_demand("ABC123", "central", days=7)
            >>> print(f"Next 7 days: {forecast['total_forecast']} units")
            >>> print(f"Confidence: {forecast['daily_forecast'][0]['confidence']:.0%}")
        """
        payload: dict[str, Any] = {
            "sku": sku,
            "store_id": store,
            "days_ahead": days,
        }
        if context:
            payload["context"] = context
        
        response = self.client.post("/forecast/demand", json=payload)
        return self._handle_response(response)
    
    def forecast_bulk(
        self,
        skus: list[str],
        store: str,
        days: int = 7,
    ) -> dict:
        """
        Batch forecast for multiple SKUs.
        
        More efficient than individual calls when forecasting many products.
        
        Args:
            skus: List of product SKUs
            store: Store identifier
            days: Number of days to forecast
        
        Returns:
            Dictionary with forecasts list containing {sku, total_7d, avg_daily}
        """
        payload = {
            "store_id": store,
            "skus": skus,
            "days_ahead": days,
        }
        response = self.client.post("/forecast/bulk", json=payload)
        return self._handle_response(response)
    
    # =========================================================================
    # Hero Products
    # =========================================================================
    
    def get_heroes(
        self,
        store: str,
        lens: str = "balanced",
        limit: int = 20,
        category: str | None = None,
        subcategory: str | None = None,
    ) -> list[dict]:
        """
        Get hero products ranked by specified lens.
        
        Hero scores are computed on RESIDUAL demand (after weather normalization),
        so they reflect TRUE product performance, not weather-driven spikes.
        
        Args:
            store: Store identifier
            lens: Ranking method - one of:
                - "balanced": Overall hero score
                - "velocity": High turnover products
                - "margin_mix": High profit contributors
                - "residual_demand": Weather-normalized demand
                - "momentum": Trending upward
                - "stability": Consistent sellers
            limit: Maximum products to return
            category: Filter by category
            subcategory: Filter by subcategory
        
        Returns:
            List of hero products with scores and components.
        
        Example:
            >>> heroes = aoc.get_heroes("central", lens="margin_mix", limit=12)
            >>> for h in heroes:
            ...     print(f"{h['rank']}. {h['product_name']} - ${h['metrics']['margin_7d']:.2f}")
        """
        params: dict[str, Any] = {
            "store": store,
            "lens": lens,
            "limit": limit,
        }
        if category:
            params["category"] = category
        if subcategory:
            params["subcategory"] = subcategory
        
        response = self.client.get("/heroes/ranked", params=params)
        data = self._handle_response(response)
        return data.get("heroes", [])
    
    def get_heroes_by_context(
        self,
        store: str,
        purpose: str,
        limit: int = 12,
    ) -> dict:
        """
        Get heroes optimized for current context and purpose.
        
        Args:
            store: Store identifier
            purpose: Screen/display purpose:
                - "move_inventory": Focus on overstock/slow movers
                - "maximize_margin": High profit products
                - "delight_customers": Customer favorites
                - "clear_old_stock": Aging inventory
                - "promote_new": New arrivals
            limit: Products to return
        
        Returns:
            Dictionary with heroes and reasoning for selections.
        """
        params = {
            "store": store,
            "purpose": purpose,
            "limit": limit,
        }
        response = self.client.get("/heroes/by-context", params=params)
        return self._handle_response(response)
    
    # =========================================================================
    # Anomalies
    # =========================================================================
    
    def get_active_anomalies(self, store: str) -> list[dict]:
        """
        Get currently active anomalies affecting a store.
        
        Anomalies include weather events, holidays, local events, etc.
        that may affect normal demand patterns.
        
        Args:
            store: Store identifier
        
        Returns:
            List of active anomalies with type, severity, and recommendations.
        
        Example:
            >>> anomalies = aoc.get_active_anomalies("central")
            >>> for a in anomalies:
            ...     if a["severity"] == "high":
            ...         print(f"⚠️ {a['description']}")
            ...         for rec in a["recommendations"]:
            ...             print(f"  → {rec}")
        """
        params = {"store": store}
        response = self.client.get("/anomalies/active", params=params)
        data = self._handle_response(response)
        return data.get("anomalies", [])
    
    def create_anomaly(
        self,
        store: str,
        anomaly_type: str,
        description: str,
        severity: str = "moderate",
        start_time: datetime | str | None = None,
        end_time: datetime | str | None = None,
        impact: dict | None = None,
    ) -> dict:
        """
        Create a manual anomaly (for events, promotions, etc.).
        
        Args:
            store: Store identifier
            anomaly_type: Type of anomaly (weather, event, promo, etc.)
            description: Human-readable description
            severity: low, moderate, high
            start_time: When anomaly begins
            end_time: When anomaly ends
            impact: Expected impact factors
        
        Returns:
            Created anomaly record.
        """
        payload: dict[str, Any] = {
            "store": store,
            "type": anomaly_type,
            "description": description,
            "severity": severity,
        }
        if start_time:
            payload["start_time"] = (
                start_time.isoformat() if isinstance(start_time, datetime) else start_time
            )
        if end_time:
            payload["end_time"] = (
                end_time.isoformat() if isinstance(end_time, datetime) else end_time
            )
        if impact:
            payload["impact"] = impact
        
        response = self.client.post("/anomalies", json=payload)
        return self._handle_response(response)
    
    def delete_anomaly(self, anomaly_id: str) -> bool:
        """
        Remove an anomaly.
        
        Args:
            anomaly_id: UUID of anomaly to remove
        
        Returns:
            True if deleted successfully.
        """
        response = self.client.delete(f"/anomalies/{anomaly_id}")
        self._handle_response(response)
        return True


# Convenience function for one-off usage
def get_client(
    base_url: str | None = None,
    api_key: str | None = None,
) -> AOCClient:
    """
    Create an AOC client with optional overrides.
    
    Args:
        base_url: Override for AOC_API_URL environment variable
        api_key: Override for AOC_API_KEY environment variable
    
    Returns:
        Configured AOCClient instance
    """
    config = AOCConfig()
    if base_url:
        config.base_url = base_url
    if api_key:
        config.api_key = api_key
    return AOCClient(config)
