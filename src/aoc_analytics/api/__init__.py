"""
FastAPI server for AOC Analytics.

Provides REST API endpoints for LocalBot and other consumers
to access analytics without direct Python imports.
"""

from .server import app, create_app

__all__ = ["app", "create_app"]
