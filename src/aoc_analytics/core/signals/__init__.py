"""
Signals module - THE KEYSTONE

This module contains the weather-as-a-priori transformation logic.
Weather is NOT correlated with sales - it is transformed into behavioral
propensities BEFORE any other analysis occurs.

The keystone formula:
    at_home = 0.1 + 0.45*rain + 0.3*cold + 0.1*wind + 0.15*snow

Weather conditions become behavioral weights that normalize sales expectations.
"""

from .builder import (
    rebuild_behavioral_signals,
    _score_at_home,
    _score_out_and_about,
    _score_local_vibe,
)
from .payday_index import build_payday_index

__all__ = [
    "rebuild_behavioral_signals",
    "_score_at_home",
    "_score_out_and_about", 
    "_score_local_vibe",
    "build_payday_index",
]
