"""
strategies package - Autonomous Trading Strategies
"""
from strategies.ict_engine import ICTStrategyEngine, INSTRUMENT_PROFILES
from strategies.crt_engine import CRTStrategyEngine, CRT_INSTRUMENT_PROFILES
from strategies.snd_engine import SNDStrategyEngine, SND_INSTRUMENT_PROFILES
from strategies.trendline_engine import TrendlineStrategyEngine, TRENDLINE_PROFILES

import os
import importlib.machinery

try:
    _root_strat = os.path.join(os.path.dirname(os.path.dirname(__file__)), "strategies.py")
    if os.path.exists(_root_strat):
        _loader = importlib.machinery.SourceFileLoader("_root_strategies", _root_strat)
        _mod = _loader.load_module()
        analyze_strategy = getattr(_mod, "analyze_strategy", None)
    else:
        analyze_strategy = None
except Exception:
    analyze_strategy = None

__all__ = [
    "ICTStrategyEngine", "INSTRUMENT_PROFILES",
    "CRTStrategyEngine", "CRT_INSTRUMENT_PROFILES",
    "SNDStrategyEngine", "SND_INSTRUMENT_PROFILES",
    "TrendlineStrategyEngine", "TRENDLINE_PROFILES",
    "analyze_strategy"
]

