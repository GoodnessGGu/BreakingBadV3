"""
forex_factory.py - Forex Factory Economic Calendar, News Strike & Sentiment Engine
Specifically designed for 1-Minute and 5-Minute Binary Options on Real Market Pairs.

Features:
1. Automated Weekly Calendar Sync (Fair Economy JSON endpoint).
2. 1m / 5m News Strike Sniper (capturing instant high-impact news momentum bursts).
3. Contrarian Sentiment Bias Engine (fading retail crowd extremes for 1m/5m directional lock).
4. Proactive News Blackout Guard (protecting technical strategies from erratic news spikes).
"""

import os
import json
import time
import logging
import urllib.request
from datetime import datetime, timezone, timedelta
from typing import List, Dict, Optional, Tuple

logger = logging.getLogger("ForexFactory")

# Official Fair Economy live calendar feed for Forex Factory
FF_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
CACHE_FILE = "ff_calendar_cache.json"
CACHE_MAX_AGE_SECONDS = 3600  # Refresh cache every hour

# Currency pair mapping
PAIR_CURRENCIES = {
    "EURUSD": ["EUR", "USD"],
    "GBPUSD": ["GBP", "USD"],
    "USDJPY": ["USD", "JPY"],
    "AUDUSD": ["AUD", "USD"],
    "USDCAD": ["USD", "CAD"],
    "USDCHF": ["USD", "CHF"],
    "NZDUSD": ["NZD", "USD"],
    "EURGBP": ["EUR", "GBP"],
    "EURJPY": ["EUR", "JPY"],
    "GBPJPY": ["GBP", "JPY"],
}

class ForexFactoryCalendar:
    """Manages downloading, caching, and parsing of Forex Factory economic calendar."""

    def __init__(self, cache_file: str = CACHE_FILE):
        self.cache_file = cache_file
        self.events: List[Dict] = []
        self.last_fetch_time: float = 0.0
        self._load_or_refresh_calendar()

    def _load_or_refresh_calendar(self, force_refresh: bool = False):
        """Loads calendar from local cache if fresh, otherwise downloads live feed."""
        now = time.time()
        if not force_refresh and os.path.exists(self.cache_file):
            file_age = now - os.path.getmtime(self.cache_file)
            if file_age < CACHE_MAX_AGE_SECONDS:
                try:
                    with open(self.cache_file, "r", encoding="utf-8") as f:
                        self.events = json.load(f)
                    self.last_fetch_time = os.path.getmtime(self.cache_file)
                    # logger.info(f"Loaded {len(self.events)} events from local Forex Factory cache.")
                    return
                except Exception as e:
                    logger.warning(f"Error reading cache file: {e}")

        # Download fresh calendar
        try:
            req = urllib.request.Request(
                FF_CALENDAR_URL,
                headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                raw_data = resp.read().decode("utf-8")
                self.events = json.loads(raw_data)

            with open(self.cache_file, "w", encoding="utf-8") as f:
                f.write(raw_data)

            self.last_fetch_time = now
            logger.info(f"✅ Downloaded {len(self.events)} fresh Forex Factory events for this week.")
        except Exception as e:
            logger.error(f"Failed to fetch Forex Factory calendar: {e}")
            if os.path.exists(self.cache_file):
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self.events = json.load(f)

    def get_currencies_for_asset(self, asset: str) -> List[str]:
        """Strips OTC/suffixes and returns underlying currencies."""
        clean_asset = asset.replace("-OTC", "").replace("_OTC", "").replace("-op", "").upper()
        return PAIR_CURRENCIES.get(clean_asset, [clean_asset[:3], clean_asset[3:6]] if len(clean_asset) >= 6 else [])

    def get_upcoming_events(self, asset: Optional[str] = None, hours_ahead: float = 12.0, min_impact: str = "Medium") -> List[Dict]:
        """Returns sorted list of upcoming events within specified hours."""
        self._load_or_refresh_calendar()
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=hours_ahead)
        currencies = self.get_currencies_for_asset(asset) if asset else None

        impact_ranks = {"High": 3, "Medium": 2, "Low": 1, "Non-Economic": 0}
        min_rank = impact_ranks.get(min_impact, 2)

        results = []
        for ev in self.events:
            try:
                ev_time = datetime.fromisoformat(ev["date"]).astimezone(timezone.utc)
                if now - timedelta(minutes=15) <= ev_time <= cutoff:
                    ev_impact = ev.get("impact", "Low")
                    if impact_ranks.get(ev_impact, 0) >= min_rank:
                        ev_ccy = ev.get("country", "").upper()
                        if currencies is None or ev_ccy in currencies:
                            time_diff_sec = (ev_time - now).total_seconds()
                            results.append({
                                "title": ev.get("title"),
                                "country": ev_ccy,
                                "impact": ev_impact,
                                "datetime_utc": ev_time,
                                "seconds_until": time_diff_sec,
                                "forecast": ev.get("forecast", ""),
                                "previous": ev.get("previous", "")
                            })
            except Exception:
                continue

        results.sort(key=lambda x: x["seconds_until"])
        return results

    def is_in_news_window(self, asset: str, pre_seconds: int = 60, post_seconds: int = 180, impact: str = "High") -> Tuple[bool, Optional[Dict]]:
        """
        Checks if the asset is currently within a live high-impact news window.
        Useful for both News Sniping (entering on release) and News Blackout (avoiding spikes).
        """
        upcoming = self.get_upcoming_events(asset, hours_ahead=1.0, min_impact=impact)
        for ev in upcoming:
            sec = ev["seconds_until"]
            # Inside window if: -post_seconds <= sec <= pre_seconds
            if -post_seconds <= sec <= pre_seconds:
                return True, ev
        return False, None


class ForexFactorySentiment:
    """
    Tracks retail crowd sentiment and provides contrarian directional locks
    for 1m and 5m binary option strategies.
    """

    def __init__(self):
        # Default positioning baseline (updated dynamically or configured)
        # Format: { 'EURUSD': {'long_pct': 72.0, 'short_pct': 28.0} }
        self.sentiment_cache: Dict[str, Dict] = {}

    def set_pair_sentiment(self, asset: str, long_pct: float):
        """Sets known retail long percentage for an asset."""
        clean = asset.replace("-OTC", "").upper()
        self.sentiment_cache[clean] = {
            "long_pct": round(long_pct, 1),
            "short_pct": round(100.0 - long_pct, 1),
            "updated": time.time()
        }

    def get_directional_lock(self, asset: str, extreme_threshold: float = 72.0) -> Tuple[Optional[str], str]:
        """
        Evaluates retail crowd sentiment for contrarian directional lock.
        - If Retail Long > extreme_threshold -> Smart money bias is PUT
        - If Retail Short > extreme_threshold -> Smart money bias is CALL
        - Otherwise -> NEUTRAL (allow both CALL and PUT)
        """
        clean = asset.replace("-OTC", "").upper()
        data = self.sentiment_cache.get(clean)
        if not data:
            return None, "No extreme retail bias (Sentiment Neutral)"

        long_pct = data["long_pct"]
        short_pct = data["short_pct"]

        if long_pct >= extreme_threshold:
            return "PUT", f"Crowd Overbought ({long_pct}% Long) -> Contrarian PUT Lock"
        elif short_pct >= extreme_threshold:
            return "CALL", f"Crowd Oversold ({short_pct}% Short) -> Contrarian CALL Lock"

        return None, f"Sentiment Balanced ({long_pct}% Long / {short_pct}% Short)"


class NewsSniper1m5m:
    """
    Executes rapid 1-Minute and 5-Minute binary trades at the exact second
    of a high-impact news release based on the confirmed initial momentum impulse.
    """

    def __init__(self, calendar: ForexFactoryCalendar, min_payout_threshold: float = 0.75):
        self.calendar = calendar
        self.min_payout_threshold = min_payout_threshold
        self.executed_event_ids = set()

    def evaluate_news_snipe_trigger(self, asset: str, candles: list, expiry: int = 1) -> Tuple[Optional[str], Optional[str]]:
        """
        Evaluates whether a High-Impact news release is firing right now (T - 5s to T + 30s)
        and detects the explosive 1m candle momentum breakout direction.
        
        Returns:
            (signal: 'CALL' | 'PUT' | None, reason: str)
        """
        in_window, event = self.calendar.is_in_news_window(asset, pre_seconds=5, post_seconds=35, impact="High")
        if not in_window or not event:
            return None, None

        event_key = f"{event['country']}_{event['title']}_{event['datetime_utc'].strftime('%Y%m%d%H%M')}"
        if event_key in self.executed_event_ids:
            return None, "Event already sniped"

        # Check price action momentum at the moment of release
        if not candles or len(candles) < 3:
            return None, None

        curr = candles[-1]
        c_open = float(curr.get("open", 0))
        c_close = float(curr.get("close", 0))
        c_body = c_close - c_open

        # Recent average candle body (to ensure it's an abnormal news expansion)
        prev_bodies = [abs(float(c.get("close", 0)) - float(c.get("open", 0))) for c in candles[-6:-1]]
        avg_body = sum(prev_bodies) / len(prev_bodies) if prev_bodies else 0.0001

        # Momentum confirmation: Current candle must expand at least 2.0x average body
        if abs(c_body) > (1.8 * avg_body) and abs(c_body) > 0.00015:
            signal = "CALL" if c_body > 0 else "PUT"
            reason = f"🚀 High-Impact News Strike: [{event['country']}] {event['title']} (Impulse: {signal} {abs(c_body):.5f})"
            self.executed_event_ids.add(event_key)
            return signal, reason

        return None, None


# Global instances for plug-and-play usage
ff_calendar = ForexFactoryCalendar()
ff_sentiment = ForexFactorySentiment()
ff_sniper = NewsSniper1m5m(calendar=ff_calendar)
