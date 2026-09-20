"""
bot/news_engine.py - Real-Time Economic Calendar & High-Impact News Engine

Monitors global high-impact economic events (CPI, NFP, FOMC, Rate Decisions, PPI, GDP)
via live financial calendar feeds and broadcasts pre-news alerts and volatility warnings to Telegram.
Provides news-shield protection to avoid high-slippage executions.
"""

import os
import time
import logging
import asyncio
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List, Optional, Callable, Set
import httpx

logger = logging.getLogger("NewsEngine")

CALENDAR_FEED_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"

COUNTRY_FLAGS = {
    "USD": "🇺🇸",
    "EUR": "🇪🇺",
    "GBP": "🇬🇧",
    "JPY": "🇯🇵",
    "AUD": "🇦🇺",
    "CAD": "🇨🇦",
    "CHF": "🇨🇭",
    "NZD": "🇳🇿",
    "CNY": "🇨🇳",
    "ALL": "🌐"
}

ASSET_IMPACT_MAP = {
    "USD": ["Gold (XAU/USD)", "EUR/USD", "GBP/USD", "USD/JPY", "AUD/USD", "BTC/USD"],
    "EUR": ["EUR/USD", "EUR/GBP", "EUR/JPY", "EUR/AUD"],
    "GBP": ["GBP/USD", "EUR/GBP", "GBP/JPY", "GBP/AUD"],
    "JPY": ["USD/JPY", "EUR/JPY", "GBP/JPY", "AUD/JPY"],
    "AUD": ["AUD/USD", "AUD/JPY", "EUR/AUD", "GBP/AUD"],
    "CAD": ["USD/CAD", "EUR/CAD", "CAD/JPY"],
    "CHF": ["USD/CHF", "EUR/CHF", "GBP/CHF"],
    "NZD": ["NZD/USD", "AUD/NZD", "NZD/JPY"],
    "ALL": ["Global Markets", "Gold (XAU/USD)", "Crypto"]
}

class EconomicNewsEngine:
    def __init__(self, broadcast_func: Optional[Callable] = None):
        self.broadcast_func = broadcast_func
        self.events: List[Dict[str, Any]] = []
        self.last_fetch_time: float = 0.0
        self.fetch_interval_secs: float = 3600.0 * 3.0  # Refresh every 3 hours
        self.triggered_alerts: Set[str] = set()
        self.is_running: bool = False
        self.is_shield_enabled: bool = True

    def set_broadcast_callback(self, func: Callable):
        self.broadcast_func = func

    async def notify(self, message: str):
        if self.broadcast_func:
            try:
                await self.broadcast_func(message)
            except Exception as e:
                logger.warning(f"[NewsEngine] Broadcast notification error: {e}")

    async def fetch_calendar(self, force: bool = False) -> bool:
        """Fetches the weekly economic calendar feed."""
        now_ts = time.time()
        if not force and self.events and (now_ts - self.last_fetch_time < self.fetch_interval_secs):
            return True

        try:
            async with httpx.AsyncClient(timeout=12.0) as client:
                resp = await client.get(CALENDAR_FEED_URL)
                if resp.status_code == 200:
                    raw_events = resp.json()
                    parsed = []
                    for item in raw_events:
                        impact = item.get("impact", "Low")
                        if impact not in ["High", "Medium"]:
                            continue

                        # Parse ISO date string (e.g. 2026-09-21T11:00:00-04:00)
                        date_str = item.get("date", "")
                        try:
                            # Handle ISO 8601 with timezone offset
                            event_dt = datetime.fromisoformat(date_str)
                            # Convert to UTC
                            event_dt_utc = event_dt.astimezone(timezone.utc)
                        except Exception:
                            continue

                        country = str(item.get("country", "USD")).upper()
                        parsed.append({
                            "title": item.get("title", "Economic Event"),
                            "country": country,
                            "flag": COUNTRY_FLAGS.get(country, "🌐"),
                            "impact": impact,
                            "forecast": item.get("forecast", "N/A"),
                            "previous": item.get("previous", "N/A"),
                            "dt_utc": event_dt_utc,
                            "raw_date": date_str,
                            "id": f"{country}_{item.get('title')}_{event_dt_utc.strftime('%Y%m%d%H%M')}"
                        })

                    # Sort by upcoming time
                    parsed.sort(key=lambda x: x["dt_utc"])
                    self.events = parsed
                    self.last_fetch_time = now_ts
                    logger.info(f"✅ [NewsEngine] Successfully fetched {len(self.events)} High/Medium impact events.")
                    return True
                else:
                    logger.warning(f"[NewsEngine] Failed to fetch calendar: HTTP {resp.status_code}")
                    return False
        except Exception as e:
            logger.error(f"[NewsEngine] Calendar fetch exception: {e}")
            return False

    def get_todays_events(self) -> List[Dict[str, Any]]:
        """Returns all high/medium impact events for today (UTC)."""
        now_utc = datetime.now(timezone.utc)
        today_date = now_utc.date()
        return [e for e in self.events if e["dt_utc"].date() == today_date]

    def get_upcoming_events(self, hours_ahead: int = 24) -> List[Dict[str, Any]]:
        """Returns upcoming events within the next N hours."""
        now_utc = datetime.now(timezone.utc)
        horizon = now_utc + timedelta(hours=hours_ahead)
        return [e for e in self.events if now_utc <= e["dt_utc"] <= horizon]

    def get_news_dashboard(self) -> str:
        """Renders an attractive, comprehensive Telegram Markdown card of economic events."""
        now_utc = datetime.now(timezone.utc)
        local_hour = (now_utc.hour + 1) % 24
        now_str = f"{now_utc.strftime('%A, %H:%M UTC')} ({local_hour:02d}:{now_utc.minute:02d} WAT)"

        todays = self.get_todays_events()
        upcoming_24h = self.get_upcoming_events(24)

        display_events = todays if len(todays) > 0 else upcoming_24h[:6]

        if not display_events:
            event_lines = "  _No High or Medium impact economic events scheduled for today._\n  _Clean market conditions for technical / ICT trading!_"
        else:
            lines = []
            for e in display_events:
                dt = e["dt_utc"]
                local_h = (dt.hour + 1) % 24
                time_str = f"{dt.strftime('%H:%M UTC')} / {local_h:02d}:{dt.minute:02d} WAT"
                diff_mins = int((dt - now_utc).total_seconds() / 60.0)

                if diff_mins < -60:
                    status_tag = "✅ _Completed_"
                elif -60 <= diff_mins <= 0:
                    status_tag = "🔥 _Released Just Now_"
                elif 0 < diff_mins <= 60:
                    status_tag = f"⏳ **In {diff_mins} mins**"
                else:
                    hrs = diff_mins // 60
                    mins = diff_mins % 60
                    status_tag = f"⏰ In {hrs}h {mins}m"

                impact_icon = "🔴 **HIGH**" if e["impact"] == "High" else "🟠 **MED**"
                prev_txt = f"Prev: `{e['previous']}`" if e.get("previous") else ""
                fc_txt = f"Forecast: `{e['forecast']}`" if e.get("forecast") else ""
                stats_part = f" ({prev_txt} | {fc_txt})" if (prev_txt or fc_txt) else ""

                lines.append(
                    f"• {e['flag']} **{e['country']}** — **{e['title']}**\n"
                    f"  {impact_icon} | 🕒 `{time_str}` | {status_tag}{stats_part}"
                )
            event_lines = "\n\n".join(lines)

        card = (
            f"📰 ━━━━━━━━━━━━━━━━━━━ 📰\n"
            f"   📊 **GLOBAL ECONOMIC CALENDAR**\n"
            f"📰 ━━━━━━━━━━━━━━━━━━━ 📰\n\n"
            f"🕒 **Current Time**: `{now_str}`\n\n"
            f"🚨 **Key Scheduled Releases**:\n"
            f"{event_lines}\n\n"
            f"💡 **Trading Advisory**:\n"
            f"• 🔴 **Red Folder releases** (CPI, NFP, FOMC) cause severe spread widening (20¢–$3.00 on Gold).\n"
            f"• Bot will broadcast 15-minute advance warnings before high-impact events.\n"
            f"📰 ━━━━━━━━━━━━━━━━━━━ 📰"
        )
        return card

    def is_news_freeze_active(self, symbol: str = "XAUUSD", window_minutes: int = 5) -> bool:
        """Returns True if high-impact news for this asset is releasing in <= window_mins or just released."""
        if not self.is_shield_enabled:
            return False

        now_utc = datetime.now(timezone.utc)
        sym_clean = symbol.upper().replace("/", "").replace("-", "")

        for e in self.events:
            if e["impact"] != "High":
                continue

            # Check if this currency affects the symbol
            country = e["country"]
            impacted_assets = ASSET_IMPACT_MAP.get(country, [])
            affects_symbol = (
                country in sym_clean or
                any(sym_clean in a.upper().replace("/", "") for a in impacted_assets) or
                (country == "USD" and sym_clean in ["XAUUSD", "BTCUSD", "EURUSD", "GBPUSD", "USDJPY", "AUDUSD"])
            )

            if affects_symbol:
                diff_secs = (e["dt_utc"] - now_utc).total_seconds()
                diff_mins = diff_secs / 60.0
                # Between window_minutes before and window_minutes after
                if -window_minutes <= diff_mins <= window_minutes:
                    return True
        return False

    async def check_and_send_alerts(self):
        """Checks for events occurring in ~15 minutes and fires notifications."""
        now_utc = datetime.now(timezone.utc)

        for e in self.events:
            if e["impact"] not in ["High", "Medium"]:
                continue

            event_id = e["id"]
            dt = e["dt_utc"]
            diff_secs = (dt - now_utc).total_seconds()
            diff_mins = diff_secs / 60.0

            # 1. 15-Minute Advance Warning (between 13 and 16 minutes prior)
            alert_15m_key = f"{event_id}_15m"
            if 13.0 <= diff_mins <= 16.0 and alert_15m_key not in self.triggered_alerts:
                self.triggered_alerts.add(alert_15m_key)
                local_h = (dt.hour + 1) % 24
                time_str = f"{dt.strftime('%H:%M UTC')} ({local_h:02d}:{dt.minute:02d} WAT)"
                impacted = ", ".join(ASSET_IMPACT_MAP.get(e["country"], ["All Pairs"]))

                impact_title = "💥 HIGH-IMPACT NEWS IN 15 MINUTES!" if e["impact"] == "High" else "⚠️ MEDIUM-IMPACT NEWS IN 15 MINUTES"
                badge = "🔴 CRITICAL / HIGH VOLATILITY" if e["impact"] == "High" else "🟠 MEDIUM VOLATILITY"

                msg = (
                    f"🚨 ━━━━━━━━━━━━━━━━━━━━ 🚨\n"
                    f"   {impact_title}\n"
                    f"🚨 ━━━━━━━━━━━━━━━━━━━━ 🚨\n\n"
                    f"📅 **Event**: {e['flag']} **{e['country']} — {e['title']}**\n"
                    f"⏰ **Release Time**: `{time_str}` (In **15 mins**)\n"
                    f"🔥 **Impact Rating**: {badge}\n\n"
                    f"🎯 **Affected Assets**: `{impacted}`\n"
                    f"📊 **Forecast**: `{e['forecast']}` | **Previous**: `{e['previous']}`\n\n"
                    f"🛡 **Risk Advisory**:\n"
                    f"• Spreads and slippage will spike during the release.\n"
                    f"• Secure existing winning trades by locking Breakeven.\n"
                    f"• Avoid placing new market orders immediately inside the release candle.\n"
                    f"🚨 ━━━━━━━━━━━━━━━━━━━━ 🚨"
                )
                logger.info(f"📢 [NewsEngine] Sending 15m advance warning for: {e['title']}")
                await self.notify(msg)

            # 2. Event Release Flash (between -1 and 2 minutes after release)
            alert_now_key = f"{event_id}_now"
            if -1.0 <= diff_mins <= 2.0 and alert_now_key not in self.triggered_alerts:
                self.triggered_alerts.add(alert_now_key)
                msg = (
                    f"⚡ ━━━━━━━━━━━━━━━━━━━━ ⚡\n"
                    f"   📢 **ECONOMIC NEWS RELEASED NOW!**\n"
                    f"⚡ ━━━━━━━━━━━━━━━━━━━━ ⚡\n\n"
                    f"📅 **Event**: {e['flag']} **{e['country']} — {e['title']}**\n"
                    f"🔥 **Impact**: {e['impact'].upper()}\n"
                    f"📊 **Expected**: `{e['forecast']}` | **Prior**: `{e['previous']}`\n\n"
                    f"🌊 Expect immediate volatility and liquidity expansion across markets.\n"
                    f"⚡ ━━━━━━━━━━━━━━━━━━━━ ⚡"
                )
                logger.info(f"📢 [NewsEngine] Sending release flash for: {e['title']}")
                await self.notify(msg)

    async def run_loop(self):
        """Main background loop fetching calendar and checking alert schedules."""
        self.is_running = True
        logger.info("📰 [NewsEngine] Economic Calendar & News Alert Engine started.")
        # Initial fetch
        await self.fetch_calendar(force=True)

        while self.is_running:
            try:
                # Refresh feed every 3 hours
                await self.fetch_calendar()
                # Check for upcoming alerts
                await self.check_and_send_alerts()
            except Exception as e:
                logger.error(f"[NewsEngine] Error in run loop: {e}")
            await asyncio.sleep(30)

    def stop(self):
        self.is_running = False
