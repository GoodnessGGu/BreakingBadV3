"""
bot/session_notifier.py - Real-Time Market Openings & Trading Session Alert Engine

Monitors global market hours and sends rich, prettified notifications to Telegram
when Forex, Gold, Asian, London, and New York trading sessions open and close.
"""

import asyncio
import logging
from datetime import datetime, time as dtime, timezone
from typing import Callable, Optional, Dict, Any, List

logger = logging.getLogger("SessionNotifier")

class MarketSessionNotifier:
    def __init__(self, broadcast_func: Optional[Callable] = None):
        self.broadcast_func = broadcast_func
        self.is_running = False
        # Store (date_str, event_id) to prevent duplicate triggers
        self.triggered_events = set()

    def set_broadcast_callback(self, func: Callable):
        self.broadcast_func = func

    async def notify(self, message: str):
        if self.broadcast_func:
            try:
                await self.broadcast_func(message)
            except Exception as e:
                logger.warning(f"[SessionNotifier] Broadcast error: {e}")

    @staticmethod
    def get_market_status(now_utc: Optional[datetime] = None) -> Dict[str, Any]:
        """Calculates current market & session states in UTC."""
        if not now_utc:
            now_utc = datetime.now(timezone.utc)

        weekday = now_utc.weekday()  # 0=Mon, 1=Tue, 2=Wed, 3=Thu, 4=Fri, 5=Sat, 6=Sun
        hour = now_utc.hour
        minute = now_utc.minute
        time_dec = hour + (minute / 60.0)

        # Forex Market Open: Sunday 22:00 UTC -> Friday 21:00 UTC
        if weekday == 6:  # Sunday
            forex_open = hour >= 22
            gold_open = hour >= 23
        elif weekday == 5:  # Saturday
            forex_open = False
            gold_open = False
        elif weekday == 4:  # Friday
            forex_open = hour < 21 or (hour == 21 and minute == 0)
            gold_open = hour < 21 or (hour == 21 and minute == 0)
        else:  # Mon-Thu
            forex_open = True
            # Gold has a daily maintenance break Mon-Thu 21:00-22:00 UTC
            gold_open = not (hour == 21)

        # Active Daily Sessions (Mon-Fri)
        active_sessions = []
        if weekday in [0, 1, 2, 3, 4] or (weekday == 6 and hour >= 22):
            # Sydney: 22:00 - 07:00 UTC
            if time_dec >= 22.0 or time_dec < 7.0:
                active_sessions.append("Sydney / Wellington 🇦🇺")
            # Tokyo: 00:00 - 09:00 UTC
            if 0.0 <= time_dec < 9.0 and weekday != 6:
                active_sessions.append("Tokyo / Asian 🇯🇵")
            # Frankfurt / London: 07:00 - 16:00 UTC
            if 7.0 <= time_dec < 16.0 and weekday != 6:
                active_sessions.append("London 🇬🇧")
            # New York: 12:00 - 21:00 UTC
            if 12.0 <= time_dec < 21.0 and weekday != 6:
                active_sessions.append("New York 🇺🇸")

        return {
            "forex_open": forex_open,
            "gold_open": gold_open,
            "crypto_open": True,  # 24/7
            "active_sessions": active_sessions,
            "now_utc": now_utc
        }

    def get_session_dashboard(self) -> str:
        """Returns a prettified markdown overview of current market states and schedules."""
        status = self.get_market_status()
        now = status["now_utc"]
        utc_str = now.strftime("%A, %H:%M UTC")
        local_hour = (now.hour + 1) % 24
        local_str = f"{local_hour:02d}:{now.minute:02d} WAT (UTC+1)"

        fx_tag = "🟢 **OPEN**" if status["forex_open"] else "🔴 **CLOSED (Weekend)**"
        gold_tag = "🟢 **OPEN**" if status["gold_open"] else "🔴 **CLOSED (Weekend)**"
        crypto_tag = "🟢 **OPEN (24/7)**"

        sessions_str = "\n".join([f"  • ⚡ `{s}`" for s in status["active_sessions"]]) if status["active_sessions"] else "  _No active major sessions (Off-Hours / Weekend)_"

        card = (
            f"🌐 ━━━━━━━━━━━━━━━━━━━ 🌐\n"
            f"      📊 **GLOBAL MARKET SESSIONS**\n"
            f"🌐 ━━━━━━━━━━━━━━━━━━━ 🌐\n\n"
            f"🕒 **Current Time**: `{utc_str}` (`{local_str}`)\n\n"
            f"🏛 **Asset Market Status**:\n"
            f"  • 💱 **Forex Pairs**: {fx_tag}\n"
            f"  • 🥇 **Gold (XAU/USD)**: {gold_tag}\n"
            f"  • 🪙 **Crypto (BTC/USD)**: {crypto_tag}\n\n"
            f"🔥 **Currently Active Sessions**:\n{sessions_str}\n\n"
            f"⏰ **Major Daily Timetable (UTC+1)**:\n"
            f"  • 🇦🇺 `23:00` — Sydney Open\n"
            f"  • 🇯🇵 `01:00` — Tokyo Asian Open\n"
            f"  • 🇬🇧 `08:00` — London Killzone Open\n"
            f"  • 🇺🇸 `13:00` — New York Overlap Open\n"
            f"  • 🔔 `14:30` — NYSE Wall Street Open\n"
            f"  • 🇬🇧 `17:00` — London Session Close\n"
            f"  • 🏁 `22:00` — New York / Week Close\n"
            f"🌐 ━━━━━━━━━━━━━━━━━━━ 🌐"
        )
        return card

    async def check_and_notify_events(self):
        """Checks for exact market open/close and session triggers."""
        now = datetime.now(timezone.utc)
        weekday = now.weekday()
        hour = now.hour
        minute = now.minute
        date_key = now.strftime("%Y-%m-%d")

        # Helper to ensure one-time firing
        async def trigger_event(event_id: str, message: str):
            key = (date_key, event_id)
            if key not in self.triggered_events:
                self.triggered_events.add(key)
                # Cleanup old keys from previous days
                if len(self.triggered_events) > 50:
                    self.triggered_events = {k for k in self.triggered_events if k[0] == date_key}
                logger.info(f"📢 [SessionNotifier] Triggering: {event_id}")
                await self.notify(message)

        # 1. Sunday 22:00 UTC (23:00 UTC+1) -> Forex Week Open & Sydney Open
        if weekday == 6 and hour == 22 and minute == 0:
            msg = (
                f"🌏 ━━━━━━━━━━━━━━━━━━━━ 🌏\n"
                f"   🚀 **FOREX MARKET OPEN — NEW WEEK**\n"
                f"🌏 ━━━━━━━━━━━━━━━━━━━━ 🌏\n\n"
                f"📅 **Welcome to the Trading Week!**\n"
                f"⏰ **Time**: `22:00 UTC` (`23:00 WAT / UTC+1`)\n\n"
                f"📈 **Active Market**: Sydney & Wellington 🇦🇺 🇳🇿\n"
                f"💱 **Assets**: All standard Forex pairs (`EURUSD`, `GBPUSD`, `USDJPY`, etc.) are now **LIVE**.\n\n"
                f"💡 **Pro Tip**: Initial 30–60 mins can have wider spreads as broker liquidity connects. Manage risk carefully!\n"
                f"🤖 **Bot Status**: Copiers and engines are actively scanning.\n"
                f"🌏 ━━━━━━━━━━━━━━━━━━━━ 🌏"
            )
            await trigger_event("forex_week_open", msg)

        # 2. Sunday 23:00 UTC (00:00 UTC+1 Monday) -> Gold (XAUUSD) Market Open
        if (weekday == 6 and hour == 23 and minute == 0) or (weekday == 0 and hour == 0 and minute == 0):
            msg = (
                f"🥇 ━━━━━━━━━━━━━━━━━━━━ 🥇\n"
                f"   ✨ **GOLD (XAU/USD) MARKET IS NOW OPEN!**\n"
                f"🥇 ━━━━━━━━━━━━━━━━━━━━ 🥇\n\n"
                f"⏰ **Time**: `23:00 UTC` (`00:00 Midnight UTC+1`)\n"
                f"👑 **Asset**: Spot Gold (`XAUUSD` / Marginal CFD #74)\n\n"
                f"🎯 **Trading Engines Activated**:\n"
                f"  • 🤖 **ICT Autonomous SMC Engine** (Scanning 15M Sweeps & FVGs)\n"
                f"  • 📡 **Gold Pips Hunter Signal Copier** (Listening for live setups)\n"
                f"  • 📊 **CallistoFx Zone Watcher**\n\n"
                f"💰 Let's have a profitable and disciplined week!\n"
                f"🥇 ━━━━━━━━━━━━━━━━━━━━ 🥇"
            )
            await trigger_event("gold_week_open", msg)

        # 3. Tokyo / Asian Session Open (Daily Mon-Fri at 00:00 UTC / 01:00 UTC+1)
        if weekday in [0, 1, 2, 3, 4] and hour == 0 and minute == 0:
            msg = (
                f"🗾 ━━━━━━━━━━━━━━━━━━━━ 🗾\n"
                f"   🏯 **TOKYO / ASIAN SESSION OPEN**\n"
                f"🗾 ━━━━━━━━━━━━━━━━━━━━ 🗾\n\n"
                f"⏰ **Time**: `00:00 UTC` (`01:00 WAT / UTC+1`)\n"
                f"🌏 **Markets**: Tokyo 🇯🇵, Singapore 🇸🇬, Hong Kong 🇭🇰\n\n"
                f"🔍 **Key Asset Focus**:\n"
                f"  • `USD/JPY`, `AUD/USD`, `NZD/USD`, `AUD/JPY`\n"
                f"  • `XAU/USD` (Asian Range Formation)\n\n"
                f"📊 **ICT Insight**: The Asian session typically defines the initial liquidity range. Look out for range extremes for London sweeps!\n"
                f"🗾 ━━━━━━━━━━━━━━━━━━━━ 🗾"
            )
            await trigger_event(f"tokyo_open_{weekday}", msg)

        # 4. Frankfurt / European Pre-Market (Daily Mon-Fri at 06:00 UTC / 07:00 UTC+1)
        if weekday in [0, 1, 2, 3, 4] and hour == 6 and minute == 0:
            msg = (
                f"🇩🇪 ━━━━━━━━━━━━━━━━━━━━ 🇩🇪\n"
                f"   🏛 **FRANKFURT / EUROPEAN PRE-MARKET**\n"
                f"🇩🇪 ━━━━━━━━━━━━━━━━━━━━ 🇩🇪\n\n"
                f"⏰ **Time**: `06:00 UTC` (`07:00 WAT / UTC+1`)\n"
                f"🇪🇺 **Markets**: Frankfurt 🇩🇪, Zurich 🇨🇭, Paris 🇫🇷\n\n"
                f"⚡ **Early Volatility**: European desks opening; preparation for London session momentum.\n"
                f"🇩🇪 ━━━━━━━━━━━━━━━━━━━━ 🇩🇪"
            )
            await trigger_event(f"frankfurt_open_{weekday}", msg)

        # 5. London Session Open (Daily Mon-Fri at 07:00 UTC / 08:00 UTC+1)
        if weekday in [0, 1, 2, 3, 4] and hour == 7 and minute == 0:
            msg = (
                f"🇬🇧 ━━━━━━━━━━━━━━━━━━━━ 🇬🇧\n"
                f"   ⚡ **LONDON SESSION OPEN (KILLZONE)**\n"
                f"🇬🇧 ━━━━━━━━━━━━━━━━━━━━ 🇬🇧\n\n"
                f"⏰ **Time**: `07:00 UTC` (`08:00 WAT / UTC+1`)\n"
                f"🏙 **Market**: London Financial Centre 🇬🇧\n\n"
                f"🔥 **High Liquidity Window**:\n"
                f"  • Major volume in `EUR/USD`, `GBP/USD`, `XAU/USD`\n"
                f"  • High probability of the classic **ICT Judas Swing** (sweeping Asian highs/lows before expansion)\n\n"
                f"🎯 **Bot**: Auto-monitoring order flows and displacement.\n"
                f"🇬🇧 ━━━━━━━━━━━━━━━━━━━━ 🇬🇧"
            )
            await trigger_event(f"london_open_{weekday}", msg)

        # 6. New York Pre-Market / Overlap (Daily Mon-Fri at 12:00 UTC / 13:00 UTC+1)
        if weekday in [0, 1, 2, 3, 4] and hour == 12 and minute == 0:
            msg = (
                f"🇺🇸 ━━━━━━━━━━━━━━━━━━━━ 🇺🇸\n"
                f"   🗽 **NEW YORK SESSION PRE-MARKET & OVERLAP**\n"
                f"🇺🇸 ━━━━━━━━━━━━━━━━━━━━ 🇺🇸\n\n"
                f"⏰ **Time**: `12:00 UTC` (`13:00 WAT / UTC+1`)\n"
                f"💥 **London + New York Overlap**: The highest volume and liquidity trading window of the 24-hour cycle.\n\n"
                f"📊 **Focus Assets**: `XAU/USD`, `EUR/USD`, `GBP/USD`, `BTC/USD`\n"
                f"🇺🇸 ━━━━━━━━━━━━━━━━━━━━ 🇺🇸"
            )
            await trigger_event(f"ny_overlap_{weekday}", msg)

        # 7. NYSE Wall Street Open (Daily Mon-Fri at 13:30 UTC / 14:30 UTC+1)
        if weekday in [0, 1, 2, 3, 4] and hour == 13 and minute == 30:
            msg = (
                f"🔔 ━━━━━━━━━━━━━━━━━━━━ 🔔\n"
                f"   🏛 **WALL STREET / NYSE EQUITY BELL**\n"
                f"🔔 ━━━━━━━━━━━━━━━━━━━━ 🔔\n\n"
                f"⏰ **Time**: `13:30 UTC` (`14:30 WAT / UTC+1`)\n"
                f"🇺🇸 **Market**: New York Stock Exchange & US Institutional Desks\n\n"
                f"⚡ **Maximum Volatility**: High-impact US news releases, heavy Gold momentum, rapid FVG formations.\n"
                f"🛡 Ensure strict stop-loss protection is active!\n"
                f"🔔 ━━━━━━━━━━━━━━━━━━━━ 🔔"
            )
            await trigger_event(f"nyse_open_{weekday}", msg)

        # 8. London Session Close (Daily Mon-Fri at 16:00 UTC / 17:00 UTC+1)
        if weekday in [0, 1, 2, 3, 4] and hour == 16 and minute == 0:
            msg = (
                f"🇬🇧 ━━━━━━━━━━━━━━━━━━━━ 🇬🇧\n"
                f"   🏁 **LONDON SESSION CLOSE (LONDON FIX)**\n"
                f"🇬🇧 ━━━━━━━━━━━━━━━━━━━━ 🇬🇧\n\n"
                f"⏰ **Time**: `16:00 UTC` (`17:00 WAT / UTC+1`)\n"
                f"📉 **Note**: European liquidity exits; market moves into late New York consolidation.\n"
                f"🇬🇧 ━━━━━━━━━━━━━━━━━━━━ 🇬🇧"
            )
            await trigger_event(f"london_close_{weekday}", msg)

        # 9. Friday Market Close (Friday at 21:00 UTC / 22:00 UTC+1)
        if weekday == 4 and hour == 21 and minute == 0:
            msg = (
                f"🛑 ━━━━━━━━━━━━━━━━━━━━ 🛑\n"
                f"   🌴 **WEEKEND MARKET CLOSE**\n"
                f"🛑 ━━━━━━━━━━━━━━━━━━━━ 🛑\n\n"
                f"⏰ **Time**: `21:00 UTC` (`22:00 WAT / UTC+1`)\n\n"
                f"🔒 **Status**:\n"
                f"  • Forex & Gold markets are now **CLOSED** for the weekend.\n"
                f"  • Crypto (`BTC/USD`) remains **OPEN 24/7**.\n"
                f"  • IQ Option OTC Blitz assets remain active.\n\n"
                f"🎉 Great job this week! Have a restful weekend and recharge for Sunday open.\n"
                f"🛑 ━━━━━━━━━━━━━━━━━━━━ 🛑"
            )
            await trigger_event("weekend_close", msg)

    async def run_loop(self):
        """Asynchronous monitoring loop checking every 20 seconds."""
        self.is_running = True
        logger.info("🌐 [SessionNotifier] Market Session Alert Loop started.")
        while self.is_running:
            try:
                await self.check_and_notify_events()
            except Exception as e:
                logger.error(f"[SessionNotifier] Error in loop: {e}")
            await asyncio.sleep(20)

    def stop(self):
        self.is_running = False
