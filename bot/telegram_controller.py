"""
bot/telegram_controller.py - Unified Telegram Bot Controller (@WalterAWbot)

Interactive command center for:
  - CallistoFx Zone Copier
  - Gold Pips Hunter Signal Copier
  - Polycarp VIP Room Blitz Options Copier (with 2-step Martingale)
  - Autonomous Gold / Multi-Instrument ICT Strategy Engine
"""

import os
import re
import sys
import time
import logging
import asyncio
import httpx
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List, Set, Tuple, Union, Callable

from telegram import Update, InlineKeyboardMarkup, InlineKeyboardButton
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, MessageHandler, ContextTypes, filters
)
from dotenv import load_dotenv

from copiers.channel_manager import ChannelManager
from strategies.ict_engine import ICTStrategyEngine, INSTRUMENT_PROFILES
from clients.forex_mcp_client import IQForexMCPClient
from clients.blitz_mcp_client import IQBlitzMCPClient
from bot.trade_ledger import TradeLedger
from bot.history_formatter import (
    DEFAULT_SYMBOL_MAP,
    build_symbol_map_from_assets,
    format_history_html,
    format_blitz_history_html,
)
from trade_database import db
from bot.keyboards import (
    main_menu_keyboard, channels_menu_keyboard,
    ict_menu_keyboard, settings_menu_keyboard, close_all_confirm_keyboard,
    persistent_reply_keyboard, history_menu_keyboard, active_setups_keyboard,
    news_menu_keyboard, crt_menu_keyboard, mission_menu_keyboard,
    individual_close_keyboard, snd_menu_keyboard, trendline_menu_keyboard,
    prop_firm_menu_keyboard, onboarding_disclaimer_keyboard, user_management_keyboard,
    investor_main_menu_keyboard, investor_reply_keyboard, admin_reply_keyboard
)
from user_manager import user_manager
from bot.session_notifier import MarketSessionNotifier
from bot.news_engine import EconomicNewsEngine
from bot.prop_firm_manager import PropFirmGuardian
from strategies.news_straddle_engine import NewsStraddleEngine
from strategies.crt_engine import CRTStrategyEngine
from strategies.snd_engine import SNDStrategyEngine, SND_INSTRUMENT_PROFILES
from strategies.trendline_engine import TrendlineStrategyEngine, TRENDLINE_PROFILES

try:
    from gsheet_logger import gsheet_logger
except ImportError:
    try:
        from utils.gsheet_logger import gsheet_logger
    except ImportError:
        gsheet_logger = None

logger = logging.getLogger("TelegramController")

class HistoryViewResult(str):
    def __new__(cls, content: str, total_pages: int = 1):
        obj = str.__new__(cls, content)
        obj.total_pages = total_pages
        return obj

    def __iter__(self):
        return iter((str(self), self.total_pages))


class TelegramTradingBot:
    def __init__(self, token: str, admin_id: int, channel_mgr: ChannelManager,
                 ict_engine: ICTStrategyEngine, forex_mcp: IQForexMCPClient, blitz_mcp: IQBlitzMCPClient,
                 lots: float = 1.0, leverage: int = 100, blitz_stake: float = 2.0,
                 straddle_engine: Optional[NewsStraddleEngine] = None,
                 crt_engine: Optional[CRTStrategyEngine] = None,
                 snd_engine: Optional[SNDStrategyEngine] = None,
                 trendline_engine: Optional[TrendlineStrategyEngine] = None):
        self.token = token
        self.admin_id = int(admin_id)
        self.channel_mgr = channel_mgr
        self.ict_engine = ict_engine
        self.forex_mcp = forex_mcp
        self.blitz_mcp = blitz_mcp
        self.session_notifier = MarketSessionNotifier(broadcast_func=self.broadcast_alert)
        self.news_engine = EconomicNewsEngine(broadcast_func=self.broadcast_alert)

        # High-Impact News Straddle Spike Engine
        self.straddle_engine = straddle_engine or NewsStraddleEngine(
            forex_mcp=self.forex_mcp,
            news_engine=self.news_engine,
            lots=lots,
            leverage=leverage
        )
        self.straddle_engine.set_notification_callback(self.broadcast_alert)

        # Candle Range Theory (CRT) Engine (Forex Majors & Crypto)
        self.crt_engine = crt_engine or CRTStrategyEngine(
            mcp_client=self.forex_mcp,
            symbols=["EURUSD", "BTCUSD"],
            account_type="training",
            lots=lots,
            leverage=leverage
        )
        self.crt_engine.set_notification_callback(self.broadcast_alert)

        # Autonomous Supply & Demand + Imbalance Engine (Config E: NZDUSD, USDJPY, AUDUSD, USDCAD)
        self.snd_engine = snd_engine or SNDStrategyEngine(
            mcp_client=self.forex_mcp,
            symbols=["NZDUSD", "USDJPY", "AUDUSD", "USDCAD"],
            account_type="training",
            lots=lots,
            leverage=leverage
        )
        self.snd_engine.set_notification_callback(self.broadcast_alert)

        # Autonomous Trendline Engine (15M 3rd Touch Bounce on Gold & NAS100)
        self.trendline_engine = trendline_engine or TrendlineStrategyEngine(
            mcp_client=self.forex_mcp,
            symbol="XAUUSD",
            account_type="training",
            lots=lots,
            leverage=leverage
        )
        self.trendline_engine.set_notification_callback(self.broadcast_alert)

        self.account_type = "training"
        self.lots = float(lots)
        self.leverage = int(leverage)
        self.blitz_stake = float(blitz_stake)
        self.is_paused = False
        self.app: Optional[Application] = None
        self._active_refresh_task: Optional[asyncio.Task] = None
        self.last_open_positions = []
        if hasattr(self.channel_mgr, "set_photo_notification_callback"):
            self.channel_mgr.set_photo_notification_callback(self.broadcast_photo)
        if hasattr(self.ict_engine, "set_photo_notification_callback"):
            self.ict_engine.set_photo_notification_callback(self.broadcast_photo)
        if hasattr(self.straddle_engine, "set_photo_notification_callback"):
            self.straddle_engine.set_photo_notification_callback(self.broadcast_photo)
        if hasattr(self.crt_engine, "set_photo_notification_callback"):
            self.crt_engine.set_photo_notification_callback(self.broadcast_photo)
        if hasattr(self.snd_engine, "set_photo_notification_callback"):
            self.snd_engine.set_photo_notification_callback(self.broadcast_photo)
        if hasattr(self.trendline_engine, "set_photo_notification_callback"):
            self.trendline_engine.set_photo_notification_callback(self.broadcast_photo)

        # Institutional 5K Prop Firm Challenge Simulator & Guardian
        self.prop_guardian = PropFirmGuardian(
            initial_capital=5000.0,
            phase=1,
            broadcast_func=self.broadcast_alert
        )
        self.prop_guardian.set_mcp_client(self.forex_mcp)
        if hasattr(self.ict_engine, "prop_guardian"):
            self.ict_engine.prop_guardian = self.prop_guardian
        if hasattr(self.crt_engine, "prop_guardian"):
            self.crt_engine.prop_guardian = self.prop_guardian
        if hasattr(self.snd_engine, "prop_guardian"):
            self.snd_engine.prop_guardian = self.prop_guardian
        if hasattr(self.trendline_engine, "prop_guardian"):
            self.trendline_engine.prop_guardian = self.prop_guardian
        if hasattr(self.straddle_engine, "prop_guardian"):
            self.straddle_engine.prop_guardian = self.prop_guardian
        if hasattr(self.channel_mgr, "copiers"):
            for copier in self.channel_mgr.copiers.values():
                setattr(copier, "prop_guardian", self.prop_guardian)

        # Persistent Trade Ledger (Addendum A)
        self.trade_ledger = TradeLedger()
        self.trade_ledger.clear_trades_before_today()
        db.clear_trades_before_today()
        for eng in (self.ict_engine, self.crt_engine, self.snd_engine, self.trendline_engine, self.straddle_engine):
            if eng:
                if hasattr(eng, "set_trade_ledger"):
                    eng.set_trade_ledger(self.trade_ledger)
                else:
                    setattr(eng, "trade_ledger", self.trade_ledger)

        if hasattr(self.channel_mgr, "copiers"):
            for copier in self.channel_mgr.copiers.values():
                setattr(copier, "trade_ledger", self.trade_ledger)

        # Performance & response-latency caches
        self._balance_id_cache: Optional[Tuple[str, int, float]] = None
        self._last_reconcile_ts: float = 0.0
        self._reconcile_cooldown: float = 15.0
        self._blitz_history_cache: Optional[Tuple[float, List[Dict[str, Any]]]] = None
        self._blitz_history_ttl: float = 20.0
        self._symbol_map_cache: Optional[Tuple[float, Dict[str, str]]] = None
        self._symbol_map_ttl: float = 3600.0

    def is_admin(self, user_id: int) -> bool:
        return int(user_id) == self.admin_id

    def update_forex_lots(self, lots: float) -> float:
        """Update lot size for ICT Engine, Straddle Engine, and Forex/CFD copiers."""
        clean_lots = max(0.01, round(float(lots), 2))
        self.lots = clean_lots
        self.ict_engine.set_lots(clean_lots)
        if self.straddle_engine:
            self.straddle_engine.set_lots(clean_lots)
        if self.crt_engine:
            self.crt_engine.set_lots(clean_lots)
        if self.snd_engine:
            self.snd_engine.set_lots(clean_lots)
        if self.trendline_engine:
            self.trendline_engine.set_lots(clean_lots)
        for c in self.channel_mgr.copiers.values():
            if hasattr(c, "set_lots"):
                c.set_lots(clean_lots)
        logger.info(f"📊 [Controller] Updated Forex/Gold lot size to: {self.lots}")
        return self.lots

    def update_forex_leverage(self, leverage: int) -> int:
        """Update leverage for ICT Engine, Straddle Engine, and Forex/CFD copiers."""
        clean_lev = int(leverage)
        self.leverage = clean_lev
        self.ict_engine.set_leverage(clean_lev)
        if self.straddle_engine:
            self.straddle_engine.set_leverage(clean_lev)
        if self.crt_engine:
            self.crt_engine.set_leverage(clean_lev)
        if self.snd_engine:
            self.snd_engine.set_leverage(clean_lev)
        for c in self.channel_mgr.copiers.values():
            if hasattr(c, "set_leverage"):
                c.set_leverage(clean_lev)
        logger.info(f"⚡ [Controller] Updated Forex/Gold leverage to: {self.leverage}x")
        return self.leverage

    def update_blitz_stake(self, stake: float) -> float:
        """Update Blitz options base stake."""
        clean_stake = max(1.0, round(float(stake), 2))
        self.blitz_stake = clean_stake
        polycarp = self.channel_mgr.get_copier("polycarpvip")
        if polycarp and hasattr(polycarp, "set_stake"):
            polycarp.set_stake(clean_stake)
        logger.info(f"⚡ [Controller] Updated Blitz base stake to: ${self.blitz_stake:.2f}")
        return self.blitz_stake

    async def broadcast_alert(self, text: str):
        """Send high-priority notification to the admin on Telegram with direct HTTP fallback, and fan out to approved users if enabled."""
        admin_delivered = False
        if self.app and getattr(self.app, "bot", None):
            try:
                await self.app.bot.send_message(
                    chat_id=self.admin_id,
                    text=text
                )
                admin_delivered = True
                first_line = text.splitlines()[0] if text else ""
                logger.info(f"📢 [Broadcast Alert] Dispatched to Admin ({self.admin_id}): {first_line}")
            except Exception as e:
                logger.warning(f"Error sending broadcast via app.bot: {e}")

        # Resilient fallback via direct Telegram HTTP API for Admin
        if not admin_delivered:
            try:
                async with httpx.AsyncClient(timeout=10.0) as client:
                    res = await client.post(
                        f"https://api.telegram.org/bot{self.token}/sendMessage",
                        json={"chat_id": self.admin_id, "text": text}
                    )
                    logger.info(f"📢 [Broadcast Alert Fallback] HTTP {res.status_code}")
            except Exception as e:
                logger.error(f"Failed to send direct Telegram broadcast alert to admin: {e}")

        # Fan out to approved investor subscribers if user alerts are enabled
        try:
            subscribers = user_manager.get_broadcast_subscribers(self.admin_id)
            if subscribers and self.app and getattr(self.app, "bot", None):
                for uid in subscribers:
                    try:
                        await self.app.bot.send_message(chat_id=uid, text=text)
                        await asyncio.sleep(0.04)
                    except Exception as err:
                        logger.debug(f"Could not deliver broadcast alert to subscriber {uid}: {err}")
        except Exception as e:
            logger.error(f"Error dispatching broadcast alert to subscribers: {e}")

    async def broadcast_photo(self, photo_bytes: bytes, caption: str = ""):
        """Send high-priority chart/image notification to the admin on Telegram, and fan out to approved users if enabled."""
        if not photo_bytes:
            await self.broadcast_alert(caption)
            return

        admin_delivered = False
        if self.app and getattr(self.app, "bot", None):
            try:
                await self.app.bot.send_photo(
                    chat_id=self.admin_id,
                    photo=photo_bytes,
                    caption=caption
                )
                admin_delivered = True
            except Exception as e:
                logger.warning(f"Error sending broadcast photo via app.bot: {e}")

        if not admin_delivered:
            # Resilient fallback via direct Telegram HTTP API multipart upload for Admin
            try:
                async with httpx.AsyncClient(timeout=20.0) as client:
                    files = {"photo": ("chart.png", photo_bytes, "image/png")}
                    data = {"chat_id": str(self.admin_id), "caption": caption}
                    await client.post(
                        f"https://api.telegram.org/bot{self.token}/sendPhoto",
                        data=data,
                        files=files
                    )
            except Exception as e:
                logger.error(f"Failed to send direct Telegram photo broadcast to admin: {e}")
                await self.broadcast_alert(caption)

        # Fan out photo to approved investor subscribers if user alerts are enabled
        try:
            subscribers = user_manager.get_broadcast_subscribers(self.admin_id)
            if subscribers and self.app and getattr(self.app, "bot", None):
                for uid in subscribers:
                    try:
                        await self.app.bot.send_photo(chat_id=uid, photo=photo_bytes, caption=caption)
                        await asyncio.sleep(0.05)
                    except Exception as err:
                        logger.debug(f"Could not deliver broadcast photo to subscriber {uid}: {err}")
        except Exception as e:
            logger.error(f"Error dispatching broadcast photo to subscribers: {e}")

    async def _fetch_dual_balances_async(self) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
        """Fetch Forex/CFD and Blitz balances concurrently off the event loop."""
        fx_fn = self.forex_mcp.get_training_balance if self.account_type == "training" else self.forex_mcp.get_real_balance
        blitz_fn = self.blitz_mcp.get_training_balance if self.account_type == "training" else self.blitz_mcp.get_real_balance
        res = await asyncio.gather(
            asyncio.to_thread(fx_fn),
            asyncio.to_thread(blitz_fn),
            return_exceptions=True
        )
        fx_bal = res[0] if not isinstance(res[0], Exception) else None
        blitz_bal = res[1] if not isinstance(res[1], Exception) else None
        return fx_bal, blitz_bal

    async def build_status_text_async(self) -> str:
        """Non-blocking status builder that fetches Forex and Blitz balances concurrently."""
        fx_bal, blitz_bal = await self._fetch_dual_balances_async()
        return self._format_status_text(fx_bal, blitz_bal)

    def build_status_text(self) -> str:
        # Fetch balances
        fx_bal = self.forex_mcp.get_training_balance() if self.account_type == "training" else self.forex_mcp.get_real_balance()
        blitz_bal = self.blitz_mcp.get_training_balance() if self.account_type == "training" else self.blitz_mcp.get_real_balance()
        return self._format_status_text(fx_bal, blitz_bal)

    def _format_status_text(self, fx_bal: Optional[Dict[str, Any]], blitz_bal: Optional[Dict[str, Any]]) -> str:
        fx_eq = fx_bal.get("equity", 0.0) if fx_bal else 0.0
        blitz_amt = blitz_bal.get("amount", 0.0) if blitz_bal else 0.0

        # ICT Engine
        ict_st = self.ict_engine.get_status()
        ict_icon = "🟢" if ict_st["enabled"] else "🔴"
        hybrid_on = ict_st.get("use_hybrid_trailing", True)
        hybrid_icon = "🟢" if hybrid_on else "🔴"
        pause_tag = " [PAUSED]" if self.is_paused else ""

        # CRT Engine
        crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
        crt_icon = "🟢" if crt_st.get("enabled") else "🔴"

        # S&D Imbalance Engine (Config E)
        snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
        snd_icon = "🟢" if snd_st.get("enabled") else "🔴"

        # Trendline Engine (15M 3rd Touch Bounce)
        tl_st = self.trendline_engine.get_status() if self.trendline_engine else {"enabled": False, "enabled_symbols": []}
        tl_icon = "🟢" if tl_st.get("enabled") else "🔴"

        # News Straddle Engine
        straddle_st = self.straddle_engine.get_status() if self.straddle_engine else {"enabled": False, "is_armed": False}
        straddle_icon = "🟢" if straddle_st.get("enabled") else "🔴"
        straddle_arm_icon = " ⚡ [ARMED - Watching Spike]" if straddle_st.get("is_armed") else ""

        text = (
            f"👑 *BreakingBad V3 — Trading Control Center*{pause_tag}\n\n"
            f"👤 *Account Mode*: `{self.account_type.upper()}`\n"
            f"💵 *Forex/CFD Equity*: `${fx_eq:.2f}`\n"
            f"⚡ *Blitz Options Balance*: `${blitz_amt:.2f}`\n\n"
            f"⚙️ *Risk & Order Sizing*:\n"
            f"  • 📊 Forex / Gold Lots: `{self.lots:.2f}` | Lev: `{self.leverage}x`\n"
            f"  • ⚡ Blitz Base Stake: `${self.blitz_stake:.2f}` *(2-Step Martingale)*\n\n"
            f"🤖 *Autonomous ICT Engine (Metals)*:\n"
            f"  • {ict_icon} Master Switch: `{'ON' if ict_st['enabled'] else 'OFF'}`\n"
            f"  • {hybrid_icon} Hybrid Profit Lock: `{'ON' if hybrid_on else 'OFF'}`\n"
            f"  • 🎯 Active Assets: `{', '.join(ict_st['enabled_symbols']) if ict_st['enabled_symbols'] else 'None'}`\n"
            f"  • 📊 Lots: `{ict_st['lots']:.2f}` | Lev: `{ict_st['leverage']}x`\n"
            f"  • ⚖️ Risk/Reward: `1:{ict_st['rr_ratio']:.1f}`\n\n"
            f"🕯️ *Autonomous CRT Engine (Forex & Crypto)*:\n"
            f"  • {crt_icon} Master Switch: `{'ON' if crt_st.get('enabled') else 'OFF'}`\n"
            f"  • 🎯 Active Assets: `{', '.join(crt_st.get('enabled_symbols', [])) if crt_st.get('enabled_symbols') else 'None'}`\n"
            f"  • 📊 Lots: `{crt_st.get('lots', self.lots):.2f}` | Lev: `{crt_st.get('leverage', self.leverage)}x`\n\n"
            f"📐 *Autonomous Trendline Engine (15M Bounce)*:\n"
            f"  • {tl_icon} Master Switch: `{'ON' if tl_st.get('enabled') else 'OFF'}`\n"
            f"  • 🎯 Active Assets: `{', '.join(tl_st.get('enabled_symbols', [])) if tl_st.get('enabled_symbols') else 'None'}`\n"
            f"  • 📊 Lots: `{tl_st.get('lots', self.lots):.2f}` (Dual Ticket) | Lev: `{tl_st.get('leverage', self.leverage)}x`\n"
            f"  • ⚖️ Structure: `Leg 1 (+1.0R) ➔ Leg 2 (+2.5R Runner with BE lock)`\n\n"
            f"🏛️ *Autonomous S&D Engine (Config E Forex)*:\n"
            f"  • {snd_icon} Master Switch: `{'ON' if snd_st.get('enabled') else 'OFF'}`\n"
            f"  • 🎯 Active Pairs: `{', '.join(snd_st.get('enabled_symbols', [])) if snd_st.get('enabled_symbols') else 'None'}`\n"
            f"  • 📊 Lots: `{snd_st.get('lots', self.lots):.2f}` | Lev: `{snd_st.get('leverage', self.leverage)}x`\n"
            f"  • ⚖️ Target R:R: `1:2.0` | Breakeven: `+1.0R`\n\n"
            f"⚡ *News Straddle Spike Engine*:\n"
            f"  • {straddle_icon} Status: `{'ON' if straddle_st.get('enabled') else 'OFF'}`{straddle_arm_icon}\n"
            f"  • 🎯 Asset: `XAUUSD (Gold)` | Lots: `{self.straddle_engine.lots if self.straddle_engine else self.lots:.2f}`\n\n"
            f"🏆 *Prop Firm Challenge ($5,000 Sim)*:\n"
            f"  • {'🟢' if self.prop_guardian.is_enabled else '🔴'} Guard: `{'ON' if self.prop_guardian.is_enabled else 'OFF'}` | Phase: `{self.prop_guardian.phase}`\n"
            f"  • Status: `{self.prop_guardian.status}` | Equity: `${self.prop_guardian.current_balance:.2f}`\n\n"
            f"🕒 Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return text

    def build_ict_dashboard(self) -> str:
        """Renders real-time control view for the Autonomous ICT Strategy Engine."""
        ict_st = self.ict_engine.get_status()
        enabled = ict_st.get("enabled", False)
        hybrid_on = ict_st.get("use_hybrid_trailing", True)
        icon = "🟢" if enabled else "🔴"
        
        active_syms = ", ".join(ict_st.get("enabled_symbols", [])) or "None"
        lots = ict_st.get("lots", 1.0)
        lev = ict_st.get("leverage", 100)
        rr = ict_st.get("rr_ratio", 2.2)
        
        active_trades = ict_st.get("active_trades", {})
        trade_lines = []
        if active_trades:
            for s, t in active_trades.items():
                if t:
                    side = t.get("side", "")
                    entry = t.get("entry_price", 0.0)
                    sl = t.get("current_sl", 0.0)
                    tp = t.get("tp", 0.0)
                    stg = t.get("trailing_stage", 0)
                    trade_lines.append(f"  • *{s}* ({side}) @ {entry} | SL: {sl} | TP: {tp} | Stage: {stg}")
        
        trades_section = ("\n📊 *Active Positions*:\n" + "\n".join(trade_lines)) if trade_lines else "\n📊 *Active Positions*: None"

        trailing_mode = (
            "🟢 *Hybrid Trailing: ON*\n"
            "   _(Locks profit at absolute pip milestones OR R-multiples: e.g. +80p locked @ +180p run-up on Gold)_"
            if hybrid_on else
            "🔴 *Hybrid Trailing: OFF*\n"
            "   _(Pure classic R-multiple scaling: 0.5R, 1.0R, 1.5R, 2.0R only)_"
        )

        return (
            f"🤖 *Autonomous Gold & Forex ICT Strategy Engine*\n\n"
            f"• Master Switch: {icon} `{'ON' if enabled else 'OFF'}`\n"
            f"• Active Assets: `{active_syms}`\n"
            f"• Lots: `{lots:.2f}` | Leverage: `{lev}x` | Target RR: `1:{rr:.1f}`\n\n"
            f"🛡️ *Profit Protection Mode*:\n"
            f"{trailing_mode}\n"
            f"{trades_section}\n\n"
            f"Tap buttons below to toggle assets, settings, or trailing mode:"
        )

    def build_mission_text(self) -> str:
        """Renders the comprehensive, beautifully styled Mission and Strategy Description card."""
        return (
            "🎯 *BREAKINGBAD V3 — MISSION & STRATEGY ARCHITECTURE*\n\n"
            "🏛 *Our Prime Mission:*\n"
            "To execute a systematic, disciplined, and multi-asset trading edge with algorithmic precision, strict risk-to-reward asymmetry, and 100% emotional detachment.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "💎 *1. ICT Institutional Engine (Metals Specialist)*\n"
            "• *Focus Assets*: `Gold (XAUUSD)` & `Silver (XAGUSD)`\n"
            "• *Core Logic*: Multi-Timeframe Institutional Order Flow (1H Macro Bias ➔ 5M/15M Precision). Identifies liquidity sweeps at external fractal swing points, confirms Change in State of Delivery (CISD), and enters on Fair Value Gap (FVG) mitigation retests.\n"
            "• *Risk Model*: 1:2.0 – 1:4.0 Risk:Reward with dynamic Breakeven safety ratchets.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🕯️ *2. CRT Engine (Candle Range Theory)*\n"
            "• *Focus Assets*: `EUR/USD`, `GBP/USD` & `Bitcoin (BTCUSD)`\n"
            "• *Core Logic*: Exploits time-based candle ranges. Utilizes the **Asian Judas Protocol** (00:00–06:00 UTC) during London Open (07:00–10:00 UTC) and H1 candle expansions, catching manipulation wicks before the true body expansion unfolds.\n"
            "• *Targets*: 50% Equilibrium Midpoint Breakeven Lock + Opposite Boundary Full Expansion.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "🏛️ *3. S&D Imbalance Engine (Config E Forex Specialist)*\n"
            "• *Focus Assets*: `NZD/USD (+27.0R)`, `USD/JPY (+17.0R)`, `AUD/USD (+15.0R)`, `USD/CAD (+7.0R)`\n"
            "• *Core Logic*: Identifies institutional Drop-Base-Rally (Demand) & Rally-Base-Drop (Supply) zones on 15M candles with displacement bodies >= 1.2x ATR leaving confirmed Fair Value Gaps (BISI/SIBI), filtered by macro EMA100.\n"
            "• *Risk Model*: 1:2.0 Fixed Target Payout with Dynamic Breakeven Ratchet locked at +1.0R displacement.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚡ *4. High-Impact News Straddle Spike Engine*\n"
            "• *Focus Assets*: `US Non-Farm Payrolls (NFP)`, `CPI`, `FOMC Decisions`\n"
            "• *Core Logic*: Calculates the tight pre-news consolidation range 60 seconds prior to release and places dual breakout triggers (±$1.50) to capture explosive multi-dollar volatility spikes.\n"
            "• *Targets*: **+$15.00** Expansion Take Profit with rapid **+$4.00** Breakeven lock.\n\n"
            "━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n"
            "⚡ *5. Institutional Gold Zones & Blitz Options*\n"
            "• *Gold Confirmation Zones*: Quantitative confirmation zones and liquidity sweeps on XAUUSD.\n"
            "• *High-Frequency Blitz Options*: Algorithmic short-term volatility options with a controlled 2-step Martingale recovery buffer.\n\n"
            "🛡️ *Golden Rule*: Capital preservation first. Never trade without invalidation."
        )

    # ==========================================
    # Command Handlers
    # ==========================================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return

        allowed, status, udata = user_manager.check_access(user.id, user.username, user.first_name)

        if status == "suspended":
            await update.message.reply_text(
                "🚫 *Access Suspended*\n"
                "Your access to BreakingBad V3 has been suspended by the administrator.\n"
                "Please contact the administrator for further details.",
                parse_mode="Markdown"
            )
            return

        if status == "pending":
            await update.message.reply_text(
                "⏳ *Registration Pending*\n"
                "Your account is registered and awaiting administrator approval.\n"
                "You will receive access once approved by the administrator.",
                parse_mode="Markdown"
            )
            return

        if status == "needs_onboarding":
            welcome_msg = (
                f"👋 *Welcome to BreakingBad V3, {user.first_name}!* 🧪📈\n\n"
                "BreakingBad V3 is an AI-augmented automated institutional trading system for IQ Option "
                "(Blitz Options, Forex, Indices & Gold).\n\n"
                "━━━━━━━━━━━━━━━━━━━━\n"
                "🏢 *SHARED DEMO ACCOUNT ENVIRONMENT*\n"
                "• This environment operates in **Shared Demo Mode** using the owner's IQ Option Demo connection.\n"
                "• All trades, setups, and signals are executed strictly inside this simulated practice environment.\n"
                "• You do **not** need to provide your own IQ Option credentials.\n"
                "• Real-money account execution is strictly disabled.\n\n"
                "🎯 *Features Available to You:*\n"
                "• Real-time bot engine status & market session monitoring\n"
                "• Shared demo account balance & equity tracking\n"
                "• Live trade signals, active setups & economic calendar\n"
                "• Verified historical trade performance & statistics\n"
                "• Institutional strategies (Gold ICT, CRT, S&D, Trendlines)\n\n"
                "⚠️ *Risk Disclaimer:*\n"
                "Trading financial markets and binary options involves substantial risk of loss. "
                "Demonstration results are simulated and for educational/evaluative purposes only.\n\n"
                "Please tap *'I Understand & Accept'* below to complete registration and enter the demo dashboard."
            )
            await update.message.reply_text(
                welcome_msg,
                parse_mode="Markdown",
                reply_markup=onboarding_disclaimer_keyboard()
            )
            return

        # User is approved or admin
        is_admin = self.is_admin(user.id)
        if is_admin:
            await update.message.reply_text(
                "👑 *BreakingBad V3 Bot is Online & Ready!*\nUse the buttons below for administrator control:",
                reply_markup=admin_reply_keyboard(),
                parse_mode="Markdown"
            )
            text = self.build_status_text()
            await update.message.reply_text(
                text=text,
                reply_markup=main_menu_keyboard(is_admin=True),
                parse_mode="Markdown"
            )
        else:
            await update.message.reply_text(
                f"👋 *Welcome back, {user.first_name}!* 🧪\n"
                "BreakingBad V3 Shared Demo Environment is active.\n"
                "Use the reply buttons below to monitor performance, check balance, or view trading history:",
                reply_markup=investor_reply_keyboard(),
                parse_mode="Markdown"
            )
            text = self.build_status_text()
            await update.message.reply_text(
                text=text,
                reply_markup=investor_main_menu_keyboard(),
                parse_mode="Markdown"
            )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start to begin onboarding.")
            return

        text = await self.build_status_text_async()
        kb = main_menu_keyboard(is_admin=True) if self.is_admin(user.id) else investor_main_menu_keyboard()
        await update.message.reply_text(text=text, reply_markup=kb, parse_mode="Markdown")

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start to begin onboarding.")
            return

        fx_bal, blitz_bal = await self._fetch_dual_balances_async()
        fx_eq = fx_bal.get("equity", 0.0) if fx_bal else 0.0
        fx_avail = fx_bal.get("free_margin", fx_bal.get("available", fx_bal.get("balance", 0.0))) if fx_bal else 0.0
        blitz_amt = blitz_bal.get("amount", 0.0) if blitz_bal else 0.0

        msg = (
            f"💰 *Shared Demo Account Balances*\n\n"
            f"💼 *Mode*: 🟡 PRACTICE (Training / Shared Demo)\n\n"
            f"📈 *Forex / CFD Account*:\n"
            f"  • Equity: `${fx_eq:,.2f}`\n"
            f"  • Available: `${fx_avail:,.2f}`\n\n"
            f"⚡ *Blitz Options Account*:\n"
            f"  • Balance: `${blitz_amt:,.2f}`\n\n"
            f"🛡️ *Demo Environment*: All users share this single demo account. Real execution is strictly disabled."
        )
        reply_kb = admin_reply_keyboard() if self.is_admin(user.id) else investor_reply_keyboard()
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=reply_kb)

    async def cmd_pause(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        self.is_paused = True
        self.ict_engine.enabled = False
        for c in self.channel_mgr.copiers.values():
            c.is_enabled = False
        logger.info("⏸️ [Controller] Bot PAUSED across all engines.")
        await update.message.reply_text(
            "⏸️ *Bot PAUSED*\nAll autonomous ICT scanning and signal copier executions are paused.",
            parse_mode="Markdown",
            reply_markup=persistent_reply_keyboard()
        )

    async def cmd_resume(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        self.is_paused = False
        self.ict_engine.enabled = True
        for c in self.channel_mgr.copiers.values():
            c.is_enabled = True
        logger.info("▶️ [Controller] Bot RESUMED across all engines.")
        await update.message.reply_text(
            "▶️ *Bot RESUMED*\nAll autonomous ICT scanning and signal copier executions are now active.",
            parse_mode="Markdown",
            reply_markup=persistent_reply_keyboard()
        )

    async def cmd_channels(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        copiers = self.channel_mgr.list_copiers()
        await update.message.reply_text(
            "📡 *Channel Copiers Management*\nToggle signal monitoring for individual channels:",
            reply_markup=channels_menu_keyboard(copiers),
            parse_mode="Markdown"
        )

    def get_live_price(self, symbol_or_id: Any) -> float:
        """Helper to dynamically fetch real-time market mid price for any asset or symbol."""
        try:
            sym = str(symbol_or_id).upper().replace("/", "").replace("-", "")
            if sym in INSTRUMENT_PROFILES:
                p = self.ict_engine.get_market_price(sym)
                if p and p.get("mid", 0) > 0:
                    return float(p["mid"])
            if sym in ["74", "GOLD", "XAU", "XAUUSD"]:
                p = self.ict_engine.get_market_price("XAUUSD")
                if p and p.get("mid", 0) > 0:
                    return float(p["mid"])
            if sym in ["1487", "54", "SILVER", "XAG", "XAGUSD"]:
                p = self.ict_engine.get_market_price("XAGUSD")
                if p and p.get("mid", 0) > 0:
                    return float(p["mid"])
            if sym in ["816", "BTC", "BTCUSD"]:
                p = self.ict_engine.get_market_price("BTCUSD")
                if p and p.get("mid", 0) > 0:
                    return float(p["mid"])
            aid = None
            try:
                aid = int(symbol_or_id)
            except Exception:
                for s, prof in INSTRUMENT_PROFILES.items():
                    if s == sym:
                        aid = prof.get("asset_id")
                        break
            if aid:
                candles = self.forex_mcp.get_candles(asset_id=aid, size=60, count=1)
                if candles and len(candles) > 0:
                    return float(candles[-1].get("close", 0.0))
        except Exception:
            pass
        return 0.0

    @staticmethod
    def calculate_pips(symbol_or_name: str, open_px: float, cur_px: float, side: str) -> float:
        """Calculates pip movement relative to entry (positive for gain, negative for loss)."""
        if open_px <= 0 or cur_px <= 0:
            return 0.0
        diff = (cur_px - open_px) if side.upper() in ["BUY", "LONG"] else (open_px - cur_px)
        s = str(symbol_or_name).upper().replace("/", "").replace("-", "")
        if "XAU" in s or "GOLD" in s or s == "74":
            # Gold: 1 pip = $0.10 (e.g. $1.00 move = 10 pips)
            return round(diff * 10.0, 1)
        elif "XAG" in s or "SILVER" in s or s in ["1487", "54"]:
            # Silver: 1 pip = $0.01 (1 cent)
            return round(diff * 100.0, 1)
        elif "JPY" in s:
            return round(diff * 100.0, 1)
        elif any(k in s for k in ["BTC", "ETH", "US100", "US500", "OIL", "WTI", "BRENT"]):
            return round(diff, 1)
        else:
            # Standard Forex 4/5 decimal pairs (EURUSD, GBPUSD): 1 pip = 0.0001
            return round(diff * 10000.0, 1)

    @staticmethod
    def get_asset_multiplier(symbol_or_id: Any) -> float:
        """Returns standard contract multiplier per lot for PnL calculations."""
        s = str(symbol_or_id).upper().replace("/", "").replace("-", "")
        if "XAU" in s or "GOLD" in s or s == "74":
            return 1.0
        elif "XAG" in s or "SILVER" in s or s in ["1487", "54"]:
            return 50.0
        elif "BTC" in s or s == "816":
            return 1.0
        elif any(k in s for k in ["EURUSD", "GBPUSD", "AUDUSD", "NZDUSD", "USDCHF"]):
            return 100000.0
        return 1.0

    def resolve_asset_name(self, asset_id: Any) -> str:
        """Helper to resolve ticker symbol from asset ID."""
        ASSET_MAP = {
            1: "EURUSD", 2: "GBPUSD", 4: "EURJPY", 5: "GBPJPY",
            6: "USDJPY", 8: "AUDUSD", 10: "EURGBP",
            54: "Silver (XAGUSD)", 74: "Gold (XAUUSD)",
            99: "AUDUSD", 100: "USDCAD", 105: "NZDUSD", 108: "USDCHF",
            212: "EURCAD", 816: "Bitcoin (BTCUSD)", 824: "Ethereum (ETHUSD)",
            836: "Ripple (XRPUSD)", 886: "Solana (SOLUSD)",
            946: "EURAUD", 951: "EURNZD", 955: "EURCHF",
            969: "Crude Oil (Brent)", 971: "Crude Oil (WTI)",
            1011: "GBPAUD", 1470: "US500", 1471: "US100",
            1487: "Silver (XAGUSD)", 1979: "Bitcoin Cash (BCH)",
        }
        try:
            aid = int(asset_id)
            if aid in ASSET_MAP:
                return ASSET_MAP[aid]
            for sym, prof in INSTRUMENT_PROFILES.items():
                if prof.get("asset_id") == aid:
                    return f"{sym}"
            for sym, prof in SND_INSTRUMENT_PROFILES.items():
                if prof.get("asset_id") == aid:
                    return f"{sym}"
            for sym, prof in TRENDLINE_PROFILES.items():
                if prof.get("asset_id") == aid:
                    return prof.get("name", sym)
            return f"Asset #{aid}"
        except Exception:
            return str(asset_id)

    def build_active_setups_view(self) -> str:
        try:
            now_str = datetime.now().strftime("%H:%M:%S")

            # 1. Open Marginal CFD Positions on IQ Option Broker
            bid = self.get_active_balance_id()
            open_cfd = []
            if bid:
                try:
                    open_cfd = self.forex_mcp.list_positions(balance_id=bid) or []
                except Exception as e:
                    logger.debug(f"[ActiveSetups] Error listing broker positions: {e}")

            parsed_positions = []
            if open_cfd:
                cfd_lines = []
                for p in open_cfd:
                    pos_id = p.get("position_id") or p.get("id", "N/A")
                    asset_id = p.get("asset_id")
                    sym = self.resolve_asset_name(asset_id)
                    side = str(p.get("side") or p.get("type", "BUY")).upper()
                    if side == "LONG": side = "BUY"
                    if side == "SHORT": side = "SELL"
                    lots = float(p.get("lots") or p.get("count") or 1.0)
                    open_px = float(p.get("open_price") or p.get("open_quote", 0.0))

                    # Live Current Price
                    cur_px = float(p.get("current_price") or p.get("close_price") or p.get("price", 0.0))
                    if cur_px <= 0:
                        cur_px = self.get_live_price(sym)

                    # PnL Resolution: IQ Option Marginal CFD uses expected_pnl for open floating PnL
                    pnl = None
                    if p.get("expected_pnl") is not None:
                        try:
                            pnl = float(p.get("expected_pnl"))
                        except (ValueError, TypeError):
                            pass
                    if pnl is None:
                        for k in ["pnl", "profit", "isolated_pnl_net", "net_profit"]:
                            if p.get(k) is not None:
                                try:
                                    pnl = float(p.get(k))
                                    break
                                except (ValueError, TypeError):
                                    pass
                    if pnl is None:
                        if cur_px > 0 and open_px > 0:
                            diff = (cur_px - open_px) if side == "BUY" else (open_px - cur_px)
                            multiplier = self.get_asset_multiplier(sym or asset_id)
                            pnl = diff * multiplier * lots
                        else:
                            pnl = 0.0

                    parsed_positions.append({
                        "position_id": pos_id,
                        "asset_id": asset_id,
                        "symbol": sym,
                        "side": side,
                        "lots": lots,
                        "open_price": open_px,
                        "current_price": cur_px,
                        "pnl": pnl
                    })

                    pnl_sign = "+" if pnl > 0 else ("" if pnl == 0 else "-")
                    pnl_abs = abs(pnl)

                    # Pips calculation relative to entry
                    pips_str = ""
                    if cur_px > 0 and open_px > 0:
                        pips = self.calculate_pips(sym or str(asset_id), open_px, cur_px, side)
                        pips_sign = "+" if pips > 0 else ("" if pips == 0 else "-")
                        pips_str = f" ({pips_sign}{abs(pips):.1f} pips)"

                    # Stop loss & take profit prices
                    sl_raw = p.get("stop_lose_price") or p.get("stop_loss") or p.get("sl")
                    tp_raw = p.get("take_profit_price") or p.get("take_profit") or p.get("tp")
                    sl_str = f"${float(sl_raw):.2f}" if sl_raw is not None and float(sl_raw) > 0 else "None"
                    tp_str = f"${float(tp_raw):.2f}" if tp_raw is not None and float(tp_raw) > 0 else "None"

                    # Detect trade origin
                    origin_tag = ""
                    for c_name, c_inst in self.channel_mgr.copiers.items():
                        if hasattr(c_inst, "open_positions"):
                            for oid, opos in c_inst.open_positions.items():
                                if opos.get("position_id") == pos_id or oid == pos_id:
                                    if "callisto" in c_name.lower(): origin_tag = " [Gold Zone]"
                                    elif "gold" in c_name.lower(): origin_tag = " [Gold Scalp]"
                                    elif "gsociety" in c_name.lower() or "society" in c_name.lower(): origin_tag = " [Gold Breakout]"
                                    elif "king" in c_name.lower(): origin_tag = " [Gold Swing]"
                                    else: origin_tag = " [Quant Setup]"
                                    break
                        if origin_tag:
                            break

                    if not origin_tag:
                        ict_trades = getattr(self.ict_engine, "active_trades", {})
                        for s_ict, t_ict in ict_trades.items():
                            if t_ict and (t_ict.get("position_id") == pos_id or t_ict.get("order_id") == pos_id):
                                origin_tag = " [ICT Engine]"
                                break

                    price_str = f" ➔ Live: ${cur_px:.2f}" if cur_px > 0 else ""
                    entry_str = f"${open_px:.2f}" if open_px > 0 else "N/A"
                    cfd_lines.append(
                        f"  • *{sym}* {side} ({lots:.2f} Lots){origin_tag}\n"
                        f"    Entry: {entry_str}{price_str}\n"
                        f"    PnL: {pnl_sign}${pnl_abs:.2f}{pips_str} | SL: {sl_str} | TP: {tp_str} [#{pos_id}]"
                    )
                self.last_open_positions = parsed_positions
                cfd_txt = "\n".join(cfd_lines)
            else:
                self.last_open_positions = []
                cfd_txt = "  • No open Marginal CFD positions on broker."

            # 2. ICT Active Autonomous Positions
            ict_trades = getattr(self.ict_engine, "active_trades", {})
            active_pos = []
            for s, t in ict_trades.items():
                if not t:
                    continue
                live_p = self.get_live_price(s)
                open_p = float(t.get("entry_price", 0.0))
                pnl_str = ""
                side_t = str(t.get("side", "BUY")).upper()
                lots_t = float(t.get("lots", 1.0))
                if live_p > 0 and open_p > 0:
                    diff = (live_p - open_p) if side_t in ["BUY", "LONG"] else (open_p - live_p)
                    mult = self.get_asset_multiplier(s)
                    pnl_calc = diff * mult * lots_t
                    pnl_sign = "+" if pnl_calc > 0 else ("" if pnl_calc == 0 else "-")
                    pips = self.calculate_pips(s, open_p, live_p, side_t)
                    pips_sign = "+" if pips > 0 else ("" if pips == 0 else "-")
                    pips_str = f" ({pips_sign}{abs(pips):.1f} pips)"
                    pnl_str = f" | PnL: {pnl_sign}${abs(pnl_calc):.2f}{pips_str}"

                stage = t.get("trailing_stage", 0)
                stage_str = f" (Stage {stage})" if stage > 0 else ""
                p_str = f" ➔ Live: ${live_p:.2f}" if live_p > 0 else ""
                sl_t = t.get("current_sl")
                tp_t = t.get("tp")
                sl_str = f"${float(sl_t):.2f}" if sl_t is not None else "None"
                tp_str = f"${float(tp_t):.2f}" if tp_t is not None else "None"
                active_pos.append(
                    f"  • *{s}* {side_t}{stage_str}\n"
                    f"    Entry: ${open_p:.2f}{p_str}{pnl_str}\n"
                    f"    SL: {sl_str} | TP: {tp_str}"
                )
            if active_pos:
                ict_trade_txt = "\n".join(active_pos)
            else:
                ict_trade_txt = "  • No active ICT positions currently running."

            # 3. ICT Pending FVGs
            ict_fvgs = getattr(self.ict_engine, "pending_fvgs", {})
            active_fvgs = []
            for s, f in ict_fvgs.items():
                if not f:
                    continue
                live_p = self.get_live_price(s)
                p_str = f" | Live: ${live_p:.2f}" if live_p > 0 else ""
                dist_str = f" (Dist: ${abs(live_p - f['fvg_low']):.2f})" if live_p > 0 else ""
                sl_val = float(f.get('sl', 0.0))
                sl_str = f"${sl_val:.2f}" if sl_val > 0 else "None"
                active_fvgs.append(f"  • *{s}* {f['side']} [{f['fvg_low']:.2f} – {f['fvg_high']:.2f}]{p_str}{dist_str} (SL: {sl_str})")
            if active_fvgs:
                fvg_txt = "\n".join(active_fvgs)
            else:
                fvg_txt = "  • No pending FVG retests waiting."

            # 4. Institutional Confirmation Zones (Gold)
            c_copier = self.channel_mgr.get_copier("callistofx")
            zones = getattr(c_copier, "active_zones", {}) if c_copier else {}
            if zones:
                zones_lines = []
                for s, z in zones.items():
                    side_label = "BUY ZONE" if str(s).upper() in ("BUY", "CALL", "LONG") else ("SELL ZONE" if str(s).upper() in ("SELL", "SHORT") else str(s))
                    live_p = self.get_live_price("XAUUSD")
                    p_str = f" | Live: ${live_p:.2f}" if live_p > 0 else ""
                    target_str = f" (Target: {z.get('target'):.2f})" if z.get("target") else ""
                    zones_lines.append(f"  • *{side_label}*: {z['zone_low']:.2f} – {z['zone_high']:.2f}{p_str}{target_str}")
                zones_txt = "\n".join(zones_lines)
            else:
                zones_txt = "  • No active confirmation zones currently watching."

            # 5. Active Blitz Option Trades
            p_copier = self.channel_mgr.get_copier("polycarpvip")
            blitz_trades = getattr(p_copier, "open_trades", {}) if p_copier else {}
            if blitz_trades:
                b_lines = []
                for pid, t in blitz_trades.items():
                    pair = t.get("pair", "OTC")
                    side = str(t.get("direction", "CALL")).upper()
                    amt = float(t.get("amount", 2.0))
                    op_px = t.get("open_price")
                    px_str = f" @ {op_px}" if op_px else ""
                    b_lines.append(f"  • *{pair}* {side}{px_str} (${amt:.2f}) [Pos #{pid}]")
                blitz_txt = "\n".join(b_lines)
            else:
                blitz_txt = "  • No active Blitz option trades."

            text = (
                f"📋 *Active Watchers, Zones & Live Setups* 🔴\n\n"
                f"👤 *Account*: {self.account_type.upper()} | 🕒 {now_str}\n\n"
                f"💼 *Open Broker CFD Positions*:\n{cfd_txt}\n\n"
                f"🎯 *ICT Autonomous Setups*:\n{ict_trade_txt}\n\n"
                f"🔥 *ICT Pending FVGs*:\n{fvg_txt}\n\n"
                f"📍 *Active Buy/Sell Zones (Gold)*:\n{zones_txt}\n\n"
                f"⚡ *Active Blitz Option Trades*:\n{blitz_txt}\n\n"
                f"📡 _Live streaming price & PnL updates every 4s..._"
            )
            return text
        except Exception as e:
            logger.error(f"[ActiveSetups] Error building view: {e}")
            return f"📋 *Active Watchers & Setups*\n\nError loading active setups: {e}"

    def start_active_view_refresh_task(self, chat_id: int, message_id: int):
        """Starts or replaces an in-place background live refresh task for active setups."""
        self.stop_active_view_refresh_task()
        self._active_refresh_task = asyncio.create_task(self._live_refresh_loop(chat_id, message_id))

    def stop_active_view_refresh_task(self):
        """Safely stops any background live refresh task."""
        if self._active_refresh_task and not self._active_refresh_task.done():
            self._active_refresh_task.cancel()
            self._active_refresh_task = None

    async def _live_refresh_loop(self, chat_id: int, message_id: int):
        """Auto-refreshes the Active Setups message every 4 seconds in-place."""
        last_text = ""
        # Stream live updates for up to 45 ticks (3 minutes) before letting user manually refresh
        for _ in range(45):
            try:
                await asyncio.sleep(4)
                new_text = self.build_active_setups_view()
                if new_text != last_text and self.app and getattr(self.app, "bot", None):
                    last_text = new_text
                    try:
                        await self.app.bot.edit_message_text(
                            chat_id=chat_id,
                            message_id=message_id,
                            text=new_text,
                            parse_mode="Markdown",
                            reply_markup=active_setups_keyboard(is_live=True, open_positions=self.last_open_positions)
                        )
                    except Exception as e:
                        err_s = str(e)
                        if "Message is not modified" in err_s:
                            pass
                        elif "Message to edit not found" in err_s or "bot was blocked" in err_s or "Chat not found" in err_s:
                            break
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.debug(f"[ActiveSetups] Live refresh tick error: {e}")
                await asyncio.sleep(4)

    async def cmd_active_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        text = self.build_active_setups_view()
        kb = active_setups_keyboard(is_live=True, open_positions=self.last_open_positions)
        try:
            sent_msg = await update.message.reply_text(
                text,
                parse_mode="Markdown",
                reply_markup=kb
            )
            if sent_msg and self.is_admin(user.id):
                self.start_active_view_refresh_task(update.effective_chat.id, sent_msg.message_id)
        except Exception:
            sent_msg = await update.message.reply_text(
                text,
                reply_markup=kb
            )
            if sent_msg and self.is_admin(user.id):
                self.start_active_view_refresh_task(update.effective_chat.id, sent_msg.message_id)

    def get_active_balance_id(self) -> Optional[int]:
        now = time.time()
        if self._balance_id_cache is not None:
            c_type, c_bid, c_ts = self._balance_id_cache
            if c_type == self.account_type and (now - c_ts) < 60.0:
                return c_bid

        try:
            bal = self.forex_mcp.get_real_balance() if self.account_type == "regular" else self.forex_mcp.get_training_balance()
            if bal:
                bid = bal.get("balance_id") or bal.get("id")
                if bid:
                    self._balance_id_cache = (self.account_type, int(bid), now)
                    return int(bid)
        except Exception as e:
            logger.debug(f"[ActiveBalance] Error checking active balance: {e}")
        if self.ict_engine.balance_id:
            return self.ict_engine.balance_id
        return None

    def get_cached_symbol_map(self) -> Dict[str, str]:
        """Return cached {asset_id: symbol} map built from broker metadata & engine profiles."""
        now = time.time()
        if self._symbol_map_cache is not None:
            ts, cached_map = self._symbol_map_cache
            if (now - ts) < self._symbol_map_ttl and cached_map:
                return cached_map

        sym_map: Dict[str, str] = dict(DEFAULT_SYMBOL_MAP)
        for prof_dict in (INSTRUMENT_PROFILES, SND_INSTRUMENT_PROFILES, TRENDLINE_PROFILES):
            for sym, prof in prof_dict.items():
                aid = prof.get("asset_id")
                if aid is not None:
                    sym_map[str(aid)] = sym

        try:
            assets: List[Dict[str, Any]] = []
            if getattr(self.forex_mcp, "asset_cache", None):
                seen = set()
                for item in self.forex_mcp.asset_cache.values():
                    aid = item.get("asset_id")
                    if aid is not None and aid not in seen:
                        seen.add(aid)
                        assets.append(item)
            elif hasattr(self.forex_mcp, "list_assets"):
                assets = self.forex_mcp.list_assets() or []
            sym_map.update(build_symbol_map_from_assets(assets))
        except Exception as e:
            logger.debug(f"[SymbolMap] Broker asset map refresh skipped: {e}")

        self._symbol_map_cache = (now, sym_map)
        return sym_map

    def _sync_ledger_and_blitz(self, category: str, force_sync: bool = False) -> Tuple[str, List[Dict[str, Any]], Dict[str, str]]:
        """Perform throttled broker reconciliation, Blitz fetch, and symbol map lookup."""
        now = time.time()
        sync_note = ""
        cat_lower = (category or "all").lower()

        if force_sync or (now - self._last_reconcile_ts) >= self._reconcile_cooldown:
            try:
                bid = self.get_active_balance_id()
                if bid and hasattr(self, "trade_ledger") and self.trade_ledger:
                    recon = self.trade_ledger.reconcile_with_broker(self.forex_mcp, balance_id=bid, max_pages=2)
                    self._last_reconcile_ts = time.time()
                    if recon.get("partial_sync"):
                        sync_note = "Broker sync partial — showing cached ledger."
            except Exception as e:
                logger.warning(f"[History] Broker reconcile error: {e}")
                sync_note = "Broker sync unavailable — showing ledger."

        raw_blitz: List[Dict[str, Any]] = []
        if cat_lower in ("all", "blitz"):
            limit = 50 if cat_lower == "blitz" else 20
            if not force_sync and self._blitz_history_cache is not None:
                b_ts, b_data = self._blitz_history_cache
                if (now - b_ts) < self._blitz_history_ttl:
                    raw_blitz = b_data[:limit]
            if not raw_blitz and (force_sync or self._blitz_history_cache is None or (now - self._blitz_history_cache[0]) >= self._blitz_history_ttl):
                try:
                    fetched = self.blitz_mcp.get_trade_history(limit=50, use_cache=not force_sync) or []
                    self._blitz_history_cache = (time.time(), fetched)
                    raw_blitz = fetched[:limit]
                except Exception as e:
                    logger.warning(f"[History] Error fetching Blitz history: {e}")

        symbol_map = self.get_cached_symbol_map()
        return sync_note, raw_blitz, symbol_map

    def _render_history_from_data(
        self,
        category: str,
        period: str,
        page: int,
        page_size: int,
        sync_note: str,
        raw_blitz: List[Dict[str, Any]],
        symbol_map: Dict[str, str],
    ) -> HistoryViewResult:
        """Render formatted HTML history view from local ledger + pre-fetched metadata."""
        cat_lower = (category or "all").lower()
        per_lower = (period or "all").lower()
        cur_page = max(1, page)

        # 2. Blitz-only view
        if cat_lower == "blitz":
            blitz_trades = []
            cleared_at = self.trade_ledger.get_cleared_at() if hasattr(self, "trade_ledger") and self.trade_ledger else None
            for t in raw_blitz:
                pos_id = t.get("position_id") or t.get("id", "N/A")
                asset = t.get("asset_name") or t.get("active") or f"ID{t.get('asset_id', '')}"
                res_str = str(t.get("result", "")).lower()
                profit = float(t.get("profit", 0.0))
                stake = float(t.get("amount") or t.get("invest") or 0.0)
                direction = str(t.get("direction") or t.get("type", "CALL")).upper()
                is_win = (res_str == "win" or profit > 0)
                is_equal = (res_str == "equal" or (profit == 0 and res_str not in ["loss", "loose"]))

                ts_raw = t.get("close_time") or t.get("open_time") or t.get("created_at")
                if isinstance(ts_raw, (int, float)):
                    ts_str = datetime.fromtimestamp(ts_raw, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
                    ts_check = datetime.fromtimestamp(ts_raw, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
                elif ts_raw:
                    ts_str = str(ts_raw)
                    ts_check = str(ts_raw)
                else:
                    ts_str = "--"
                    ts_check = "--"

                if cleared_at and ts_check != "--" and ts_check < cleared_at:
                    continue

                blitz_trades.append({
                    "id": pos_id, "asset": asset, "direction": direction,
                    "stake": stake, "pnl": profit, "is_win": is_win,
                    "is_equal": is_equal, "time": ts_str
                })

            tot_blitz = len(blitz_trades)
            tot_pages = max(1, (tot_blitz + page_size - 1) // page_size)
            cur_page = min(cur_page, tot_pages)
            offset = (cur_page - 1) * page_size
            page_trades = blitz_trades[offset : offset + page_size]

            wins = sum(1 for t in blitz_trades if t["is_win"])
            ties = sum(1 for t in blitz_trades if t["is_equal"])
            losses = tot_blitz - wins - ties
            net_pnl = sum(t["pnl"] for t in blitz_trades)
            wr = (wins / max(1, wins + losses)) * 100.0 if (wins + losses) > 0 else 0.0

            text = format_blitz_history_html(
                account_type=self.account_type,
                page=cur_page,
                total_pages=tot_pages,
                blitz_trades=page_trades,
                total_count=tot_blitz,
                wins=wins,
                losses=losses,
                ties=ties,
                net_pnl=net_pnl,
                win_rate=wr,
            )
            return HistoryViewResult(text, tot_pages)

        # 3. CFD / Autonomous / Ledger History
        strat_arg = None
        if cat_lower == "snd":
            strat_arg = "S&D"
        elif cat_lower in ("trendline", "trendlines", "tl"):
            strat_arg = "Trendline"
        elif cat_lower == "crt":
            strat_arg = "CRT"
        elif cat_lower == "ict":
            strat_arg = "ICT"
        elif cat_lower in ("cfd", "copiers", "copier"):
            strat_arg = "cfd"
        elif cat_lower in ("manual", "unattributed"):
            strat_arg = "Manual/Unattributed"

        stats = self.trade_ledger.get_stats(strategy=strat_arg, period=per_lower)
        total_count = self.trade_ledger.count_grouped_closed_trades(strategy=strat_arg, period=per_lower)
        tot_pages = max(1, (total_count + page_size - 1) // page_size)
        cur_page = min(cur_page, tot_pages)
        offset = (cur_page - 1) * page_size
        grouped_trades = self.trade_ledger.get_grouped_closed_trades(
            strategy=strat_arg, period=per_lower, limit=page_size, offset=offset
        )

        blitz_summary = None
        if cat_lower == "all" and raw_blitz:
            b_wins = sum(1 for b in raw_blitz if (str(b.get("result", "")).lower() == "win" or float(b.get("profit", 0.0)) > 0))
            b_pnl = sum(float(b.get("profit", 0.0)) for b in raw_blitz)
            blitz_summary = {
                "count": len(raw_blitz),
                "wins": b_wins,
                "losses": len(raw_blitz) - b_wins,
                "pnl": b_pnl,
            }

        total_text = format_history_html(
            account_type=self.account_type,
            category=category,
            period=per_lower,
            page=cur_page,
            total_pages=tot_pages,
            stats=stats,
            trades=grouped_trades,
            symbol_map=symbol_map,
            blitz_summary=blitz_summary,
            sync_note=sync_note,
        )
        return HistoryViewResult(total_text, tot_pages)

    async def build_history_view_async(
        self,
        category: str = "all",
        period: str = "all",
        page: int = 1,
        page_size: int = 12,
        force_sync: bool = False,
    ) -> HistoryViewResult:
        """Non-blocking history builder that runs broker sync & Blitz fetch off the event loop."""
        sync_note, raw_blitz, symbol_map = await asyncio.to_thread(
            self._sync_ledger_and_blitz, category, force_sync
        )
        return self._render_history_from_data(
            category=category,
            period=period,
            page=page,
            page_size=page_size,
            sync_note=sync_note,
            raw_blitz=raw_blitz,
            symbol_map=symbol_map,
        )

    def build_history_view(
        self,
        category: str = "all",
        period: str = "all",
        page: int = 1,
        page_size: int = 12,
        force_sync: bool = False,
    ) -> HistoryViewResult:
        """
        Fetches and compiles compact HTML trade history using Persistent Trade Ledger:
          - Reconciles with broker history across endpoints (with short-TTL throttle)
          - Resolves numeric asset IDs via cached broker symbol map
          - Groups trades by strategy with <pre> fixed-width columns (<=38 chars wide)
          - Places Blitz Options summary directly under header stats
        """
        sync_note, raw_blitz, symbol_map = self._sync_ledger_and_blitz(category, force_sync=force_sync)
        return self._render_history_from_data(
            category=category,
            period=period,
            page=page,
            page_size=page_size,
            sync_note=sync_note,
            raw_blitz=raw_blitz,
            symbol_map=symbol_map,
        )

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        cat = "all"
        per = "today"
        pg = 1
        valid_cats = ["all", "cfd", "ict", "crt", "trendline", "snd", "blitz", "manual"]
        if context.args:
            if context.args[0].lower() in valid_cats:
                cat = context.args[0].lower()
            if len(context.args) > 1 and context.args[1].lower() in ["all", "today", "7d", "30d"]:
                per = context.args[1].lower()
            if len(context.args) > 2 and context.args[2].isdigit():
                pg = int(context.args[2])

        res = await self.build_history_view_async(cat, period=per, page=pg)
        msg = str(res)
        tot_pages = getattr(res, "total_pages", 1)
        is_adm = self.is_admin(user.id) if user else False
        kb = history_menu_keyboard(category=cat, period=per, page=pg, total_pages=tot_pages, is_admin=is_adm)
        try:
            await update.message.reply_text(
                msg,
                parse_mode="HTML",
                reply_markup=kb
            )
        except Exception as e:
            logger.warning(f"HTML error sending history: {e}. Retrying plain text.")
            plain_msg = re.sub(r"<[^>]+>", "", msg)
            await update.message.reply_text(
                plain_msg,
                reply_markup=kb
            )

    async def cmd_clear_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Purges recorded trades prior to today and resets history to retain only today's trades."""
        user = update.effective_user
        if not user or not self.is_admin(user.id):
            if update.message:
                await update.message.reply_text("⛔ Administrator access required for this command.")
            return

        if hasattr(self, "trade_ledger") and self.trade_ledger:
            self.trade_ledger.clear_trades_before_today()
        db.clear_trades_before_today()
        user_manager.log_admin_action(user.id, "clear_trade_history", "Cleared pre-today trades, retaining only today")
        if update.message:
            await update.message.reply_text(
                "🧹 *Trade History Cleared Successfully!*\n\n"
                "All trades prior to today have been purged. "
                "The history view now retains and displays **only trades from today**.",
                parse_mode="Markdown"
            )

    async def cmd_export(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Export persistent trade ledger to CSV file."""
        if not self.is_admin(update.effective_user.id):
            return
        cat = "all"
        per = "all"
        valid_cats = ["all", "cfd", "ict", "crt", "trendline", "snd", "blitz", "manual"]
        if context.args:
            if context.args[0].lower() in valid_cats:
                cat = context.args[0].lower()
            if len(context.args) > 1 and context.args[1].lower() in ["all", "today", "7d", "30d"]:
                per = context.args[1].lower()

        export_dir = os.path.join(os.getcwd(), "data", "exports")
        os.makedirs(export_dir, exist_ok=True)
        ts_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
        export_file = os.path.join(export_dir, f"trades_{cat}_{per}_{ts_filename}.csv")
        self.trade_ledger.export_csv(export_file, strategy=cat, period=per)

        caption = (
            f"📁 *Trade Ledger CSV Export*\n"
            f"• Strategy Filter: `{cat.upper()}`\n"
            f"• Period Filter: `{per.upper()}`\n"
            f"• Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`"
        )
        try:
            with open(export_file, "rb") as f:
                await update.message.reply_document(
                    document=f,
                    filename=os.path.basename(export_file),
                    caption=caption,
                    parse_mode="Markdown"
                )
        except Exception as e:
            logger.error(f"[Export] Failed to send CSV: {e}")
            await update.message.reply_text(f"❌ Failed to export CSV: {e}")

    async def cmd_account(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        if not context.args:
            await update.message.reply_text(
                f"💼 Shared Demo Mode: *TRAINING (DEMO)*\n\n"
                f"🛡️ *Note*: Strict Demo Mode is enforced across all bot engines.\n"
                f"Real account execution is strictly disabled in this shared demonstration environment.",
                parse_mode="Markdown",
                reply_markup=admin_reply_keyboard() if self.is_admin(update.effective_user.id) else investor_reply_keyboard()
            )
            return
        arg = context.args[0].lower()
        if arg in ["real", "regular", "live"]:
            await update.message.reply_text(
                "⚠️ *Strict Demo Mode Enforced*\n"
                "Switching to real money accounts is strictly disabled in this shared demonstration environment.",
                parse_mode="Markdown"
            )
            return
        elif arg in ["demo", "training", "practice"]:
            target = "training"
        else:
            await update.message.reply_text("❌ Invalid account mode. Use: `demo`", parse_mode="Markdown")
            return

        self.account_type = "training"
        bal_fx = self.forex_mcp.get_training_balance()
        if bal_fx:
            bid = bal_fx["balance_id"]
            self.ict_engine.set_balance(bid, "training")
            if self.straddle_engine:
                self.straddle_engine.set_balance(bid, "training")
            if self.crt_engine:
                self.crt_engine.set_balance(bid, "training")
            if self.snd_engine:
                self.snd_engine.set_balance(bid, "training")
            if self.trendline_engine:
                self.trendline_engine.set_balance(bid, "training")
            for c in self.channel_mgr.copiers.values():
                if hasattr(c, "set_balance"):
                    c.set_balance(bid, "training")

        bal_blitz = self.blitz_mcp.get_training_balance()
        if bal_blitz:
            p_copier = self.channel_mgr.get_copier("polycarpvip")
            if p_copier and hasattr(p_copier, "set_balance"):
                p_copier.set_balance(bal_blitz["balance_id"], "training")

        await update.message.reply_text(
            "✅ Re-synchronized to *🟡 PRACTICE (Training / Shared Demo)* Account mode across all engines.",
            parse_mode="Markdown",
            reply_markup=admin_reply_keyboard()
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return

        if not self.is_admin(user.id):
            msg = (
                "ℹ️ *BreakingBad V3 — Investor Quick Guide*\n\n"
                "Welcome to the shared demonstration portal of BreakingBad V3.\n\n"
                "📱 *Available Commands & Features:*\n"
                "• `/status` - Complete bot and strategy status\n"
                "• `/balance` - View shared demo account equity\n"
                "• `/setups` or `/active` - Live institutional orderflow setups\n"
                "• `/history` or `/stats` - Verified trade performance & win rates\n"
                "• `/news` or `/calendar` - High-impact macroeconomic calendar\n"
                "• `/sessions` - Global market trading sessions\n"
                "• `/help` - View this help guide\n"
                "• `/disclaimer` - Review shared demo rules and risk disclaimer\n"
                "• `/set_timezone <tz>` - Customize display timezone (e.g. UTC, US/Eastern)\n"
                "• `/notifications <on/off>` - Enable or disable personal trade alerts\n\n"
                "🛡️ *Demo Environment*: All operations execute strictly on the owner's IQ Option Demo connection."
            )
            await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=investor_reply_keyboard())
            return

        msg = (
            "ℹ️ *BreakingBad V3 Command Center (Administrator)*\n\n"
            "🖱 *Quick Buttons:*\n"
            "Use the bottom keyboard for one-tap balance, status, setups, news, and pause/resume.\n\n"
            "⚡ *Core Commands:*\n"
            "• `/status` - Complete bot and account status\n"
            "• `/balance` - View Forex and Blitz balances\n"
            "• `/news` or `/calendar` - High-impact economic news & CPI/NFP calendar\n"
            "• `/straddle` - Experimental NFP/News Breakout Spike Engine\n"
            "• `/sessions` - View live global market hours & active sessions\n"
            "• `/ict` - Autonomous Gold ICT engine controls\n"
            "• `/channels` - Toggle signal copier channels\n"
            "• `/settings` or `/risk` - Risk & order sizing menu\n"
            "• `/active` - Active zones, FVGs, and open trades\n"
            "• `/prop` or `/propfirm` - Institutional Prop Firm $5,000 Challenge Simulator\n"
            "• `/prop_reset` - Reset $5,000 Evaluation challenge\n"
            "• `/history` - View categorized trade & PnL history\n"
            "• `/pause` / `/resume` - Master trading pause/resume\n"
            "• `/close [id|sym]` - Interactively or directly close an open position\n"
            "• `/closeall` - Emergency close all open positions\n"
            "• `/users` - User management & RBAC list\n"
            "• `/suspend <id>` - Suspend user access\n"
            "• `/activate <id>` - Activate/approve user access\n"
            "• `/audit_logs` - View administrative audit trail\n\n"
            "📊 *Sizing Commands:*\n"
            "• `/lots <val>` - Set global Forex lot size (e.g. `/lots 0.5`)\n"
            "• `/leverage <val>` - Set global leverage (e.g. `/leverage 100`)\n"
            "• `/stake <val>` - Set Blitz base stake (e.g. `/stake 5.0`)\n"
            "• `/account <demo>` - Sync demo account mode"
        )
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=admin_reply_keyboard())

    async def cmd_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        text = self.session_notifier.get_session_dashboard()
        kb = admin_reply_keyboard() if self.is_admin(user.id) else investor_reply_keyboard()
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=kb)

    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        # Ensure fresh calendar data
        await self.news_engine.fetch_calendar()
        text = self.news_engine.get_news_dashboard()
        st_enabled = self.straddle_engine.is_enabled if self.straddle_engine else False
        await update.message.reply_text(
            text,
            parse_mode="Markdown",
            reply_markup=news_menu_keyboard(st_enabled, self.news_engine.is_shield_enabled)
        )

    async def cmd_straddle(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        if not self.straddle_engine:
            await update.message.reply_text("❌ News Straddle Engine is not initialized.")
            return

        if context.args:
            arg = context.args[0].lower()
            if arg in ["on", "enable", "1", "start"]:
                self.straddle_engine.set_enabled(True)
            elif arg in ["off", "disable", "0", "stop"]:
                self.straddle_engine.set_enabled(False)
            elif arg in ["toggle"]:
                self.straddle_engine.toggle_enabled()

        st = self.straddle_engine.get_status()
        icon = "🟢" if st["enabled"] else "🔴"
        st_txt = "ON" if st["enabled"] else "OFF"
        arm_txt = " ⚡ [ARMED - Watching Spike]" if st.get("is_armed") else ""

        msg = (
            f"⚡ *High-Impact News Straddle Spike Engine*\n\n"
            f"• Status: {icon} `{st_txt}`{arm_txt}\n"
            f"• Asset: `XAUUSD (Gold)`\n"
            f"• Lots & Leverage: `{st['lots']:.2f}` Lots | `{st['leverage']}x`\n"
            f"• Trigger Buffer: `±${st['buffer']:.2f}` around pre-news M1 range\n"
            f"• Take Profit: `+${st['tp']:.2f}` | Stop Loss: `-${st['sl']:.2f}`\n"
            f"• Rapid Breakeven: `+$2.00` gain with dynamic trailing\n\n"
            f"Usage: `/straddle on` | `/straddle off` | `/straddle`"
        )
        await update.message.reply_text(
            msg,
            parse_mode="Markdown",
            reply_markup=news_menu_keyboard(self.straddle_engine.is_enabled, self.news_engine.is_shield_enabled)
        )

    async def handle_reply_button(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Processes taps on the persistent reply keyboard."""
        if not update.message or not update.message.text:
            return

        user = update.effective_user
        if not user:
            return

        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start to begin onboarding.")
            return

        is_adm = self.is_admin(user.id)
        text = update.message.text.strip()

        # Investor & Admin shared permitted buttons
        if text in ["📊 Status", "Status"]:
            await self.cmd_status(update, context)
        elif text in ["💰 Balance", "Balance"]:
            await self.cmd_balance(update, context)
        elif text in ["🌐 Market Sessions", "Market Sessions", "🌐 Sessions", "Sessions"]:
            await self.cmd_sessions(update, context)
        elif text in ["📰 Economic News", "Economic News", "📰 News", "News", "Calendar", "📅 Calendar"]:
            await self.cmd_news(update, context)
        elif text in ["🤖 Gold ICT", "Gold ICT", "ICT"]:
            await self.cmd_ict(update, context)
        elif text in ["🕯️ CRT Engine", "CRT Engine", "CRT", "🕯️ CRT"]:
            await self.cmd_crt(update, context)
        elif text in ["🏛️ S&D Engine", "S&D Engine", "SND Engine", "S&D", "SND", "🏛️ SND"]:
            await self.cmd_snd(update, context)
        elif text in ["📐 Trendlines", "Trendlines", "Trendline", "📐 Trendline"]:
            await self.cmd_trendline(update, context)
        elif text in ["🎯 Mission & Strategies", "🎯 Mission", "Mission", "Strategies"]:
            await self.cmd_mission(update, context)
        elif text in ["📋 Active Setups", "Active Setups", "Active", "📈 Active Trades"]:
            await self.cmd_active_trades(update, context)
        elif text in ["🏆 Prop Firm 5K", "Prop Firm 5K", "🏆 Prop Firm", "Prop Firm"]:
            await self.cmd_prop(update, context)
        elif text in ["📜 History", "History", "📋 History", "Trade History", "📊 Stats"]:
            await self.cmd_history(update, context)
        elif text in ["ℹ️ Help", "Help"]:
            await self.cmd_help(update, context)
        elif text in ["🏢 Shared Demo Notice", "Demo Notice", "Disclaimer"]:
            await self.cmd_disclaimer(update, context)
        elif text in ["⚙️ My Preferences", "Preferences"]:
            pref = user_manager.get_user_preferences(user.id)
            tz = pref.get("timezone", "UTC")
            notif = "ON" if pref.get("notifications_enabled", 1) else "OFF"
            msg = (
                f"⚙️ *Your Personal Preferences*\n\n"
                f"• Timezone: `{tz}` (Change with `/set_timezone <tz>`)\n"
                f"• Notifications: `{notif}` (Toggle with `/notifications <on/off>`)\n\n"
                f"🛡️ *Note*: Global trading parameters are managed by the administrator."
            )
            await update.message.reply_text(msg, parse_mode="Markdown")

        # Admin-only buttons
        elif text in ["📡 Channels", "Channels"]:
            if not is_adm:
                await update.message.reply_text("⛔ Administrator access required.")
                return
            await self.cmd_channels(update, context)
        elif text in ["⚙️ Risk & Sizing", "Risk & Sizing", "Settings", "⚙️ Settings"]:
            if not is_adm:
                await update.message.reply_text("⛔ Administrator access required.")
                return
            await self.cmd_settings(update, context)
        elif text in ["👥 Users", "Users", "👥 User Management"]:
            if not is_adm:
                await update.message.reply_text("⛔ Administrator access required.")
                return
            await self.cmd_users(update, context)
        elif text in ["⏸ Pause", "Pause"]:
            if not is_adm:
                await update.message.reply_text("⛔ Administrator access required.")
                return
            await self.cmd_pause(update, context)
        elif text in ["▶ Resume", "Resume"]:
            if not is_adm:
                await update.message.reply_text("⛔ Administrator access required.")
                return
            await self.cmd_resume(update, context)
        elif text in ["🛑 Close All", "Close All"]:
            if not is_adm:
                await update.message.reply_text("⛔ Administrator access required.")
                return
            await self.cmd_closeall(update, context)

    async def cmd_ict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        ict_st = self.ict_engine.get_status()
        kb = ict_menu_keyboard(ict_st) if self.is_admin(user.id) else None
        await update.message.reply_text(
            self.build_ict_dashboard(),
            reply_markup=kb,
            parse_mode="Markdown"
        )

    async def cmd_crt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
        kb = crt_menu_keyboard(crt_st) if self.is_admin(user.id) else None
        await update.message.reply_text(
            "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nActive Forex / Crypto instruments & setup status:",
            reply_markup=kb,
            parse_mode="Markdown"
        )

    async def cmd_snd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
        kb = snd_menu_keyboard(snd_st) if self.is_admin(user.id) else None
        await update.message.reply_text(
            "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\n"
            "Trading top-performing pairs: NZD/USD (+27.0R), USD/JPY (+17.0R), AUD/USD (+15.0R), USD/CAD (+7.0R):\n\n"
            "Current instrument tracking & orderflow status:",
            reply_markup=kb,
            parse_mode="Markdown"
        )

    async def cmd_trendline(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        tl_st = self.trendline_engine.get_status() if self.trendline_engine else {"enabled": False, "enabled_symbols": []}
        kb = trendline_menu_keyboard(tl_st) if self.is_admin(user.id) else None
        await update.message.reply_text(
            "📐 *Autonomous Trendline Bounce Strategy Engine (15M)*\n"
            "Model 1: 3rd Touch Trendline Bounces with 50 EMA Macro Alignment.\n"
            "Validated on **NAS100 (67.3% WR)** and **Gold (62.0% WR)**.\n\n"
            "Current status & signals:",
            reply_markup=kb,
            parse_mode="Markdown"
        )

    async def cmd_mission(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            await update.message.reply_text("⛔ Unauthorized. Please send /start.")
            return
        text = self.build_mission_text()
        await update.message.reply_text(
            text,
            reply_markup=mission_menu_keyboard(),
            parse_mode="Markdown"
        )

    async def cmd_settings(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        await update.message.reply_text(
            f"⚙️ *Risk & Order Sizing Settings*\n"
            f"• Account: `{self.account_type.upper()}`\n"
            f"• Forex Lots: `{self.lots:.2f}`\n"
            f"• Leverage: `{self.leverage}x`\n"
            f"• Blitz Stake: `${self.blitz_stake:.2f}`\n\n"
            f"Adjust below or use `/lots <val>`, `/leverage <val>`, `/stake <val>`:",
            reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
            parse_mode="Markdown"
        )

    async def cmd_lots(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text(f"ℹ️ Current lot size: `{self.lots:.2f}` Lots. Use `/lots <number>` (e.g. `/lots 0.5`) to change.")
            return
        try:
            val = float(context.args[0])
            new_lots = self.update_forex_lots(val)
            await update.message.reply_text(f"✅ Forex / Gold lot size updated to: *{new_lots:.2f} Lots* across all engines.", parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example usage: `/lots 1.5`")

    async def cmd_leverage(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text(f"ℹ️ Current leverage: `{self.leverage}x`. Use `/leverage <number>` (e.g. `/leverage 100`) to change.")
            return
        try:
            val = int(context.args[0])
            new_lev = self.update_forex_leverage(val)
            await update.message.reply_text(f"✅ Forex / Gold leverage updated to: *{new_lev}x* across all engines.", parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("❌ Invalid integer. Example usage: `/leverage 100`")

    async def cmd_stake(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text(f"ℹ️ Current Blitz base stake: `${self.blitz_stake:.2f}`. Use `/stake <number>` (e.g. `/stake 5`) to change.")
            return
        try:
            val = float(context.args[0])
            new_stake = self.update_blitz_stake(val)
            await update.message.reply_text(f"✅ Blitz base stake updated to: *${new_stake:.2f}* (Gale 1: ${new_stake*2.2:.2f}, Gale 2: ${new_stake*(2.2**2):.2f}).", parse_mode="Markdown")
        except ValueError:
            await update.message.reply_text("❌ Invalid number. Example usage: `/stake 2.0`")

    async def cmd_prop(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Displays interactive Prop Firm Challenge $5,000 Simulator Dashboard."""
        if not self.is_admin(update.effective_user.id):
            return
        dash_text = self.prop_guardian.get_dashboard_text()
        kb = prop_firm_menu_keyboard(
            is_enabled=self.prop_guardian.is_enabled,
            phase=self.prop_guardian.phase,
            risk_pct=self.prop_guardian.risk_per_trade_pct
        )
        await update.message.reply_text(
            dash_text,
            reply_markup=kb,
            parse_mode="Markdown"
        )

    async def cmd_prop_reset(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Resets the Prop Firm challenge to fresh $5,000 evaluation."""
        if not self.is_admin(update.effective_user.id):
            return
        self.prop_guardian.reset_challenge(5000.0, 1)
        dash_text = self.prop_guardian.get_dashboard_text()
        kb = prop_firm_menu_keyboard(
            is_enabled=self.prop_guardian.is_enabled,
            phase=self.prop_guardian.phase,
            risk_pct=self.prop_guardian.risk_per_trade_pct
        )
        await update.message.reply_text(
            "🔄 *Prop Firm $5,000 Challenge Reset successfully!*\n\n" + dash_text,
            reply_markup=kb,
            parse_mode="Markdown"
        )

    async def _cleanup_closed_trade(self, pos_id: Any, symbol: Optional[str] = None, close_reason: str = "client_close"):
        """Clean up active tracking in ICT, CRT, and Channel copiers when a position is closed, and dispatch closure settlement notification."""
        pid_str = str(pos_id)
        handled = False

        # 1. Channel Copiers
        if hasattr(self.channel_mgr, "copiers"):
            for c_name, copier in self.channel_mgr.copiers.items():
                if hasattr(copier, "open_positions"):
                    for oid, opos in list(copier.open_positions.items()):
                        if str(oid) == pid_str or str(opos.get("position_id")) == pid_str:
                            handled = True
                            try:
                                if hasattr(copier, "_log_trade_closure"):
                                    await copier._log_trade_closure(oid, reason_override=close_reason)
                            except Exception as ce:
                                logger.warning(f"Error logging trade closure for {c_name} #{pos_id}: {ce}")
                            copier.open_positions.pop(oid, None)
                            logger.info(f"[Cleanup] Cleared {c_name} position #{pos_id}")

        # 2. ICT Engine
        if hasattr(self.ict_engine, "active_trades"):
            for s, t in list(self.ict_engine.active_trades.items()):
                if t and (str(t.get("position_id")) == pid_str or str(t.get("order_id")) == pid_str or any(str(tk.get("position_id")) == pid_str or str(tk.get("order_id")) == pid_str for tk in t.get("tickets", [])) or (symbol and s == symbol)):
                    handled = True
                    try:
                        if hasattr(self.ict_engine, "_log_trade_closure"):
                            await self.ict_engine._log_trade_closure(s, int(pos_id), reason_override=close_reason)
                    except Exception as ie:
                        logger.warning(f"Error logging ICT trade closure #{pos_id}: {ie}")
                    self.ict_engine.active_trades.pop(s, None)
                    logger.info(f"[Cleanup] Cleared ICT trade for {s} (#{pos_id})")

        # 3. CRT Engine
        if hasattr(self.crt_engine, "active_trades"):
            for s, t in list(self.crt_engine.active_trades.items()):
                if t and (str(t.get("position_id")) == pid_str or str(t.get("order_id")) == pid_str or (symbol and s == symbol)):
                    handled = True
                    try:
                        if hasattr(self.crt_engine, "_log_trade_closure"):
                            await self.crt_engine._log_trade_closure(s, t, reason_override=close_reason)
                    except Exception as cre:
                        logger.warning(f"Error logging CRT trade closure #{pos_id}: {cre}")
                    self.crt_engine.active_trades.pop(s, None)
                    logger.info(f"[Cleanup] Cleared CRT trade for {s} (#{pos_id})")

        # 4. SND Engine
        if hasattr(self.snd_engine, "active_trades"):
            for s, t in list(self.snd_engine.active_trades.items()):
                if t and (str(t.get("position_id")) == pid_str or str(t.get("order_id")) == pid_str or (symbol and s == symbol)):
                    handled = True
                    try:
                        if hasattr(self.snd_engine, "_log_trade_closure"):
                            await self.snd_engine._log_trade_closure(s, t, reason_override=close_reason)
                    except Exception as sne:
                        logger.warning(f"Error logging SND trade closure #{pos_id}: {sne}")
                    self.snd_engine.active_trades[s] = None
                    logger.info(f"[Cleanup] Cleared SND trade for {s} (#{pos_id})")

        # 5. Trendline Engine (NAS100 / US100 & Gold)
        if hasattr(self, "trendline_engine") and self.trendline_engine and hasattr(self.trendline_engine, "active_trades"):
            for s, t in list(self.trendline_engine.active_trades.items()):
                if t and (str(t.get("leg1_pos_id")) == pid_str or str(t.get("leg2_pos_id")) == pid_str or (symbol and s == symbol)):
                    handled = True
                    try:
                        if hasattr(self.trendline_engine, "_log_trade_closure"):
                            await self.trendline_engine._log_trade_closure(s, pos_id, reason_override=close_reason)
                    except Exception as tle:
                        logger.warning(f"Error logging Trendline trade closure #{pos_id}: {tle}")
                    logger.info(f"[Cleanup] Handled Trendline trade closure for {s} (#{pos_id})")

        # 6. Settlement to Prop Firm Simulator & fallback closure card for manual positions
        try:
            # Check if this trade was already settled into PropFirmGuardian (e.g. by SND/Trendline/ICT engine)
            already_settled = False
            if hasattr(self, "prop_guardian") and self.prop_guardian:
                if pid_str in self.prop_guardian.processed_trade_ids:
                    already_settled = True
                    logger.info(f"[Cleanup] Trade #{pid_str} was already settled in PropFirmGuardian - skipping duplicate settlement")

            if handled and already_settled:
                # Engine handled both closure notification and guardian settlement
                return

            bid = self.get_active_balance_id()
            if bid:
                matched = None
                for attempt in range(3):
                    hist = self.forex_mcp.get_trade_history(balance_id=bid, limit=20)
                    if hist is not None:
                        matched = next(
                            (h for h in hist if 
                             str(h.get("position_id")) == pid_str or 
                             str(h.get("order_id")) == pid_str or 
                             str(h.get("id")) == pid_str),
                            None
                        )
                        if matched:
                            break
                    if attempt < 2:
                        await asyncio.sleep(0.8)

                if matched:
                    pnl = float(matched.get("pnl", 0.0))
                    exit_px = float(matched.get("close_price", matched.get("exit_price", 0.0)))
                    entry_px = float(matched.get("open_price", matched.get("entry_price", 0.0)))
                    side = str(matched.get("side") or matched.get("direction", "BUY")).upper()
                    lots = float(matched.get("lots") or matched.get("count", 1.0))
                    sym = symbol or self.resolve_asset_name(matched.get("asset_id"))
                    r_name = close_reason or matched.get("close_reason", "client_close")

                    # Settle into Prop Firm Evaluation Simulator (idempotent via trade_id)
                    if not already_settled and hasattr(self, "prop_guardian") and self.prop_guardian.is_enabled:
                        await self.prop_guardian.on_trade_closed(
                            symbol=sym,
                            pnl_usd=pnl,
                            side=side,
                            lots=lots,
                            reason=r_name,
                            trade_id=pid_str
                        )

                    # Persistent Trade Ledger recording (Addendum A)
                    if hasattr(self, "trade_ledger") and self.trade_ledger:
                        try:
                            self.trade_ledger.record_close(
                                position_id=pid_str,
                                exit_price=exit_px,
                                pnl=pnl,
                                exit_reason=r_name,
                                strategy_fallback="Manual/Unattributed",
                                symbol_fallback=sym,
                                side_fallback=side,
                                lots_fallback=lots,
                                entry_fallback=entry_px
                            )
                        except Exception as tle:
                            logger.debug(f"[Cleanup] Error recording to ledger: {tle}")

                    if not handled:
                        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                        if pnl > 0:
                            header_line = f"🏆 [TRADE WON] {sym} {pnl_str}"
                        elif pnl == 0.0 and ("breakeven" in r_name.lower()):
                            header_line = f"🛡️ [TRADE BREAKEVEN] {sym} $0.00"
                        elif pnl < 0:
                            header_line = f"❌ [TRADE CLOSED] {sym} {pnl_str}"
                        else:
                            header_line = f"🛡️ [TRADE CLOSED] {sym} {pnl_str}"

                        card = (
                            f"{header_line}\n\n"
                            f"• Side: {side} ({lots} Lots)\n"
                            f"• Entry: {entry_px:.2f}\n"
                            f"• Exit: {exit_px:.2f}\n"
                            f"• Net PnL: {pnl_str}\n"
                            f"• Reason: {r_name}"
                        )
                        await self.broadcast_alert(card)

                        # Safety-net Google Sheets logging for all unhandled broker closures
                        if gsheet_logger:
                            try:
                                eq = 0.0
                                if hasattr(self, "prop_guardian") and self.prop_guardian:
                                    eq = self.prop_guardian.current_balance
                                fallback_payload = {
                                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                    "asset": f"CFD {sym}",
                                    "side": side,
                                    "lots": lots,
                                    "entry_price": entry_px,
                                    "stop_loss": 0.0,
                                    "take_profit": 0.0,
                                    "exit_price": exit_px,
                                    "pnl": pnl,
                                    "pips": round(abs(exit_px - entry_px), 2),
                                    "risk_reward": "Broker Market Order",
                                    "exit_reason": r_name,
                                    "position_id": str(pid_str),
                                    "balance_equity": eq
                                }
                                asyncio.create_task(asyncio.to_thread(gsheet_logger.log_forex_margin_trade, fallback_payload))
                                logger.info(f"📊 [Controller] Safety-net logged position #{pid_str} ({sym}) to Google Sheets.")
                            except Exception as ge:
                                logger.warning(f"[Controller] Fallback GSheet log error: {ge}")
        except Exception as e:
            logger.warning(f"Closure settlement lookup error for #{pos_id}: {e}")

    async def cmd_close(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Close open trades individually by position ID or symbol, or display interactive close menu."""
        if not self.is_admin(update.effective_user.id):
            return

        bid = self.get_active_balance_id()
        open_positions = []
        if bid:
            try:
                open_positions = self.forex_mcp.list_positions(balance_id=bid) or []
            except Exception as e:
                logger.error(f"[cmd_close] Error fetching positions: {e}")

        if not open_positions:
            await update.message.reply_text("ℹ️ No open positions found on broker to close.", reply_markup=main_menu_keyboard())
            return

        # Build parsed list of open positions
        parsed = []
        for p in open_positions:
            pid = p.get("position_id") or p.get("id")
            aid = p.get("asset_id")
            sym = self.resolve_asset_name(aid)
            side = str(p.get("side") or p.get("type", "BUY")).upper()
            if side == "LONG": side = "BUY"
            if side == "SHORT": side = "SELL"
            lots = float(p.get("lots") or p.get("count") or 1.0)
            pnl = float(p.get("expected_pnl") or p.get("pnl") or 0.0)
            parsed.append({
                "position_id": pid,
                "asset_id": aid,
                "symbol": sym,
                "side": side,
                "lots": lots,
                "pnl": pnl
            })

        # Check if user specified an argument: e.g. /close 100104887557 or /close btc or /close gold
        args = context.args if context and context.args else []
        if args:
            target_str = str(args[0]).strip().lower()
            matched = None
            for p in parsed:
                pid = str(p.get("position_id", ""))
                sym = str(p.get("symbol", "")).lower()
                aid = str(p.get("asset_id", ""))
                if target_str == pid:
                    matched = p
                    break
                if target_str in sym or sym in target_str:
                    matched = p
                    break
                if target_str == aid:
                    matched = p
                    break
                # Common aliases
                if target_str in ["btc", "bitcoin"] and "btc" in sym:
                    matched = p
                    break
                if target_str in ["gold", "xau"] and "xau" in sym:
                    matched = p
                    break
                if target_str in ["silver", "xag"] and "xag" in sym:
                    matched = p
                    break
                if target_str in ["eur", "eurusd"] and "eur" in sym:
                    matched = p
                    break

            if not matched:
                await update.message.reply_text(
                    f"❌ Could not find an open position matching `{args[0]}`.\n\nSelect a trade below to close:",
                    reply_markup=individual_close_keyboard(parsed),
                    parse_mode="Markdown"
                )
                return

            # Found matching position, execute closure
            close_pid = matched["position_id"]
            sym = matched["symbol"]
            res = self.forex_mcp.close_position(position_id=int(close_pid))
            await self._cleanup_closed_trade(close_pid, sym)
            pnl_val = matched.get("pnl", 0.0)
            pnl_sign = f"+${pnl_val:.2f}" if pnl_val >= 0 else f"-${abs(pnl_val):.2f}"
            await update.message.reply_text(
                f"✅ Closed *{sym}* position `#{close_pid}`.\nRealized PnL: *{pnl_sign}*",
                reply_markup=main_menu_keyboard(),
                parse_mode="Markdown"
            )
            return

        # No argument passed: display interactive close menu with buttons
        await update.message.reply_text(
            f"🎯 *Close Open Trades Individually*\nFound *{len(parsed)}* open position(s). Tap any trade below to close it:",
            reply_markup=individual_close_keyboard(parsed),
            parse_mode="Markdown"
        )

    async def cmd_closeall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        await update.message.reply_text(
            "⚠️ Are you sure you want to CLOSE ALL open positions?",
            reply_markup=close_all_confirm_keyboard()
        )

    def _build_users_overview_message(self) -> Tuple[str, str, bool]:
        """Builds formatted markdown and plain text overview of registered users & alert status."""
        users = user_manager.list_all_users()
        counts = user_manager.get_user_counts()
        alerts_on = user_manager.are_user_alerts_enabled()
        alert_status_str = "🟢 ENABLED (Broadcasts active)" if alerts_on else "🔴 DISABLED (Muted)"

        md_lines = [
            f"👥 *Registered Bot Users* ({counts.get('total', 0)} total, {counts.get('approved', 0)} approved)",
            f"📢 *User Trade Alerts*: {alert_status_str}\n"
        ]
        plain_lines = [
            f"👥 Registered Bot Users ({counts.get('total', 0)} total, {counts.get('approved', 0)} approved)",
            f"📢 User Trade Alerts: {alert_status_str}\n"
        ]

        for u in users[:25]:
            status_val = u.get("status") or u.get("access_status", "pending")
            status_icon = "🟢" if status_val == "approved" else "🟡" if status_val == "pending" else "🔴"
            role_icon = "👑" if u.get("role") == "admin" else "👤"

            raw_uname = (u.get("username") or "").replace("_", "\\_")
            safe_uname = f"@{raw_uname}" if raw_uname else "No username"
            safe_fname = (u.get("first_name") or "").replace("_", "\\_").replace("*", "\\*")

            md_lines.append(f"{status_icon} {role_icon} *{safe_fname}* ({safe_uname})")
            md_lines.append(f"   ID: `{u['telegram_id']}` | Status: `{status_val}`")

            plain_uname = f"@{u['username']}" if u.get("username") else "No username"
            plain_fname = u.get("first_name") or "User"
            plain_lines.append(f"{status_icon} {role_icon} {plain_fname} ({plain_uname})")
            plain_lines.append(f"   ID: {u['telegram_id']} | Status: {status_val}")

        controls_md = "\n💡 Admin commands:\n• `/user_alerts [on|off]` - Toggle investor broadcasts\n• `/suspend <id>`\n• `/activate <id>`\n• `/user <id>`\n• `/audit_logs`"
        controls_plain = "\n💡 Admin commands:\n• /user_alerts [on|off] - Toggle investor broadcasts\n• /suspend <id>\n• /activate <id>\n• /user <id>\n• /audit_logs"

        md_lines.append(controls_md)
        plain_lines.append(controls_plain)

        return "\n".join(md_lines), "\n".join(plain_lines), alerts_on

    async def cmd_users(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        md_msg, plain_msg, alerts_on = self._build_users_overview_message()
        try:
            await update.message.reply_text(md_msg, parse_mode="Markdown", reply_markup=user_management_keyboard(alerts_enabled=alerts_on))
        except Exception as e:
            logger.warning(f"Markdown send failed in cmd_users: {e}. Retrying without markdown.")
            await update.message.reply_text(plain_msg, reply_markup=user_management_keyboard(alerts_enabled=alerts_on))

    async def cmd_toggle_user_alerts(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        """Allows super-admin to toggle or set user broadcast alerts on/off."""
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return

        current = user_manager.are_user_alerts_enabled()
        new_state = not current

        if context.args:
            arg = context.args[0].lower().strip()
            if arg in ("on", "enable", "true", "1"):
                new_state = True
            elif arg in ("off", "disable", "false", "0"):
                new_state = False

        user_manager.set_user_alerts_enabled(new_state, admin_id=update.effective_user.id)
        status_txt = "🟢 ENABLED (Approved users will receive trade & signal updates)" if new_state else "🔴 DISABLED (User updates muted; Admin-only)"
        await update.message.reply_text(
            f"📢 *User Broadcast Alerts*: {status_txt}",
            parse_mode="Markdown"
        )

    async def cmd_suspend(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        if not context.args:
            await update.message.reply_text("Usage: `/suspend <telegram_id>`", parse_mode="Markdown")
            return
        try:
            target_id = int(context.args[0])
            user_manager.suspend_user(target_id, admin_id=update.effective_user.id)
            await update.message.reply_text(f"✅ User `{target_id}` has been SUSPENDED.", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Error suspending user: {e}")

    async def cmd_activate(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        if not context.args:
            await update.message.reply_text("Usage: `/activate <telegram_id>`", parse_mode="Markdown")
            return
        try:
            target_id = int(context.args[0])
            user_manager.approve_user(target_id, admin_id=update.effective_user.id)
            await update.message.reply_text(f"✅ User `{target_id}` has been ACTIVATED/APPROVED.", parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Error activating user: {e}")

    async def cmd_user(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        if not context.args:
            await update.message.reply_text("Usage: `/user <telegram_id>`", parse_mode="Markdown")
            return
        try:
            target_id = int(context.args[0])
            u = user_manager.get_user(target_id)
            if not u:
                await update.message.reply_text("❌ User not found.")
                return
            safe_fname = (u.get("first_name") or "").replace("_", "\\_").replace("*", "\\*")
            raw_uname = (u.get("username") or "").replace("_", "\\_")
            safe_uname = f"@{raw_uname}" if raw_uname else "N/A"
            msg = (
                f"👤 *User Details*\n\n"
                f"• ID: `{u['telegram_id']}`\n"
                f"• Name: *{safe_fname}*\n"
                f"• Username: {safe_uname}\n"
                f"• Role: `{u.get('role', 'investor')}`\n"
                f"• Status: `{u.get('access_status', 'approved')}`\n"
                f"• Registered: `{u.get('created_at', 'N/A')}`\n"
                f"• Last Active: `{u.get('last_active', 'N/A')}`\n"
                f"• Onboarded: `{'Yes' if u.get('onboarding_completed') else 'No'}`\n"
            )
            await update.message.reply_text(msg, parse_mode="Markdown")
        except Exception as e:
            await update.message.reply_text(f"❌ Error fetching user: {e}")

    async def cmd_audit_logs(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Administrator access required.")
            return
        logs = user_manager.get_audit_logs(limit=15)
        if not logs:
            await update.message.reply_text("📋 No administrative audit logs found.")
            return
        msg = "📜 *Recent Administrative Audit Logs*\n\n"
        for l in logs:
            msg += f"• `[{l.get('timestamp', '')[:19]}]` Admin `{l.get('admin_id')}`: *{l.get('action')}* on user `{l.get('target_user_id')}`\n"
        await update.message.reply_text(msg, parse_mode="Markdown")

    async def cmd_set_tz(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        if not context.args:
            await update.message.reply_text("Usage: `/set_timezone <e.g. UTC, America/New_York, Europe/London>`", parse_mode="Markdown")
            return
        tz_name = context.args[0].strip()
        user_manager.set_user_preference(user.id, "timezone", tz_name)
        await update.message.reply_text(f"✅ Your preferred display timezone is set to `{tz_name}`.", parse_mode="Markdown")

    async def cmd_notifications(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        user = update.effective_user
        if not user:
            return
        if not context.args:
            await update.message.reply_text("Usage: `/notifications <on/off>`", parse_mode="Markdown")
            return
        setting = 1 if context.args[0].lower() in ["on", "1", "true", "yes"] else 0
        user_manager.set_user_preference(user.id, "notifications_enabled", setting)
        status_txt = "ENABLED" if setting else "DISABLED"
        await update.message.reply_text(f"✅ Your trade notifications are now *{status_txt}*.", parse_mode="Markdown")

    async def cmd_disclaimer(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        msg = (
            "⚠️ *BreakingBad V3 Shared Demo Environment Disclaimer*\n\n"
            "1. **Simulated Demonstration Account**: All operations, signals, and executions run on the owner's IQ Option Demo account.\n"
            "2. **Risk Disclosure**: Binary options and CFD trading carry high risk. Information provided is for demonstration only.\n"
            "3. **Zero Credentials Needed**: You will never be asked for your IQ Option password or credentials.\n"
            "4. **Strict Demo Mode**: Real-money mode is permanently disabled for all users in this environment."
        )
        await update.message.reply_text(msg, parse_mode="Markdown")

    # ==========================================
    # Callback Query Handler
    # ==========================================

    async def on_button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            pass

        data = query.data

        # Onboarding disclaimer acceptance & informational callbacks
        if data == "btn_accept_disclaimer":
            user_manager.complete_user_onboarding(query.from_user.id)
            await query.edit_message_text(
                "✅ *Onboarding Completed!*\n\n"
                "Welcome to the BreakingBad V3 Shared Demonstration Environment.\n"
                "You now have access to live strategy monitoring, shared demo balances, and verified history.",
                parse_mode="Markdown"
            )
            await query.message.reply_text(
                "📱 Use the reply menu below to explore BreakingBad V3:",
                reply_markup=investor_reply_keyboard()
            )
            return

        elif data == "btn_disclaimer":
            disclaimer_text = (
                "⚠️ *BreakingBad V3 Shared Demo Environment Disclaimer*\n\n"
                "1. **Simulated Demonstration Account**: All operations, signals, and executions run on the owner's IQ Option Demo account.\n"
                "2. **Risk Disclosure**: Binary options and CFD trading carry high risk. Information provided is for demonstration only.\n"
                "3. **Zero Credentials Needed**: You will never be asked for your IQ Option password or credentials.\n"
                "4. **Strict Demo Mode**: Real-money mode is permanently disabled for all users in this environment."
            )
            await query.message.reply_text(disclaimer_text, parse_mode="Markdown")
            return

        elif data == "btn_view_config":
            cfg_text = (
                "⚙️ *Current Shared Demo Configuration*\n\n"
                "• Mode: `🟡 PRACTICE (Training)`\n"
                "• Strategy Engines: `Gold ICT, CRT, S&D Config E, Trendlines`\n"
                "• Execution Engines: `Gold Confirmation Zones, Momentum Scalp, Blitz Options`\n"
                "• Base Blitz Stake: `$2.00` | Blitz Martingale: `2-Step Smart Recovery`\n"
                "• Forex Sizing: `1.00 Lot` | `100x Leverage`\n"
                "• Account Protection: `Prop Firm 5K Guardian`\n\n"
                "🛡️ *Note*: Global trading parameters are managed by the administrator."
            )
            await query.message.reply_text(cfg_text, parse_mode="Markdown")
            return

        elif data == "btn_investor_menu":
            text = await self.build_status_text_async()
            await query.edit_message_text(text=text, reply_markup=investor_main_menu_keyboard(), parse_mode="Markdown")
            return

        # Check user authorization for standard buttons
        user = query.from_user
        allowed, status, _ = user_manager.check_access(user.id, user.username, user.first_name)
        if not allowed:
            try:
                await query.answer("⛔ Access denied or suspended.", show_alert=True)
            except Exception:
                pass
            return

        investor_allowed_prefixes = (
            "btn_status", "btn_main_menu", "btn_investor_menu", "hist_", "histref_", "history_cat_", "btn_history_menu",
            "btn_active_trades", "btn_news_menu", "btn_sessions", "btn_mission_menu",
            "btn_snd_menu", "btn_trendline_menu", "btn_crt_menu", "btn_ict_menu",
            "btn_prop_menu", "btn_disclaimer", "btn_view_config", "btn_accept_disclaimer"
        )
        if not self.is_admin(user.id) and not any(data.startswith(p) for p in investor_allowed_prefixes):
            try:
                await query.answer("⛔ Administrator access required for this action.", show_alert=True)
            except Exception:
                pass
            return

        if not data.startswith("btn_active_trades"):
            self.stop_active_view_refresh_task()

        if data.startswith("noop"):
            return

        # Main Navigation
        if data == "btn_main_menu":
            text = await self.build_status_text_async()
            kb = main_menu_keyboard(is_admin=True) if self.is_admin(user.id) else investor_main_menu_keyboard()
            await query.edit_message_text(text=text, reply_markup=kb, parse_mode="Markdown")

        elif data == "btn_status":
            text = await self.build_status_text_async()
            kb = main_menu_keyboard(is_admin=True) if self.is_admin(user.id) else investor_main_menu_keyboard()
            await query.edit_message_text(text=text, reply_markup=kb, parse_mode="Markdown")

        # Channels Menu
        elif data == "btn_channels_menu":
            copiers = self.channel_mgr.list_copiers()
            await query.edit_message_text(
                "📡 *Channel Copiers Management*\nToggle signal monitoring for individual channels:",
                reply_markup=channels_menu_keyboard(copiers),
                parse_mode="Markdown"
            )

        elif data.startswith("toggle_channel_"):
            name = data.replace("toggle_channel_", "")
            self.channel_mgr.toggle_copier(name)
            copiers = self.channel_mgr.list_copiers()
            await query.edit_message_text(
                "📡 *Channel Copiers Management*\nToggle signal monitoring for individual channels:",
                reply_markup=channels_menu_keyboard(copiers),
                parse_mode="Markdown"
            )

        # History Menu & Filtering (Addendum A)
        elif data.startswith("hist_") or data.startswith("histref_") or data.startswith("history_cat_") or data == "btn_history_menu":
            force_sync = data.startswith("histref_")
            if data == "btn_history_menu":
                cat, per, pg = "all", "today", 1
            elif data.startswith("history_cat_"):
                cat = data.replace("history_cat_", "")
                per = "all"
                pg = 1
            else:
                parts = data.split("_")
                cat = parts[1] if len(parts) > 1 else "all"
                per = parts[2] if len(parts) > 2 else "all"
                pg = int(parts[3]) if len(parts) > 3 and parts[3].isdigit() else 1

            res = await self.build_history_view_async(cat, period=per, page=pg, force_sync=force_sync)
            txt = str(res)
            tot_pages = getattr(res, "total_pages", 1)
            is_adm = self.is_admin(user.id)
            kb = history_menu_keyboard(category=cat, period=per, page=pg, total_pages=tot_pages, is_admin=is_adm)
            try:
                await query.edit_message_text(
                    txt,
                    reply_markup=kb,
                    parse_mode="HTML"
                )
            except Exception as e:
                if "Message is not modified" in str(e):
                    return
                logger.warning(f"HTML error in history button callback: {e}. Retrying plain text.")
                plain_txt = re.sub(r"<[^>]+>", "", txt)
                await query.edit_message_text(
                    plain_txt,
                    reply_markup=kb
                )

        elif data == "clear_history_prompt":
            if not self.is_admin(user.id):
                await query.answer("⛔ Administrator access required.", show_alert=True)
                return
            prompt_text = (
                "⚠️ *Are you sure you want to clear all trade history?*\n\n"
                "This will purge all CFD, options, and persistent ledger records. "
                "History views will start fresh from this point forward."
            )
            confirm_kb = InlineKeyboardMarkup([
                [
                    InlineKeyboardButton("⚠️ YES, CLEAR ALL TRADE HISTORY", callback_data="action_clear_history_execute")
                ],
                [
                    InlineKeyboardButton("❌ CANCEL", callback_data="btn_history_menu")
                ]
            ])
            await query.edit_message_text(prompt_text, reply_markup=confirm_kb, parse_mode="Markdown")
            return

        elif data == "action_clear_history_execute":
            if not self.is_admin(user.id):
                await query.answer("⛔ Administrator access required.", show_alert=True)
                return
            if hasattr(self, "trade_ledger") and self.trade_ledger:
                self.trade_ledger.clear_trades_before_today()
            db.clear_trades_before_today()
            user_manager.log_admin_action(user.id, "clear_trade_history", "Cleared pre-today trades, retaining only today via Telegram button")
            await query.answer("🧹 Cleared pre-today trades! Retaining only today's trades.", show_alert=True)
            res = await self.build_history_view_async(category="all", period="today", page=1, force_sync=False)
            kb = history_menu_keyboard(category="all", period="today", page=1, total_pages=res.total_pages, is_admin=True)
            await query.edit_message_text(str(res), reply_markup=kb, parse_mode="HTML")
            return

        elif data.startswith("export_hist_"):
            parts = data.split("_")
            cat = parts[2] if len(parts) > 2 else "all"
            per = parts[3] if len(parts) > 3 else "all"
            export_dir = os.path.join(os.getcwd(), "data", "exports")
            os.makedirs(export_dir, exist_ok=True)
            ts_filename = datetime.now().strftime("%Y%m%d_%H%M%S")
            export_file = os.path.join(export_dir, f"trades_{cat}_{per}_{ts_filename}.csv")
            self.trade_ledger.export_csv(export_file, strategy=cat, period=per)

            caption = (
                f"📁 *Trade Ledger CSV Export*\n"
                f"• Strategy Filter: `{cat.upper()}`\n"
                f"• Period Filter: `{per.upper()}`\n"
                f"• Generated: `{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}`"
            )
            try:
                with open(export_file, "rb") as f:
                    await query.message.reply_document(
                        document=f,
                        filename=os.path.basename(export_file),
                        caption=caption,
                        parse_mode="Markdown"
                    )
            except Exception as e:
                logger.error(f"[Export] Failed to send CSV via callback: {e}")
                await query.answer(f"Failed to export CSV: {e}", show_alert=True)

        # Prop Firm Simulator Menu & Callbacks
        elif data == "btn_prop_menu":
            dash_text = self.prop_guardian.get_dashboard_text()
            kb = prop_firm_menu_keyboard(
                is_enabled=self.prop_guardian.is_enabled,
                phase=self.prop_guardian.phase,
                risk_pct=self.prop_guardian.risk_per_trade_pct
            )
            await query.edit_message_text(
                dash_text,
                reply_markup=kb,
                parse_mode="Markdown"
            )

        elif data == "toggle_prop_master":
            self.prop_guardian.toggle()
            dash_text = self.prop_guardian.get_dashboard_text()
            kb = prop_firm_menu_keyboard(
                is_enabled=self.prop_guardian.is_enabled,
                phase=self.prop_guardian.phase,
                risk_pct=self.prop_guardian.risk_per_trade_pct
            )
            await query.edit_message_text(
                dash_text,
                reply_markup=kb,
                parse_mode="Markdown"
            )

        elif data.startswith("set_prop_phase_"):
            p_val = int(data.replace("set_prop_phase_", ""))
            self.prop_guardian.set_phase(p_val)
            dash_text = self.prop_guardian.get_dashboard_text()
            kb = prop_firm_menu_keyboard(
                is_enabled=self.prop_guardian.is_enabled,
                phase=self.prop_guardian.phase,
                risk_pct=self.prop_guardian.risk_per_trade_pct
            )
            await query.edit_message_text(
                dash_text,
                reply_markup=kb,
                parse_mode="Markdown"
            )

        elif data.startswith("set_prop_risk_"):
            r_val = float(data.replace("set_prop_risk_", ""))
            self.prop_guardian.set_risk(r_val)
            dash_text = self.prop_guardian.get_dashboard_text()
            kb = prop_firm_menu_keyboard(
                is_enabled=self.prop_guardian.is_enabled,
                phase=self.prop_guardian.phase,
                risk_pct=self.prop_guardian.risk_per_trade_pct
            )
            await query.edit_message_text(
                dash_text,
                reply_markup=kb,
                parse_mode="Markdown"
            )

        elif data == "reset_prop_challenge":
            self.prop_guardian.reset_challenge(5000.0, 1)
            dash_text = self.prop_guardian.get_dashboard_text()
            kb = prop_firm_menu_keyboard(
                is_enabled=self.prop_guardian.is_enabled,
                phase=self.prop_guardian.phase,
                risk_pct=self.prop_guardian.risk_per_trade_pct
            )
            await query.edit_message_text(
                "🔄 *Challenge Reset! Fresh $5,000.00 Evaluation Account Initialized.*\n\n" + dash_text,
                reply_markup=kb,
                parse_mode="Markdown"
            )

        # ICT Engine Menu
        elif data == "btn_ict_menu":
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_ict_engine":
            self.ict_engine.toggle()
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_ict_hybrid_trail":
            is_on = self.ict_engine.toggle_hybrid_trailing()
            ict_st = self.ict_engine.get_status()
            status_desc = "ON (Locks profit @ pip milestones + R)" if is_on else "OFF (Pure R-Multiples)"
            await query.answer(f"Hybrid Pip Trailing: {status_desc}")
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("toggle_inst_") or data.startswith("set_inst_"):
            sym_raw = data.replace("toggle_inst_", "").replace("set_inst_", "").upper()
            self.ict_engine.toggle_symbol(sym_raw)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "ict_lots_minus":
            new_l = max(0.01, round(self.ict_engine.lots - 0.1, 2))
            self.ict_engine.set_lots(new_l)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "ict_lots_plus":
            new_l = round(self.ict_engine.lots + 0.1, 2)
            self.ict_engine.set_lots(new_l)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_ict_lots_"):
            val = float(data.replace("set_ict_lots_", ""))
            self.ict_engine.set_lots(val)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_ict_lev_"):
            val = int(data.replace("set_ict_lev_", ""))
            self.ict_engine.set_leverage(val)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_rr_"):
            rr = float(data.replace("set_rr_", ""))
            self.ict_engine.rr_ratio = rr
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                self.build_ict_dashboard(),
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        # CRT Engine Menu
        elif data == "btn_crt_menu":
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nConfigure active Forex / Crypto instruments & sizing:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_crt_master":
            if self.crt_engine:
                self.crt_engine.toggle()
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nConfigure active Forex / Crypto instruments & sizing:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        elif data.startswith("toggle_crt_"):
            sym = data.replace("toggle_crt_", "").upper()
            if self.crt_engine:
                self.crt_engine.toggle_symbol(sym)
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nConfigure active Forex / Crypto instruments & sizing:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        elif data == "crt_lots_minus":
            if self.crt_engine:
                self.crt_engine.set_lots(self.crt_engine.lots - 0.1)
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nAdjust instrument, lot size, and leverage:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        elif data == "crt_lots_plus":
            if self.crt_engine:
                self.crt_engine.set_lots(self.crt_engine.lots + 0.1)
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nAdjust instrument, lot size, and leverage:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_crt_lots_"):
            val = float(data.replace("set_crt_lots_", ""))
            if self.crt_engine:
                self.crt_engine.set_lots(val)
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nAdjust instrument, lot size, and leverage:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_crt_lev_"):
            val = int(data.replace("set_crt_lev_", ""))
            if self.crt_engine:
                self.crt_engine.set_leverage(val)
            crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nAdjust instrument, lot size, and leverage:",
                reply_markup=crt_menu_keyboard(crt_st),
                parse_mode="Markdown"
            )

        # Trendline Engine Menu
        elif data == "btn_trendline_menu":
            tl_st = self.trendline_engine.get_status() if self.trendline_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "📐 *Autonomous Trendline Bounce Strategy Engine (15M)*\n"
                "Model 1: 3rd Touch Trendline Bounces with 50 EMA Macro Alignment.\n"
                "Validated on **NAS100 (67.3% WR)** and **Gold (62.0% WR)**.\n\n"
                "Configure active assets and master switches:",
                reply_markup=trendline_menu_keyboard(tl_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_trendline_master":
            if self.trendline_engine:
                self.trendline_engine.toggle()
            tl_st = self.trendline_engine.get_status() if self.trendline_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "📐 *Autonomous Trendline Bounce Strategy Engine (15M)*\n"
                "Model 1: 3rd Touch Trendline Bounces with 50 EMA Macro Alignment.\n"
                "Validated on **NAS100 (67.3% WR)** and **Gold (62.0% WR)**.\n\n"
                "Configure active assets and master switches:",
                reply_markup=trendline_menu_keyboard(tl_st),
                parse_mode="Markdown"
            )

        elif data.startswith("toggle_trendline_"):
            sym = data.replace("toggle_trendline_", "").upper()
            if self.trendline_engine:
                self.trendline_engine.toggle_symbol(sym)
            tl_st = self.trendline_engine.get_status() if self.trendline_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "📐 *Autonomous Trendline Bounce Strategy Engine (15M)*\n"
                "Model 1: 3rd Touch Trendline Bounces with 50 EMA Macro Alignment.\n"
                "Validated on **NAS100 (67.3% WR)** and **Gold (62.0% WR)**.\n\n"
                "Configure active assets and master switches:",
                reply_markup=trendline_menu_keyboard(tl_st),
                parse_mode="Markdown"
            )

        # S&D Engine Menu (Config E)
        elif data == "btn_snd_menu":
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\n"
                "Trading top-performing pairs: NZD/USD (+27.0R), USD/JPY (+17.0R), AUD/USD (+15.0R), USD/CAD (+7.0R):\n\n"
                "Configure active pairs, lots, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_snd_master":
            if self.snd_engine:
                self.snd_engine.toggle()
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\n"
                "Trading top-performing pairs: NZD/USD (+27.0R), USD/JPY (+17.0R), AUD/USD (+15.0R), USD/CAD (+7.0R):\n\n"
                "Configure active pairs, lots, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        elif data.startswith("toggle_snd_"):
            sym = data.replace("toggle_snd_", "").upper()
            if self.snd_engine:
                self.snd_engine.toggle_symbol(sym)
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\n"
                "Trading top-performing pairs: NZD/USD (+27.0R), USD/JPY (+17.0R), AUD/USD (+15.0R), USD/CAD (+7.0R):\n\n"
                "Configure active pairs, lots, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        elif data == "snd_lots_minus":
            if self.snd_engine:
                self.snd_engine.set_lots(self.snd_engine.lots - 0.1)
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\nAdjust instrument, lot size, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        elif data == "snd_lots_plus":
            if self.snd_engine:
                self.snd_engine.set_lots(self.snd_engine.lots + 0.1)
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\nAdjust instrument, lot size, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_snd_lots_"):
            val = float(data.replace("set_snd_lots_", ""))
            if self.snd_engine:
                self.snd_engine.set_lots(val)
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\nAdjust instrument, lot size, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_snd_lev_"):
            val = int(data.replace("set_snd_lev_", ""))
            if self.snd_engine:
                self.snd_engine.set_leverage(val)
            snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
            await query.edit_message_text(
                "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\nAdjust instrument, lot size, and leverage:",
                reply_markup=snd_menu_keyboard(snd_st),
                parse_mode="Markdown"
            )

        # Mission & Strategy Description
        elif data == "btn_mission_strategies":
            text = self.build_mission_text()
            await query.edit_message_text(
                text=text,
                reply_markup=mission_menu_keyboard(),
                parse_mode="Markdown"
            )

        # Risk & Sizing / Account Menu
        elif data in ["btn_settings_menu", "btn_account_menu"]:
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\n"
                f"• Account: `{self.account_type.upper()}`\n"
                f"• Forex Lots: `{self.lots:.2f}`\n"
                f"• Leverage: `{self.leverage}x`\n"
                f"• Blitz Stake: `${self.blitz_stake:.2f}`\n\n"
                f"Adjust below or use commands `/lots`, `/leverage`, `/stake`:",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data == "stake_minus":
            self.update_blitz_stake(self.blitz_stake - 1.0)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nBlitz base stake set to: *${self.blitz_stake:.2f}*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data == "stake_plus":
            self.update_blitz_stake(self.blitz_stake + 1.0)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nBlitz base stake set to: *${self.blitz_stake:.2f}*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data.startswith("set_stake_"):
            val = float(data.replace("set_stake_", ""))
            self.update_blitz_stake(val)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nBlitz base stake set to: *${self.blitz_stake:.2f}*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data == "global_lots_minus":
            self.update_forex_lots(self.lots - 0.1)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nGlobal Forex lots set to: *{self.lots:.2f}*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data == "global_lots_plus":
            self.update_forex_lots(self.lots + 0.1)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nGlobal Forex lots set to: *{self.lots:.2f}*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data.startswith("set_global_lots_"):
            val = float(data.replace("set_global_lots_", ""))
            self.update_forex_lots(val)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nGlobal Forex lots set to: *{self.lots:.2f}*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data.startswith("set_global_lev_"):
            val = int(data.replace("set_global_lev_", ""))
            self.update_forex_leverage(val)
            await query.edit_message_text(
                f"⚙️ *Risk & Order Sizing Settings*\nGlobal Forex leverage set to: *{self.leverage}x*",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data == "set_acc_training":
            self.account_type = "training"
            bal_fx = self.forex_mcp.get_training_balance()
            if bal_fx:
                bid = bal_fx["balance_id"]
                self.ict_engine.set_balance(bid, "training")
                if self.straddle_engine:
                    self.straddle_engine.set_balance(bid, "training")
                if self.crt_engine:
                    self.crt_engine.set_balance(bid, "training")
                if self.snd_engine:
                    self.snd_engine.set_balance(bid, "training")
                if self.trendline_engine:
                    self.trendline_engine.set_balance(bid, "training")
                for c in self.channel_mgr.copiers.values():
                    if hasattr(c, "set_balance"):
                        c.set_balance(bid, "training")

            bal_blitz = self.blitz_mcp.get_training_balance()
            if bal_blitz:
                p_copier = self.channel_mgr.get_copier("polycarpvip")
                if p_copier and hasattr(p_copier, "set_balance"):
                    p_copier.set_balance(bal_blitz["balance_id"], "training")

            await query.edit_message_text(
                "✅ Switched to *PRACTICE (Training)* Account mode!",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data == "set_acc_regular":
            try:
                await query.answer("⚠️ Strict Demo Mode Enforced. Real accounts disabled.", show_alert=True)
            except Exception:
                pass
            await query.edit_message_text(
                "⚠️ *Strict Demo Mode Enforced*\n"
                "Switching to real money accounts is strictly disabled in this shared demonstration environment.",
                reply_markup=settings_menu_keyboard("training", self.lots, self.leverage, self.blitz_stake),
                parse_mode="Markdown"
            )

        elif data in ["btn_active_trades", "btn_active_trades_refresh"]:
            text = self.build_active_setups_view()
            kb = active_setups_keyboard(is_live=True, open_positions=self.last_open_positions)
            try:
                await query.edit_message_text(
                    text=text,
                    reply_markup=kb,
                    parse_mode="Markdown"
                )
            except Exception:
                await query.edit_message_text(
                    text=text,
                    reply_markup=kb
                )
            self.start_active_view_refresh_task(query.message.chat_id, query.message.message_id)

        elif data == "toggle_user_broadcasts":
            if not self.is_admin(user.id):
                await query.answer("⛔ Administrator access required.", show_alert=True)
                return
            current = user_manager.are_user_alerts_enabled()
            new_state = not current
            user_manager.set_user_alerts_enabled(new_state, admin_id=user.id)
            alert_txt = "ON 🟢 (Broadcasts active)" if new_state else "OFF 🔴 (Broadcasts muted)"
            await query.answer(f"User Alerts toggled {alert_txt}", show_alert=False)

            md_msg, plain_msg, alerts_on = self._build_users_overview_message()
            try:
                await query.edit_message_text(md_msg, parse_mode="Markdown", reply_markup=user_management_keyboard(alerts_enabled=alerts_on))
            except Exception:
                await query.edit_message_text(plain_msg, reply_markup=user_management_keyboard(alerts_enabled=alerts_on))
            return

        elif data == "btn_users_refresh":
            if not self.is_admin(user.id):
                await query.answer("⛔ Administrator access required.", show_alert=True)
                return
            await query.answer("Refreshing user list...")
            md_msg, plain_msg, alerts_on = self._build_users_overview_message()
            try:
                await query.edit_message_text(md_msg, parse_mode="Markdown", reply_markup=user_management_keyboard(alerts_enabled=alerts_on))
            except Exception:
                await query.edit_message_text(plain_msg, reply_markup=user_management_keyboard(alerts_enabled=alerts_on))
            return

        elif data == "btn_users_audit_logs":
            if not self.is_admin(user.id):
                await query.answer("⛔ Administrator access required.", show_alert=True)
                return
            logs = user_manager.get_audit_logs(limit=15)
            if not logs:
                log_txt = "📜 *Admin Audit Logs*\n\n_No audit logs recorded yet._"
            else:
                log_txt = f"📜 *Recent Admin Audit Logs* ({len(logs)} records)\n━━━━━━━━━━━━━━━━━━━━\n"
                for l in logs:
                    ts = l.get("timestamp", "")[:19].replace("T", " ")
                    act = l.get("action", "")
                    detail = l.get("details", "")
                    log_txt += f"• `{ts}`: *{act}*\n  {detail}\n"
            back_kb = InlineKeyboardMarkup([[InlineKeyboardButton("🔙 Back to Users", callback_data="btn_users_refresh")]])
            try:
                await query.edit_message_text(log_txt, parse_mode="Markdown", reply_markup=back_kb)
            except Exception:
                plain_log = log_txt.replace("*", "").replace("`", "").replace("_", "")
                await query.edit_message_text(plain_log, reply_markup=back_kb)
            return

        elif data == "btn_sessions":
            text = self.session_notifier.get_session_dashboard()
            await query.edit_message_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

        elif data == "btn_news":
            await self.news_engine.fetch_calendar()
            text = self.news_engine.get_news_dashboard()
            st_enabled = self.straddle_engine.is_enabled if self.straddle_engine else False
            await query.edit_message_text(
                text=text,
                reply_markup=news_menu_keyboard(st_enabled, self.news_engine.is_shield_enabled),
                parse_mode="Markdown"
            )

        elif data == "toggle_news_straddle":
            if self.straddle_engine:
                self.straddle_engine.toggle_enabled()
            st_enabled = self.straddle_engine.is_enabled if self.straddle_engine else False
            await query.edit_message_text(
                text=self.news_engine.get_news_dashboard(),
                reply_markup=news_menu_keyboard(st_enabled, self.news_engine.is_shield_enabled),
                parse_mode="Markdown"
            )

        elif data == "toggle_news_shield":
            self.news_engine.toggle_shield()
            st_enabled = self.straddle_engine.is_enabled if self.straddle_engine else False
            await query.edit_message_text(
                text=self.news_engine.get_news_dashboard(),
                reply_markup=news_menu_keyboard(st_enabled, self.news_engine.is_shield_enabled),
                parse_mode="Markdown"
            )

        elif data == "btn_news_refresh":
            await self.news_engine.fetch_calendar()
            st_enabled = self.straddle_engine.is_enabled if self.straddle_engine else False
            await query.edit_message_text(
                text=self.news_engine.get_news_dashboard(),
                reply_markup=news_menu_keyboard(st_enabled, self.news_engine.is_shield_enabled),
                parse_mode="Markdown"
            )

        elif data == "btn_close_all_confirm":
            await query.edit_message_text(
                "⚠️ *Confirm Emergency Close All*\nThis will close ALL open Marginal CFD/Forex positions immediately.",
                reply_markup=close_all_confirm_keyboard(),
                parse_mode="Markdown"
            )

        elif data == "action_close_all_execute":
            fx_bal = self.forex_mcp.get_training_balance() if self.account_type == "training" else self.forex_mcp.get_real_balance()
            closed_cnt = 0
            if fx_bal:
                positions = self.forex_mcp.list_positions(balance_id=fx_bal["balance_id"])
                for p in positions:
                    pos_id = p.get("position_id") or p.get("id")
                    if pos_id:
                        self.forex_mcp.close_position(position_id=pos_id)
                        await self._cleanup_closed_trade(pos_id, close_reason="client_close_all")
                        closed_cnt += 1
            await query.edit_message_text(
                f"🛑 Closed *{closed_cnt}* open position(s) across all engines.",
                reply_markup=main_menu_keyboard(),
                parse_mode="Markdown"
            )

        elif data.startswith("close_trade_"):
            pos_id_str = data.replace("close_trade_", "").strip()
            try:
                pos_id = int(pos_id_str)
                res = self.forex_mcp.close_position(position_id=pos_id)
                await self._cleanup_closed_trade(pos_id, close_reason="client_close")
                if res and not res.get("error") and not res.get("isError"):
                    msg = f"✅ Position *#{pos_id}* closed successfully on broker."
                else:
                    err_msg = res.get("error", {}).get("message", str(res)) if isinstance(res, dict) else str(res)
                    msg = f"⚠️ Close request processed for *#{pos_id}* ({err_msg})"
            except Exception as e:
                msg = f"❌ Error closing position #{pos_id_str}: {e}"

            await asyncio.sleep(0.8)
            text = self.build_active_setups_view()
            kb = active_setups_keyboard(is_live=True, open_positions=self.last_open_positions)
            try:
                await query.edit_message_text(
                    text=f"{msg}\n\n{text}",
                    reply_markup=kb,
                    parse_mode="Markdown"
                )
            except Exception:
                await query.edit_message_text(
                    text=f"{msg}\n\n{text}",
                    reply_markup=kb
                )
            self.start_active_view_refresh_task(query.message.chat_id, query.message.message_id)

    async def initialize(self):
        """Build and configure python-telegram-bot Application."""
        self.app = ApplicationBuilder().token(self.token).build()

        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("bal", self.cmd_balance))
        self.app.add_handler(CommandHandler("sessions", self.cmd_sessions))
        self.app.add_handler(CommandHandler("market", self.cmd_sessions))
        self.app.add_handler(CommandHandler("session", self.cmd_sessions))
        self.app.add_handler(CommandHandler("news", self.cmd_news))
        self.app.add_handler(CommandHandler("calendar", self.cmd_news))
        self.app.add_handler(CommandHandler("events", self.cmd_news))
        self.app.add_handler(CommandHandler("straddle", self.cmd_straddle))
        self.app.add_handler(CommandHandler("newsstraddle", self.cmd_straddle))
        self.app.add_handler(CommandHandler("spike", self.cmd_straddle))
        self.app.add_handler(CommandHandler("ict", self.cmd_ict))
        self.app.add_handler(CommandHandler("crt", self.cmd_crt))
        self.app.add_handler(CommandHandler("candlerange", self.cmd_crt))
        self.app.add_handler(CommandHandler("snd", self.cmd_snd))
        self.app.add_handler(CommandHandler("supplydemand", self.cmd_snd))
        self.app.add_handler(CommandHandler("trendline", self.cmd_trendline))
        self.app.add_handler(CommandHandler("trendlines", self.cmd_trendline))
        self.app.add_handler(CommandHandler("tl", self.cmd_trendline))
        self.app.add_handler(CommandHandler("mission", self.cmd_mission))
        self.app.add_handler(CommandHandler("strategies", self.cmd_mission))
        self.app.add_handler(CommandHandler("about", self.cmd_mission))
        self.app.add_handler(CommandHandler("channels", self.cmd_channels))
        self.app.add_handler(CommandHandler("copiers", self.cmd_channels))
        self.app.add_handler(CommandHandler("active", self.cmd_active_trades))
        self.app.add_handler(CommandHandler("trades", self.cmd_active_trades))
        self.app.add_handler(CommandHandler("setups", self.cmd_active_trades))
        self.app.add_handler(CommandHandler("settings", self.cmd_settings))
        self.app.add_handler(CommandHandler("risk", self.cmd_settings))
        self.app.add_handler(CommandHandler("lots", self.cmd_lots))
        self.app.add_handler(CommandHandler("lot", self.cmd_lots))
        self.app.add_handler(CommandHandler("set_lots", self.cmd_lots))
        self.app.add_handler(CommandHandler("leverage", self.cmd_leverage))
        self.app.add_handler(CommandHandler("lev", self.cmd_leverage))
        self.app.add_handler(CommandHandler("set_leverage", self.cmd_leverage))
        self.app.add_handler(CommandHandler("stake", self.cmd_stake))
        self.app.add_handler(CommandHandler("blitz", self.cmd_stake))
        self.app.add_handler(CommandHandler("set_stake", self.cmd_stake))
        self.app.add_handler(CommandHandler("account", self.cmd_account))
        self.app.add_handler(CommandHandler("acc", self.cmd_account))
        self.app.add_handler(CommandHandler("set_account", self.cmd_account))
        self.app.add_handler(CommandHandler("pause", self.cmd_pause))
        self.app.add_handler(CommandHandler("resume", self.cmd_resume))
        self.app.add_handler(CommandHandler("prop", self.cmd_prop))
        self.app.add_handler(CommandHandler("propfirm", self.cmd_prop))
        self.app.add_handler(CommandHandler("prop_reset", self.cmd_prop_reset))
        self.app.add_handler(CommandHandler("history", self.cmd_history))
        self.app.add_handler(CommandHandler("stats", self.cmd_history))
        self.app.add_handler(CommandHandler("clearhistory", self.cmd_clear_history))
        self.app.add_handler(CommandHandler("clear_history", self.cmd_clear_history))
        self.app.add_handler(CommandHandler("export", self.cmd_export))
        self.app.add_handler(CommandHandler("export_csv", self.cmd_export))
        self.app.add_handler(CommandHandler("close", self.cmd_close))
        self.app.add_handler(CommandHandler("closetrade", self.cmd_close))
        self.app.add_handler(CommandHandler("exit", self.cmd_close))
        self.app.add_handler(CommandHandler("closeall", self.cmd_closeall))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CommandHandler("users", self.cmd_users))
        self.app.add_handler(CommandHandler("user_alerts", self.cmd_toggle_user_alerts))
        self.app.add_handler(CommandHandler("toggle_alerts", self.cmd_toggle_user_alerts))
        self.app.add_handler(CommandHandler("useralerts", self.cmd_toggle_user_alerts))
        self.app.add_handler(CommandHandler("suspend", self.cmd_suspend))
        self.app.add_handler(CommandHandler("activate", self.cmd_activate))
        self.app.add_handler(CommandHandler("user", self.cmd_user))
        self.app.add_handler(CommandHandler("audit_logs", self.cmd_audit_logs))
        self.app.add_handler(CommandHandler("set_timezone", self.cmd_set_tz))
        self.app.add_handler(CommandHandler("timezone", self.cmd_set_tz))
        self.app.add_handler(CommandHandler("notifications", self.cmd_notifications))
        self.app.add_handler(CommandHandler("disclaimer", self.cmd_disclaimer))
        self.app.add_handler(CallbackQueryHandler(self.on_button_click))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_reply_button))

        # Wire notification callbacks
        self.channel_mgr.set_notification_callback(self.broadcast_alert)
        self.ict_engine.set_notification_callback(self.broadcast_alert)
        if self.straddle_engine:
            self.straddle_engine.set_notification_callback(self.broadcast_alert)
        if self.crt_engine:
            self.crt_engine.set_notification_callback(self.broadcast_alert)
        if self.snd_engine:
            self.snd_engine.set_notification_callback(self.broadcast_alert)
        if self.trendline_engine:
            self.trendline_engine.set_notification_callback(self.broadcast_alert)

    async def start(self):
        await self.initialize()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self.is_running = True
        logger.info("🤖 Telegram Bot UI active & listening for user commands!")
        logger.info("🚀 [Controller] BreakingBad V3 Ecosystem Online.")

        asyncio.create_task(self.session_notifier.run_loop())
        asyncio.create_task(self.news_engine.run_loop())
        if self.straddle_engine:
            asyncio.create_task(self.straddle_engine.run_loop())

        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        self.is_running = False
        self.stop_active_view_refresh_task()
        self.session_notifier.stop()
        self.news_engine.stop()
        if self.straddle_engine:
            self.straddle_engine.stop()
        try:
            if self.app:
                if self.app.updater and getattr(self.app.updater, "running", False):
                    await self.app.updater.stop()
                if getattr(self.app, "running", False):
                    await self.app.stop()
                    await self.app.shutdown()
                logger.info("Telegram Bot stopped.")
        except Exception as e:
            logger.warning(f"Error stopping Telegram Bot: {e}")
