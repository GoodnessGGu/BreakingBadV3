"""
bot/telegram_controller.py - Unified Telegram Bot Controller (@WalterAWbot)

Interactive command center for:
  - CallistoFx Zone Copier
  - Gold Pips Hunter Signal Copier
  - Polycarp VIP Room Blitz Options Copier (with 2-step Martingale)
  - Autonomous Gold / Multi-Instrument ICT Strategy Engine
"""

import os
import sys
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional

from telegram import Update
from telegram.ext import (
    Application, ApplicationBuilder, CommandHandler,
    CallbackQueryHandler, MessageHandler, ContextTypes, filters
)
from dotenv import load_dotenv

from copiers.channel_manager import ChannelManager
from strategies.ict_engine import ICTStrategyEngine, INSTRUMENT_PROFILES
from clients.forex_mcp_client import IQForexMCPClient
from clients.blitz_mcp_client import IQBlitzMCPClient
from bot.keyboards import (
    main_menu_keyboard, channels_menu_keyboard,
    ict_menu_keyboard, settings_menu_keyboard, close_all_confirm_keyboard,
    persistent_reply_keyboard, history_menu_keyboard, active_setups_keyboard,
    news_menu_keyboard, crt_menu_keyboard, mission_menu_keyboard,
    individual_close_keyboard, snd_menu_keyboard
)
from bot.session_notifier import MarketSessionNotifier
from bot.news_engine import EconomicNewsEngine
from strategies.news_straddle_engine import NewsStraddleEngine
from strategies.crt_engine import CRTStrategyEngine
from strategies.snd_engine import SNDStrategyEngine, SND_INSTRUMENT_PROFILES

logger = logging.getLogger("TelegramController")

class TelegramTradingBot:
    def __init__(self, token: str, admin_id: int, channel_mgr: ChannelManager,
                 ict_engine: ICTStrategyEngine, forex_mcp: IQForexMCPClient, blitz_mcp: IQBlitzMCPClient,
                 lots: float = 1.0, leverage: int = 100, blitz_stake: float = 2.0,
                 straddle_engine: Optional[NewsStraddleEngine] = None,
                 crt_engine: Optional[CRTStrategyEngine] = None,
                 snd_engine: Optional[SNDStrategyEngine] = None):
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
        """Send high-priority notification to the admin on Telegram with direct HTTP fallback."""
        if self.app and getattr(self.app, "bot", None):
            try:
                await self.app.bot.send_message(
                    chat_id=self.admin_id,
                    text=text
                )
                first_line = text.splitlines()[0] if text else ""
                logger.info(f"📢 [Broadcast Alert] Dispatched to Admin ({self.admin_id}): {first_line}")
                return
            except Exception as e:
                logger.warning(f"Error sending broadcast via app.bot: {e}")

        # Resilient fallback via direct Telegram HTTP API
        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                res = await client.post(
                    f"https://api.telegram.org/bot{self.token}/sendMessage",
                    json={"chat_id": self.admin_id, "text": text}
                )
                logger.info(f"📢 [Broadcast Alert Fallback] HTTP {res.status_code}")
        except Exception as e:
            logger.error(f"Failed to send direct Telegram broadcast alert: {e}")

    async def broadcast_photo(self, photo_bytes: bytes, caption: str = ""):
        """Send high-priority chart/image notification to the admin on Telegram."""
        if not photo_bytes:
            await self.broadcast_alert(caption)
            return

        if self.app and getattr(self.app, "bot", None):
            try:
                await self.app.bot.send_photo(
                    chat_id=self.admin_id,
                    photo=photo_bytes,
                    caption=caption
                )
                return
            except Exception as e:
                logger.warning(f"Error sending broadcast photo via app.bot: {e}")

        # Resilient fallback via direct Telegram HTTP API multipart upload
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
            logger.error(f"Failed to send direct Telegram photo broadcast: {e}")
            await self.broadcast_alert(caption)

    def build_status_text(self) -> str:
        # Fetch balances
        fx_bal = self.forex_mcp.get_training_balance() if self.account_type == "training" else self.forex_mcp.get_real_balance()
        fx_eq = fx_bal.get("equity", 0.0) if fx_bal else 0.0

        blitz_bal = self.blitz_mcp.get_training_balance() if self.account_type == "training" else self.blitz_mcp.get_real_balance()
        blitz_amt = blitz_bal.get("amount", 0.0) if blitz_bal else 0.0

        # Copiers
        copier_lines = []
        for c in self.channel_mgr.list_copiers():
            icon = "🟢" if c["enabled"] else "🔴"
            copier_lines.append(f"  • {icon} {c['name']}: `{'ON' if c['enabled'] else 'OFF'}`")

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
            f"📡 *Signal Copiers*:\n" + "\n".join(copier_lines) + "\n\n"
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
            f"🏛️ *Autonomous S&D Engine (Config E Forex)*:\n"
            f"  • {snd_icon} Master Switch: `{'ON' if snd_st.get('enabled') else 'OFF'}`\n"
            f"  • 🎯 Active Pairs: `{', '.join(snd_st.get('enabled_symbols', [])) if snd_st.get('enabled_symbols') else 'None'}`\n"
            f"  • 📊 Lots: `{snd_st.get('lots', self.lots):.2f}` | Lev: `{snd_st.get('leverage', self.leverage)}x`\n"
            f"  • ⚖️ Target R:R: `1:2.0` | Breakeven: `+1.0R`\n\n"
            f"⚡ *News Straddle Spike Engine*:\n"
            f"  • {straddle_icon} Status: `{'ON' if straddle_st.get('enabled') else 'OFF'}`{straddle_arm_icon}\n"
            f"  • 🎯 Asset: `XAUUSD (Gold)` | Lots: `{self.straddle_engine.lots if self.straddle_engine else self.lots:.2f}`\n\n"
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
            "📡 *5. Signal Copiers & Blitz Options*\n"
            "• *CFD Copiers*: CallistoFx, Gold Pips Hunter, GSociety, Kingmahn.\n"
            "• *Polycarp VIP*: High-frequency Blitz Binary Options with a controlled 2-step Martingale recovery buffer.\n\n"
            "🛡️ *Golden Rule*: Capital preservation first. Never trade without invalidation."
        )

    # ==========================================
    # Command Handlers
    # ==========================================

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Unauthorized access.")
            return

        # Show persistent keyboard buttons
        await update.message.reply_text(
            "👑 *BreakingBad V3 Bot is Online & Ready!*\nUse the buttons below for quick control:",
            reply_markup=persistent_reply_keyboard(),
            parse_mode="Markdown"
        )
        text = self.build_status_text()
        await update.message.reply_text(
            text=text,
            reply_markup=main_menu_keyboard(),
            parse_mode="Markdown"
        )

    async def cmd_status(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        text = self.build_status_text()
        await update.message.reply_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

    async def cmd_balance(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        fx_bal = self.forex_mcp.get_training_balance() if self.account_type == "training" else self.forex_mcp.get_real_balance()
        fx_eq = fx_bal.get("equity", 0.0) if fx_bal else 0.0
        fx_avail = fx_bal.get("available", fx_bal.get("balance", 0.0)) if fx_bal else 0.0

        blitz_bal = self.blitz_mcp.get_training_balance() if self.account_type == "training" else self.blitz_mcp.get_real_balance()
        blitz_amt = blitz_bal.get("amount", 0.0) if blitz_bal else 0.0

        mode_lbl = "🟡 PRACTICE (Training)" if self.account_type == "training" else "🔴 REAL MONEY (Regular)"
        msg = (
            f"💰 *Trading Account Balances*\n\n"
            f"💼 *Mode*: {mode_lbl}\n\n"
            f"📈 *Forex / CFD Account*:\n"
            f"  • Equity: `${fx_eq:,.2f}`\n"
            f"  • Available: `${fx_avail:,.2f}`\n\n"
            f"⚡ *Blitz Options Account*:\n"
            f"  • Balance: `${blitz_amt:,.2f}`\n\n"
            f"💡 Adjust sizing via ⚙️ Risk & Sizing or `/account <real/demo>`."
        )
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=persistent_reply_keyboard())

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
        try:
            aid = int(asset_id)
            if aid == 74:
                return "Gold (XAUUSD)"
            elif aid in [1487, 54]:
                return "Silver (XAGUSD)"
            elif aid == 816:
                return "Bitcoin (BTCUSD)"
            elif aid == 1:
                return "EURUSD"
            elif aid == 2:
                return "GBPUSD"
            elif aid == 1471:
                return "US100"
            elif aid == 1470:
                return "US500"
            elif aid == 971:
                return "Crude Oil (WTI)"
            elif aid == 969:
                return "Crude Oil (Brent)"
            for sym, prof in INSTRUMENT_PROFILES.items():
                if prof.get("asset_id") == aid:
                    return f"{sym}"
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
                                    if "callisto" in c_name.lower(): origin_tag = " [Callisto Fx]"
                                    elif "gold" in c_name.lower(): origin_tag = " [Gold Pips]"
                                    elif "gsociety" in c_name.lower() or "society" in c_name.lower(): origin_tag = " [G Society]"
                                    elif "king" in c_name.lower(): origin_tag = " [Kingmahn]"
                                    else: origin_tag = f" [{c_inst.name}]"
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

            # 4. Callisto Active Zones
            c_copier = self.channel_mgr.get_copier("callistofx")
            zones = getattr(c_copier, "active_zones", {}) if c_copier else {}
            if zones:
                zones_lines = []
                for s, z in zones.items():
                    live_p = self.get_live_price(s)
                    p_str = f" | Live: ${live_p:.2f}" if live_p > 0 else ""
                    zones_lines.append(f"  • *{s}*: {z['zone_low']:.2f} – {z['zone_high']:.2f}{p_str} (Target: {z.get('target', 'N/A')})")
                zones_txt = "\n".join(zones_lines)
            else:
                zones_txt = "  • No active zones currently watching."

            # 5. Polycarp Blitz Trades
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
                f"📍 *Callisto Active Zones*:\n{zones_txt}\n\n"
                f"⚡ *Polycarp Blitz Trades*:\n{blitz_txt}\n\n"
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
        if not self.is_admin(update.effective_user.id):
            return
        text = self.build_active_setups_view()
        kb = active_setups_keyboard(is_live=True, open_positions=self.last_open_positions)
        try:
            sent_msg = await update.message.reply_text(
                text,
                parse_mode="Markdown",
                reply_markup=kb
            )
            if sent_msg:
                self.start_active_view_refresh_task(update.effective_chat.id, sent_msg.message_id)
        except Exception:
            sent_msg = await update.message.reply_text(
                text,
                reply_markup=kb
            )
            if sent_msg:
                self.start_active_view_refresh_task(update.effective_chat.id, sent_msg.message_id)

    def get_active_balance_id(self) -> Optional[int]:
        try:
            bal = self.forex_mcp.get_real_balance() if self.account_type == "regular" else self.forex_mcp.get_training_balance()
            if bal:
                bid = bal.get("balance_id") or bal.get("id")
                if bid:
                    return int(bid)
        except Exception as e:
            logger.debug(f"[ActiveBalance] Error checking active balance: {e}")
        if self.ict_engine.balance_id:
            return self.ict_engine.balance_id
        return None

    def build_history_view(self, category: str = "all") -> str:
        """
        Fetches and compiles prettified trade history categorized by:
          - Blitz Options (Polycarp VIP)
          - Marginal CFD Copiers (Callisto / Gold Pips)
          - Autonomous ICT Engine (Gold / BTC / Forex)
        """
        # 1. Fetch Blitz Options History
        blitz_trades = []
        try:
            raw_blitz = self.blitz_mcp.get_trade_history(limit=30) or []
            for t in raw_blitz:
                pos_id = t.get("position_id") or t.get("id", "N/A")
                asset = t.get("asset_name") or t.get("active") or f"Asset {t.get('asset_id', '')}"
                asset = str(asset).replace("_", " ")
                res_str = str(t.get("result", "")).lower()
                profit = float(t.get("profit", 0.0))
                stake = float(t.get("amount") or t.get("invest") or 0.0)
                direction = str(t.get("direction") or t.get("type", "CALL")).upper()
                is_win = (res_str == "win" or profit > 0)
                is_equal = (res_str == "equal" or (profit == 0 and res_str not in ["loss", "loose"]))

                ts_raw = t.get("close_time") or t.get("open_time") or t.get("created_at")
                if isinstance(ts_raw, (int, float)):
                    ts_str = datetime.fromtimestamp(ts_raw).strftime("%H:%M")
                elif ts_raw:
                    ts_str = str(ts_raw)[11:16]
                else:
                    ts_str = "--:--"

                blitz_trades.append({
                    "id": pos_id,
                    "asset": asset,
                    "direction": direction,
                    "stake": stake,
                    "pnl": profit,
                    "is_win": is_win,
                    "is_equal": is_equal,
                    "time": ts_str
                })
        except Exception as e:
            logger.warning(f"[History] Error fetching Blitz history: {e}")

        # 2. Fetch Marginal CFD History (Forex, Gold, ICT)
        cfd_copier_trades = []
        ict_trades = []
        try:
            bid = self.get_active_balance_id()
            raw_cfd = self.forex_mcp.get_trade_history(balance_id=bid, limit=30) or []
            for t in raw_cfd:
                pos_id = t.get("position_id") or t.get("id", "N/A")
                asset_id = t.get("asset_id")
                asset_name = f"Asset #{asset_id}"
                if asset_id == 74:
                    asset_name = "Gold (XAUUSD)"
                elif asset_id == 816:
                    asset_name = "Bitcoin (BTCUSD)"
                else:
                    for sym, prof in INSTRUMENT_PROFILES.items():
                        if prof.get("asset_id") == asset_id:
                            asset_name = sym
                            break

                asset_name = str(asset_name).replace("_", " ")
                side = str(t.get("side", "BUY")).upper()
                lots = float(t.get("lots") or t.get("count") or 1.0)
                open_px = float(t.get("open_price", 0.0))
                close_px = float(t.get("close_price", 0.0))
                pnl = float(t.get("pnl") or t.get("profit") or 0.0)
                reason = str(t.get("close_reason", "closed")).replace("_", " ").title()

                ts_raw = t.get("close_time") or t.get("open_time")
                if isinstance(ts_raw, (int, float)):
                    ts_str = datetime.fromtimestamp(ts_raw).strftime("%H:%M")
                elif ts_raw:
                    ts_str = str(ts_raw)[11:16]
                else:
                    ts_str = "--:--"

                # Check if ICT trade vs Copier trade
                comment = str(t.get("comment", "")).lower()
                is_ict = "ict" in comment or ("ict" in asset_name.lower())

                item = {
                    "id": pos_id,
                    "asset": asset_name,
                    "side": side,
                    "lots": lots,
                    "open_price": open_px,
                    "close_price": close_px,
                    "pnl": pnl,
                    "reason": reason,
                    "is_win": pnl > 0,
                    "is_be": pnl == 0,
                    "time": ts_str
                }
                if is_ict:
                    ict_trades.append(item)
                else:
                    cfd_copier_trades.append(item)
        except Exception as e:
            logger.warning(f"[History] Error fetching CFD history: {e}")

        # Metrics calculation
        all_count = len(blitz_trades) + len(cfd_copier_trades) + len(ict_trades)
        total_pnl = sum(t["pnl"] for t in blitz_trades) + sum(t["pnl"] for t in cfd_copier_trades) + sum(t["pnl"] for t in ict_trades)
        total_wins = sum(1 for t in blitz_trades if t["is_win"]) + sum(1 for t in cfd_copier_trades if t["is_win"]) + sum(1 for t in ict_trades if t["is_win"])
        total_losses = sum(1 for t in blitz_trades if not t["is_win"] and not t["is_equal"]) + sum(1 for t in cfd_copier_trades if not t["is_win"] and not t["is_be"]) + sum(1 for t in ict_trades if not t["is_win"] and not t["is_be"])
        overall_wr = (total_wins / max(1, total_wins + total_losses)) * 100.0 if (total_wins + total_losses) > 0 else 0.0

        pnl_sign = "+" if total_pnl >= 0 else ""
        pnl_icon = "🟢" if total_pnl >= 0 else "🔴"

        # Build output message
        header = (
            f"📜 *Trading Execution & PnL History*\n\n"
            f"👤 *Account Mode*: `{self.account_type.upper()}`\n"
            f"{pnl_icon} *Total Realized PnL*: `{pnl_sign}${total_pnl:.2f}`\n"
            f"🎯 *Overall Win Rate*: `{overall_wr:.1f}%` ({total_wins}W - {total_losses}L)\n"
            f"📊 *Total Closed Trades*: `{all_count}`\n\n"
        )

        body = ""

        # Section 1: Blitz Options (Polycarp VIP)
        if category in ["all", "blitz"]:
            b_cnt = len(blitz_trades)
            b_wins = sum(1 for t in blitz_trades if t["is_win"])
            b_losses = sum(1 for t in blitz_trades if not t["is_win"] and not t["is_equal"])
            b_pnl = sum(t["pnl"] for t in blitz_trades)
            b_wr = (b_wins / max(1, b_wins + b_losses)) * 100.0 if (b_wins + b_losses) > 0 else 0.0
            b_sign = "+" if b_pnl >= 0 else ""

            body += (
                f"⚡ *Polycarp Blitz Options* (`{b_cnt}` Trades | `{b_sign}${b_pnl:.2f}` | `{b_wr:.0f}% WR`)\n"
            )
            if blitz_trades:
                for t in blitz_trades[:6]:
                    icon = "✅" if t["is_win"] else ("🛡️" if t["is_equal"] else "❌")
                    p_sign = "+" if t["pnl"] >= 0 else ""
                    body += f"  {icon} `{t['time']}` *{t['asset']}* {t['direction']} ➔ `{p_sign}${t['pnl']:.2f}`\n"
            else:
                body += "  • No recent Blitz trades found.\n"
            body += "\n"

        # Section 2: CFD Copiers (Callisto / Gold Pips)
        if category in ["all", "cfd"]:
            c_cnt = len(cfd_copier_trades)
            c_wins = sum(1 for t in cfd_copier_trades if t["is_win"])
            c_losses = sum(1 for t in cfd_copier_trades if not t["is_win"] and not t["is_be"])
            c_pnl = sum(t["pnl"] for t in cfd_copier_trades)
            c_wr = (c_wins / max(1, c_wins + c_losses)) * 100.0 if (c_wins + c_losses) > 0 else 0.0
            c_sign = "+" if c_pnl >= 0 else ""

            body += (
                f"📈 *Forex & Gold CFD Copiers* (`{c_cnt}` Trades | `{c_sign}${c_pnl:.2f}` | `{c_wr:.0f}% WR`)\n"
            )
            if cfd_copier_trades:
                for t in cfd_copier_trades[:6]:
                    icon = "🏆" if t["is_win"] else ("🛡️" if t["is_be"] else "❌")
                    p_sign = "+" if t["pnl"] >= 0 else ""
                    body += f"  {icon} `{t['time']}` *{t['asset']}* {t['side']} ➔ `{p_sign}${t['pnl']:.2f}` ({t['reason']})\n"
            else:
                body += "  • No recent CFD copier trades found.\n"
            body += "\n"

        # Section 3: Autonomous ICT Engine
        if category in ["all", "ict"]:
            i_cnt = len(ict_trades)
            i_wins = sum(1 for t in ict_trades if t["is_win"])
            i_losses = sum(1 for t in ict_trades if not t["is_win"] and not t["is_be"])
            i_pnl = sum(t["pnl"] for t in ict_trades)
            i_wr = (i_wins / max(1, i_wins + i_losses)) * 100.0 if (i_wins + i_losses) > 0 else 0.0
            i_sign = "+" if i_pnl >= 0 else ""

            body += (
                f"🤖 *Autonomous ICT Engine* (`{i_cnt}` Trades | `{i_sign}${i_pnl:.2f}` | `{i_wr:.0f}% WR`)\n"
            )
            if ict_trades:
                for t in ict_trades[:6]:
                    icon = "🏆" if t["is_win"] else ("🛡️" if t["is_be"] else "❌")
                    p_sign = "+" if t["pnl"] >= 0 else ""
                    body += f"  {icon} `{t['time']}` *{t['asset']}* {t['side']} ➔ `{p_sign}${t['pnl']:.2f}`\n"
            else:
                body += "  • No recent ICT autonomous trades found.\n"

        return header + body

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        cat = "all"
        if context.args and context.args[0].lower() in ["blitz", "cfd", "ict", "all"]:
            cat = context.args[0].lower()
        msg = self.build_history_view(cat)
        try:
            await update.message.reply_text(
                msg,
                parse_mode="Markdown",
                reply_markup=history_menu_keyboard(cat)
            )
        except Exception as e:
            logger.warning(f"Markdown error sending history: {e}. Retrying without markdown.")
            await update.message.reply_text(
                msg,
                reply_markup=history_menu_keyboard(cat)
            )

    async def cmd_account(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        if not context.args:
            await update.message.reply_text(
                f"💼 Current Account Mode: *{self.account_type.upper()}*\nUsage: `/account <training/regular>` or `/set_account <real/demo>`",
                parse_mode="Markdown",
                reply_markup=persistent_reply_keyboard()
            )
            return
        arg = context.args[0].lower()
        if arg in ["real", "regular", "live"]:
            target = "regular"
        elif arg in ["demo", "training", "practice"]:
            target = "training"
        else:
            await update.message.reply_text("❌ Invalid account mode. Use: `real` or `demo`", parse_mode="Markdown")
            return

        self.account_type = target
        bal_fx = self.forex_mcp.get_real_balance() if target == "regular" else self.forex_mcp.get_training_balance()
        if bal_fx:
            bid = bal_fx["balance_id"]
            self.ict_engine.set_balance(bid, target)
            if self.straddle_engine:
                self.straddle_engine.set_balance(bid, target)
            for c in self.channel_mgr.copiers.values():
                if hasattr(c, "set_balance"):
                    c.set_balance(bid, target)

        bal_blitz = self.blitz_mcp.get_real_balance() if target == "regular" else self.blitz_mcp.get_training_balance()
        if bal_blitz:
            p_copier = self.channel_mgr.get_copier("polycarpvip")
            if p_copier and hasattr(p_copier, "set_balance"):
                p_copier.set_balance(bal_blitz["balance_id"], target)

        lbl = "🔴 REAL MONEY (Regular)" if target == "regular" else "🟡 PRACTICE (Training)"
        await update.message.reply_text(
            f"✅ Switched to *{lbl}* Account mode across all engines.",
            parse_mode="Markdown",
            reply_markup=persistent_reply_keyboard()
        )

    async def cmd_help(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        msg = (
            "ℹ️ *BreakingBad V3 Command Center*\n\n"
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
            "• `/history` - View categorized trade & PnL history\n"
            "• `/pause` / `/resume` - Master trading pause/resume\n"
            "• `/close [id|sym]` - Interactively or directly close an open position\n"
            "• `/closeall` - Emergency close all open positions\n\n"
            "📊 *Sizing Commands:*\n"
            "• `/lots <val>` - Set global Forex lot size (e.g. `/lots 0.5`)\n"
            "• `/leverage <val>` - Set global leverage (e.g. `/leverage 100`)\n"
            "• `/stake <val>` - Set Blitz base stake (e.g. `/stake 5.0`)\n"
            "• `/account <real/demo>` - Switch account mode"
        )
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=persistent_reply_keyboard())

    async def cmd_sessions(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        text = self.session_notifier.get_session_dashboard()
        await update.message.reply_text(text, parse_mode="Markdown", reply_markup=persistent_reply_keyboard())

    async def cmd_news(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
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
        if not self.is_admin(update.effective_user.id):
            return

        text = update.message.text.strip()
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
        elif text in ["🎯 Mission & Strategies", "🎯 Mission", "Mission", "Strategies"]:
            await self.cmd_mission(update, context)
        elif text in ["📡 Channels", "Channels"]:
            await self.cmd_channels(update, context)
        elif text in ["⚙️ Risk & Sizing", "Risk & Sizing", "Settings", "⚙️ Settings"]:
            await self.cmd_settings(update, context)
        elif text in ["📋 Active Setups", "Active Setups", "Active", "📈 Active Trades"]:
            await self.cmd_active_trades(update, context)
        elif text in ["📜 History", "History", "📋 History", "Trade History", "📊 Stats"]:
            await self.cmd_history(update, context)
        elif text in ["⏸ Pause", "Pause"]:
            await self.cmd_pause(update, context)
        elif text in ["▶ Resume", "Resume"]:
            await self.cmd_resume(update, context)
        elif text in ["🛑 Close All", "Close All"]:
            await self.cmd_closeall(update, context)
        elif text in ["ℹ️ Help", "Help"]:
            await self.cmd_help(update, context)

    async def cmd_ict(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        ict_st = self.ict_engine.get_status()
        await update.message.reply_text(
            self.build_ict_dashboard(),
            reply_markup=ict_menu_keyboard(ict_st),
            parse_mode="Markdown"
        )

    async def cmd_crt(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        crt_st = self.crt_engine.get_status() if self.crt_engine else {"enabled": False, "enabled_symbols": []}
        await update.message.reply_text(
            "🕯️ *Autonomous Candle Range Theory (CRT) Engine*\nConfigure active Forex / Crypto instruments & sizing:",
            reply_markup=crt_menu_keyboard(crt_st),
            parse_mode="Markdown"
        )

    async def cmd_snd(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        snd_st = self.snd_engine.get_status() if self.snd_engine else {"enabled": False, "enabled_symbols": []}
        await update.message.reply_text(
            "🏛️ *Autonomous Supply & Demand + Imbalance Engine (Config E)*\n"
            "Trading top-performing pairs: NZD/USD (+27.0R), USD/JPY (+17.0R), AUD/USD (+15.0R), USD/CAD (+7.0R):\n\n"
            "Configure active pairs, lots, and leverage:",
            reply_markup=snd_menu_keyboard(snd_st),
            parse_mode="Markdown"
        )

    async def cmd_mission(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
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

        # 5. Fallback for manual broker positions not tracked in any engine
        if not handled:
            try:
                bid = self.get_active_balance_id()
                if bid:
                    matched = None
                    for attempt in range(3):
                        hist = self.forex_mcp.get_trade_history(balance_id=bid, limit=15) or []
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
                        pnl_str = f"+${pnl:.2f}" if pnl >= 0 else f"-${abs(pnl):.2f}"
                        if pnl > 0:
                            header_line = f"🏆 [TRADE WON] {sym} {pnl_str}"
                        elif pnl == 0:
                            header_line = f"🛡️ [TRADE BREAKEVEN] {sym} $0.00"
                        else:
                            header_line = f"❌ [TRADE CLOSED] {sym} {pnl_str}"

                        card = (
                            f"{header_line}\n\n"
                            f"• Side: {side} ({lots} Lots)\n"
                            f"• Entry: {entry_px:.2f}\n"
                            f"• Exit: {exit_px:.2f}\n"
                            f"• Net PnL: {pnl_str}\n"
                            f"• Reason: {r_name}"
                        )
                        await self.broadcast_alert(card)
            except Exception as e:
                logger.warning(f"Fallback closure card lookup error for #{pos_id}: {e}")

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
            return
        await update.message.reply_text(
            "⚠️ Are you sure you want to CLOSE ALL open positions?",
            reply_markup=close_all_confirm_keyboard()
        )

    # ==========================================
    # Callback Query Handler
    # ==========================================

    async def on_button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            pass

        if not self.is_admin(query.from_user.id):
            return

        data = query.data

        if not data.startswith("btn_active_trades"):
            self.stop_active_view_refresh_task()

        if data.startswith("noop"):
            return

        # Main Navigation
        if data == "btn_main_menu":
            text = self.build_status_text()
            await query.edit_message_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

        elif data == "btn_status":
            text = self.build_status_text()
            await query.edit_message_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

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

        # History Menu & Filtering
        elif data.startswith("history_cat_"):
            cat = data.replace("history_cat_", "")
            txt = self.build_history_view(cat)
            try:
                await query.edit_message_text(
                    txt,
                    reply_markup=history_menu_keyboard(cat),
                    parse_mode="Markdown"
                )
            except Exception as e:
                logger.warning(f"Markdown error in history button callback: {e}. Retrying without markdown.")
                await query.edit_message_text(
                    txt,
                    reply_markup=history_menu_keyboard(cat)
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
            self.account_type = "regular"
            bal_fx = self.forex_mcp.get_real_balance()
            if bal_fx:
                bid = bal_fx["balance_id"]
                self.ict_engine.set_balance(bid, "regular")
                if self.straddle_engine:
                    self.straddle_engine.set_balance(bid, "regular")
                for c in self.channel_mgr.copiers.values():
                    if hasattr(c, "set_balance"):
                        c.set_balance(bid, "regular")

            bal_blitz = self.blitz_mcp.get_real_balance()
            if bal_blitz:
                p_copier = self.channel_mgr.get_copier("polycarpvip")
                if p_copier and hasattr(p_copier, "set_balance"):
                    p_copier.set_balance(bal_blitz["balance_id"], "regular")

            await query.edit_message_text(
                "🚨 Switched to *REAL MONEY (Regular)* Account mode!",
                reply_markup=settings_menu_keyboard(self.account_type, self.lots, self.leverage, self.blitz_stake),
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
        self.app.add_handler(CommandHandler("history", self.cmd_history))
        self.app.add_handler(CommandHandler("stats", self.cmd_history))
        self.app.add_handler(CommandHandler("close", self.cmd_close))
        self.app.add_handler(CommandHandler("closetrade", self.cmd_close))
        self.app.add_handler(CommandHandler("exit", self.cmd_close))
        self.app.add_handler(CommandHandler("closeall", self.cmd_closeall))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
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

    async def start(self):
        await self.initialize()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self.is_running = True
        logger.info("🤖 Telegram Bot UI active & listening for user commands!")
        
        # Send startup notification to Admin
        try:
            startup_text = (
                "🚀 *BreakingBad V3 Ecosystem Online!*\n\n"
                + self.build_status_text()
            )
            await self.broadcast_alert(startup_text)
            logger.info("📢 [Controller] Startup message dispatched to Admin.")
        except Exception as e:
            logger.warning(f"Failed to broadcast startup alert: {e}")

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
