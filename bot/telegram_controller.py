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
    persistent_reply_keyboard, history_menu_keyboard
)

logger = logging.getLogger("TelegramController")

class TelegramTradingBot:
    def __init__(self, token: str, admin_id: int, channel_mgr: ChannelManager,
                 ict_engine: ICTStrategyEngine, forex_mcp: IQForexMCPClient, blitz_mcp: IQBlitzMCPClient,
                 lots: float = 1.0, leverage: int = 100, blitz_stake: float = 2.0):
        self.token = token
        self.admin_id = int(admin_id)
        self.channel_mgr = channel_mgr
        self.ict_engine = ict_engine
        self.forex_mcp = forex_mcp
        self.blitz_mcp = blitz_mcp

        self.account_type = "training"
        self.lots = float(lots)
        self.leverage = int(leverage)
        self.blitz_stake = float(blitz_stake)
        self.is_paused = False
        self.app: Optional[Application] = None

    def is_admin(self, user_id: int) -> bool:
        return int(user_id) == self.admin_id

    def update_forex_lots(self, lots: float) -> float:
        """Update lot size for ICT Engine and Forex/CFD copiers."""
        clean_lots = max(0.01, round(float(lots), 2))
        self.lots = clean_lots
        self.ict_engine.set_lots(clean_lots)
        for c in self.channel_mgr.copiers.values():
            if hasattr(c, "set_lots"):
                c.set_lots(clean_lots)
        logger.info(f"📊 [Controller] Updated Forex/Gold lot size to: {self.lots}")
        return self.lots

    def update_forex_leverage(self, leverage: int) -> int:
        """Update leverage for ICT Engine and Forex/CFD copiers."""
        clean_lev = int(leverage)
        self.leverage = clean_lev
        self.ict_engine.set_leverage(clean_lev)
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
        """Send high-priority notification to the admin on Telegram."""
        if not self.app:
            return
        try:
            await self.app.bot.send_message(
                chat_id=self.admin_id,
                text=text
            )
        except Exception as e:
            logger.warning(f"Error sending broadcast alert: {e}")

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
            copier_lines.append(f"  {icon} {c['name']}: `{'ON' if c['enabled'] else 'OFF'}`")

        # ICT Engine
        ict_st = self.ict_engine.get_status()
        ict_icon = "🟢" if ict_st["enabled"] else "🔴"
        pause_tag = " [PAUSED]" if self.is_paused else ""

        text = (
            f"👑 *BreakingBad V3 — Trading Control Center*{pause_tag}\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Account Mode*: `{self.account_type.upper()}`\n"
            f"💵 *Forex/CFD Equity*: `${fx_eq:.2f}`\n"
            f"⚡ *Blitz Options Balance*: `${blitz_amt:.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"⚙️ *Risk & Order Sizing*:\n"
            f"  📊 Forex / Gold Lots: `{self.lots:.2f}` | Lev: `{self.leverage}x`\n"
            f"  ⚡ Blitz Base Stake: `${self.blitz_stake:.2f}` *(2-Step Martingale)*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📡 *Signal Copiers*:\n" + "\n".join(copier_lines) + "\n\n"
            f"🤖 *Autonomous ICT Engine*:\n"
            f"  {ict_icon} Master Switch: `{'ON' if ict_st['enabled'] else 'OFF'}`\n"
            f"  🎯 Active Assets: `{', '.join(ict_st['enabled_symbols']) if ict_st['enabled_symbols'] else 'None'}`\n"
            f"  📊 Lots: `{ict_st['lots']:.2f}` | Lev: `{ict_st['leverage']}x`\n"
            f"  ⚖️ Risk/Reward: `1:{ict_st['rr_ratio']:.1f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return text

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
            f"💰 *Trading Account Balances*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"💼 *Mode*: {mode_lbl}\n\n"
            f"📈 *Forex / CFD Account*:\n"
            f"  • Equity: `${fx_eq:,.2f}`\n"
            f"  • Available: `${fx_avail:,.2f}`\n\n"
            f"⚡ *Blitz Options Account*:\n"
            f"  • Balance: `${blitz_amt:,.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
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

    async def cmd_active_trades(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        c_copier = self.channel_mgr.get_copier("callistofx")
        zones_txt = "None"
        if c_copier and getattr(c_copier, "active_zones", None):
            zones_txt = "\n".join([f"  • {s}: `{z['zone_low']:.2f} – {z['zone_high']:.2f}` (Target: {z.get('target', 'N/A')})" for s, z in c_copier.active_zones.items()])

        p_copier = self.channel_mgr.get_copier("polycarpvip")
        blitz_open_cnt = len(getattr(p_copier, "open_trades", {}))

        # ICT Setups & Trades across all assets
        ict_fvgs = self.ict_engine.pending_fvgs
        fvg_lines = [f"  • {s}: {f['side']} `[{f['fvg_low']} - {f['fvg_high']}]` (SL: {f['sl']})" for s, f in ict_fvgs.items() if f]
        ict_fvg_txt = "\n".join(fvg_lines) if fvg_lines else "None"

        ict_trades = self.ict_engine.active_trades
        trade_lines = [f"  • {s}: {t['side']} @ `{t['entry_price']}` (SL: `{t['current_sl']}`, TP: `{t['tp']}`)" for s, t in ict_trades.items() if t]
        ict_trade_txt = "\n".join(trade_lines) if trade_lines else "None"

        text = (
            f"📋 *Active Watchers, Zones & Setups*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📍 *Callisto Active Zones*:\n{zones_txt}\n\n"
            f"⚡ *Polycarp Blitz Trades*: `{blitz_open_cnt}` active\n\n"
            f"🔥 *ICT Pending FVGs*:\n{ict_fvg_txt}\n\n"
            f"🎯 *ICT Active Positions*:\n{ict_trade_txt}\n"
            f"━━━━━━━━━━━━━━━━━━━━"
        )
    def get_active_balance_id(self) -> Optional[int]:
        if self.ict_engine.balance_id:
            return self.ict_engine.balance_id
        bal = self.forex_mcp.get_real_balance() if self.account_type == "regular" else self.forex_mcp.get_training_balance()
        return bal.get("balance_id") if bal else None

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

                side = str(t.get("side", "BUY")).upper()
                lots = float(t.get("lots") or t.get("count") or 1.0)
                open_px = float(t.get("open_price", 0.0))
                close_px = float(t.get("close_price", 0.0))
                pnl = float(t.get("pnl") or t.get("profit") or 0.0)
                reason = str(t.get("close_reason", "closed"))

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
            f"📜 *Trading Execution & PnL History*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Account Mode*: `{self.account_type.upper()}`\n"
            f"{pnl_icon} *Total Realized PnL*: `{pnl_sign}${total_pnl:.2f}`\n"
            f"🎯 *Overall Win Rate*: `{overall_wr:.1f}%` ({total_wins}W - {total_losses}L)\n"
            f"📊 *Total Closed Trades*: `{all_count}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n\n"
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
                body += "  _No recent Blitz trades found._\n"
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
                body += "  _No recent CFD copier trades found._\n"
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
                body += "  _No recent ICT autonomous trades found._\n"

        return header + body

    async def cmd_history(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        cat = "all"
        if context.args and context.args[0].lower() in ["blitz", "cfd", "ict", "all"]:
            cat = context.args[0].lower()
        msg = self.build_history_view(cat)
        await update.message.reply_text(
            msg,
            parse_mode="Markdown",
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
            "Use the bottom keyboard for one-tap balance, status, setups, and pause/resume.\n\n"
            "⚡ *Core Commands:*\n"
            "• `/status` - Complete bot and account status\n"
            "• `/balance` - View Forex and Blitz balances\n"
            "• `/ict` - Autonomous Gold ICT engine controls\n"
            "• `/channels` - Toggle signal copier channels\n"
            "• `/settings` or `/risk` - Risk & order sizing menu\n"
            "• `/active` - Active zones, FVGs, and open trades\n"
            "• `/history` - View categorized trade & PnL history\n"
            "• `/pause` / `/resume` - Master trading pause/resume\n"
            "• `/closeall` - Emergency close all open positions\n\n"
            "📊 *Sizing Commands:*\n"
            "• `/lots <val>` - Set global Forex lot size (e.g. `/lots 0.5`)\n"
            "• `/leverage <val>` - Set global leverage (e.g. `/leverage 100`)\n"
            "• `/stake <val>` - Set Blitz base stake (e.g. `/stake 5.0`)\n"
            "• `/account <real/demo>` - Switch account mode"
        )
        await update.message.reply_text(msg, parse_mode="Markdown", reply_markup=persistent_reply_keyboard())

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
        elif text in ["🤖 Gold ICT", "Gold ICT", "ICT"]:
            await self.cmd_ict(update, context)
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
            "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nAdjust instrument, lot size, leverage, and RR ratio:",
            reply_markup=ict_menu_keyboard(ict_st),
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
            await query.edit_message_text(
                txt,
                reply_markup=history_menu_keyboard(cat),
                parse_mode="Markdown"
            )

        # ICT Engine Menu
        elif data == "btn_ict_menu":
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nConfigure active instrument and strategy settings:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_ict_engine":
            self.ict_engine.toggle()
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nConfigure active instrument and strategy settings:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("toggle_inst_") or data.startswith("set_inst_"):
            sym_raw = data.replace("toggle_inst_", "").replace("set_inst_", "").upper()
            self.ict_engine.toggle_symbol(sym_raw)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Multi-Asset ICT Strategy Engine*\nToggle assets ON/OFF or adjust execution settings:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "ict_lots_minus":
            new_l = max(0.01, round(self.ict_engine.lots - 0.1, 2))
            self.ict_engine.set_lots(new_l)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nAdjust instrument, lot size, leverage, and RR ratio:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "ict_lots_plus":
            new_l = round(self.ict_engine.lots + 0.1, 2)
            self.ict_engine.set_lots(new_l)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nAdjust instrument, lot size, leverage, and RR ratio:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_ict_lots_"):
            val = float(data.replace("set_ict_lots_", ""))
            self.ict_engine.set_lots(val)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nAdjust instrument, lot size, leverage, and RR ratio:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_ict_lev_"):
            val = int(data.replace("set_ict_lev_", ""))
            self.ict_engine.set_leverage(val)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous Gold & Forex ICT Strategy Engine*\nAdjust instrument, lot size, leverage, and RR ratio:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_rr_"):
            rr = float(data.replace("set_rr_", ""))
            self.ict_engine.rr_ratio = rr
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                f"🤖 Risk/Reward ratio set to: *1:{rr:.1f}*",
                reply_markup=ict_menu_keyboard(ict_st),
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

        elif data == "btn_active_trades":
            c_copier = self.channel_mgr.get_copier("callistofx")
            zones_txt = "None"
            if c_copier and getattr(c_copier, "active_zones", None):
                zones_txt = ", ".join([f"{s} [{z['zone_low']} - {z['zone_high']}]" for s, z in c_copier.active_zones.items()])

            ict_fvg = self.ict_engine.pending_fvg
            ict_txt = f"{ict_fvg['side']} [{ict_fvg['fvg_low']} - {ict_fvg['fvg_high']}]" if ict_fvg else "None"

            text = (
                f"📋 *Active Watchers & Setups*\n"
                f"━━━━━━━━━━━━━━━━━━━━\n"
                f"📍 *Callisto Active Zones*: {zones_txt}\n"
                f"🔥 *ICT Pending FVG*: {ict_txt}\n"
                f"⚡ *ICT Active Trade*: {'Active' if self.ict_engine.active_trade else 'None'}\n"
            )
            await query.edit_message_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

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
                        closed_cnt += 1
            await query.edit_message_text(
                f"🛑 Closed *{closed_cnt}* open position(s) across all engines.",
                reply_markup=main_menu_keyboard(),
                parse_mode="Markdown"
            )

    async def initialize(self):
        """Build and configure python-telegram-bot Application."""
        self.app = ApplicationBuilder().token(self.token).build()

        self.app.add_handler(CommandHandler("start", self.cmd_start))
        self.app.add_handler(CommandHandler("menu", self.cmd_start))
        self.app.add_handler(CommandHandler("status", self.cmd_status))
        self.app.add_handler(CommandHandler("balance", self.cmd_balance))
        self.app.add_handler(CommandHandler("bal", self.cmd_balance))
        self.app.add_handler(CommandHandler("ict", self.cmd_ict))
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
        self.app.add_handler(CommandHandler("closeall", self.cmd_closeall))
        self.app.add_handler(CommandHandler("help", self.cmd_help))
        self.app.add_handler(CallbackQueryHandler(self.on_button_click))
        self.app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, self.handle_reply_button))

        # Wire notification callbacks
        self.channel_mgr.set_notification_callback(self.broadcast_alert)
        self.ict_engine.set_notification_callback(self.broadcast_alert)

    async def start(self):
        await self.initialize()
        await self.app.initialize()
        await self.app.start()
        await self.app.updater.start_polling(drop_pending_updates=True)
        self.is_running = True
        logger.info("🤖 Telegram Bot UI active & listening for user commands!")
        try:
            while self.is_running:
                await asyncio.sleep(1)
        except asyncio.CancelledError:
            pass

    async def stop(self):
        self.is_running = False
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
