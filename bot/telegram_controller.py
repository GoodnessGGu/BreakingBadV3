"""
bot/telegram_controller.py - Unified Telegram Bot Controller (@WalterAWbot)

Interactive command center for:
  - CallistoFx Zone Copier
  - Gold Pips Hunter Signal Copier
  - Polycarp VIP Room Blitz Options Copier
  - Multi-Instrument Autonomous ICT Strategy Engine
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
    CallbackQueryHandler, ContextTypes
)
from dotenv import load_dotenv

from copiers.channel_manager import ChannelManager
from strategies.ict_engine import ICTStrategyEngine
from clients.forex_mcp_client import IQForexMCPClient
from clients.blitz_mcp_client import IQBlitzMCPClient
from bot.keyboards import (
    main_menu_keyboard, channels_menu_keyboard,
    ict_menu_keyboard, account_menu_keyboard, close_all_confirm_keyboard
)

logger = logging.getLogger("TelegramController")

class TelegramTradingBot:
    def __init__(self, token: str, admin_id: int, channel_mgr: ChannelManager,
                 ict_engine: ICTStrategyEngine, forex_mcp: IQForexMCPClient, blitz_mcp: IQBlitzMCPClient):
        self.token = token
        self.admin_id = int(admin_id)
        self.channel_mgr = channel_mgr
        self.ict_engine = ict_engine
        self.forex_mcp = forex_mcp
        self.blitz_mcp = blitz_mcp

        self.account_type = "training"
        self.app: Optional[Application] = None

    def is_admin(self, user_id: int) -> bool:
        return int(user_id) == self.admin_id

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
            copier_lines.append(f"  {icon} {c['name']}: {'ON' if c['enabled'] else 'OFF'}")

        # ICT Engine
        ict_st = self.ict_engine.get_status()
        ict_icon = "🟢" if ict_st["enabled"] else "🔴"

        text = (
            f"👑 *BreakingBad V3 — Trading Control Center*\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"👤 *Account Mode*: `{self.account_type.upper()}`\n"
            f"💵 *Forex/CFD Equity*: `${fx_eq:.2f}`\n"
            f"⚡ *Blitz Options Balance*: `${blitz_amt:.2f}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"📡 *Signal Copiers*:\n" + "\n".join(copier_lines) + "\n\n"
            f"🤖 *Autonomous ICT Engine*:\n"
            f"  {ict_icon} Status: `{'ON' if ict_st['enabled'] else 'OFF'}`\n"
            f"  🎯 Instrument: `{ict_st['symbol']}` ({ict_st['name']})\n"
            f"  ⚖️ Risk/Reward: `1:{ict_st['rr_ratio']:.1f}` | Lots: `{ict_st['lots']}`\n"
            f"━━━━━━━━━━━━━━━━━━━━\n"
            f"🕒 Time: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`"
        )
        return text

    async def cmd_start(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            await update.message.reply_text("⛔ Unauthorized access.")
            return

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

    async def cmd_closeall(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        if not self.is_admin(update.effective_user.id):
            return
        await update.message.reply_text(
            "⚠️ Are you sure you want to CLOSE ALL open positions?",
            reply_markup=close_all_confirm_keyboard()
        )

    async def on_button_click(self, update: Update, context: ContextTypes.DEFAULT_TYPE):
        query = update.callback_query
        try:
            await query.answer()
        except Exception:
            pass

        if not self.is_admin(query.from_user.id):
            return

        data = query.data

        if data == "btn_main_menu":
            text = self.build_status_text()
            await query.edit_message_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

        elif data == "btn_status":
            text = self.build_status_text()
            await query.edit_message_text(text=text, reply_markup=main_menu_keyboard(), parse_mode="Markdown")

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

        elif data == "btn_ict_menu":
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous ICT / SMC Strategy Engine*\nConfigure active instrument and strategy settings:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data == "toggle_ict_engine":
            self.ict_engine.toggle()
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                "🤖 *Autonomous ICT / SMC Strategy Engine*\nConfigure active instrument and strategy settings:",
                reply_markup=ict_menu_keyboard(ict_st),
                parse_mode="Markdown"
            )

        elif data.startswith("set_inst_"):
            symbol = data.replace("set_inst_", "").upper()
            self.ict_engine.set_instrument(symbol)
            ict_st = self.ict_engine.get_status()
            await query.edit_message_text(
                f"🤖 Active instrument set to: *{ict_st['symbol']}* ({ict_st['name']})",
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

        elif data == "btn_account_menu":
            await query.edit_message_text(
                f"⚙️ *Account Settings*\nCurrent Mode: `{self.account_type.upper()}`\nSelect account balance type:",
                reply_markup=account_menu_keyboard(self.account_type),
                parse_mode="Markdown"
            )

        elif data == "set_acc_training":
            self.account_type = "training"
            # Update all components
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
                reply_markup=account_menu_keyboard(self.account_type),
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
                reply_markup=account_menu_keyboard(self.account_type),
                parse_mode="Markdown"
            )

        elif data == "btn_active_trades":
            # List active Callisto zones & ICT setup
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
        self.app.add_handler(CommandHandler("closeall", self.cmd_closeall))
        self.app.add_handler(CallbackQueryHandler(self.on_button_click))

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
