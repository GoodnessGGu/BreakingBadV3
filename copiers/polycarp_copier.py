"""
copiers/polycarp_copier.py - Polycarp VIP Room Blitz Options MCP Copier

Parses Polycarp VIP room binary/turbo signals and executes them as Blitz Options
via IQ Option's official Blitz MCP server (https://blitz-options.mcp.iqoption.com).
"""

import os
import re
import time
import logging
import asyncio
from datetime import datetime, timedelta
import pytz
from typing import Dict, Any, Optional, List
from copiers.base_copier import BaseCopier
from clients.blitz_mcp_client import IQBlitzMCPClient
from gsheet_logger import gsheet_logger

logger = logging.getLogger("PolycarpCopier")

POLYCARP_DEFAULT_CHANNELS = [-1002551711564, -1003457213931]

class PolycarpSignalParser:
    @staticmethod
    def is_signal_message(text: str) -> bool:
        u = text.upper()
        return ("TRADE:" in u or "NEW SIGNAL" in u) and ("TIMER:" in u or "DIRECTION:" in u)

    @staticmethod
    def parse_signal(text: str, default_tz: str = "Africa/Lagos") -> Optional[Dict[str, Any]]:
        """
        Parses Polycarp VIP room signal format:
        🔔 NEW SIGNAL!
        🎫 Trade: 🇦🇺 AUD/JPY 🇯🇵 (OTC)
        ⏳ Timer: 5 minutes
        ➡️ Entry: 12:36 PM
        📈 Direction: BUY 🟩
        """
        try:
            # 1. Extract Pair (supports OTC)
            trade_match = re.search(r'Trade:\s*.*?([A-Z]{3})/([A-Z]{3}).*?(\(OTC\)|OTC)\b', text, re.IGNORECASE)
            otc = False
            if trade_match:
                base = trade_match.group(1).upper()
                quote = trade_match.group(2).upper()
                otc = True
            else:
                trade_match = re.search(r'Trade:\s*.*?([A-Z]{3})/([A-Z]{3})', text, re.IGNORECASE)
                if not trade_match:
                    return None
                base = trade_match.group(1).upper()
                quote = trade_match.group(2).upper()

            pair = f"{base}/{quote} (OTC)" if otc else f"{base}/{quote}"

            # 2. Extract Timer (expiry)
            timer_m = re.search(r'Timer:\s*(\d+)\s*minute', text, re.IGNORECASE)
            expiry_mins = int(timer_m.group(1)) if timer_m else 5
            expiry_secs = expiry_mins * 60

            # 3. Extract Direction
            dir_m = re.search(r'Direction:\s*(BUY|SELL|CALL|PUT)', text, re.IGNORECASE)
            if not dir_m:
                return None
            raw_dir = dir_m.group(1).upper()
            direction = "call" if raw_dir in ["BUY", "CALL"] else "put"

            # 4. Extract Entry Time
            entry_time = None
            entry_m = re.search(r'Entry:\s*(\d{1,2}):(\d{2})\s*(AM|PM)', text, re.IGNORECASE)
            if entry_m:
                hour = int(entry_m.group(1))
                minute = int(entry_m.group(2))
                ampm = entry_m.group(3).upper()

                if ampm == "PM" and hour < 12:
                    hour += 12
                elif ampm == "AM" and hour == 12:
                    hour = 0

                try:
                    tz = pytz.timezone(default_tz)
                except Exception:
                    tz = pytz.timezone("Africa/Lagos")

                now_tz = datetime.now(tz)
                entry_dt = now_tz.replace(hour=hour, minute=minute, second=0, microsecond=0)
                if (now_tz - entry_dt).total_seconds() > 43200:
                    entry_dt += timedelta(days=1)
                entry_time = entry_dt

            return {
                "pair": pair,
                "direction": direction,
                "expiry_mins": expiry_mins,
                "expiry_secs": expiry_secs,
                "entry_time": entry_time,
                "raw": text
            }
        except Exception as e:
            logger.error(f"[Polycarp] Parser error: {e}")
            return None

class PolycarpCopier(BaseCopier):
    def __init__(self, blitz_mcp: IQBlitzMCPClient, channel_id: int = -1002551711564,
                 stake_amount: float = 2.0, enabled: bool = True):
        super().__init__("PolycarpVIP", channel_id, enabled)
        self.blitz = blitz_mcp
        self.stake_amount = stake_amount

        self.balance_id: Optional[int] = None
        self.account_type = "training"
        self.open_trades: Dict[int, Dict] = {}
        self.processed_msg_ids = set()

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type

    async def handle_message(self, text: str, message_id: int, event: Any = None):
        if not self.is_enabled:
            return
        if message_id in self.processed_msg_ids:
            return
        self.processed_msg_ids.add(message_id)

        if not PolycarpSignalParser.is_signal_message(text):
            return

        sig = PolycarpSignalParser.parse_signal(text)
        if not sig:
            return

        logger.info(f"📨 [Polycarp] Parsed Signal: {sig['pair']} {sig['direction'].upper()} ({sig['expiry_mins']}m)")
        asyncio.create_task(self.schedule_and_execute(sig))

    async def schedule_and_execute(self, sig: Dict[str, Any]):
        pair = sig["pair"]
        direction = sig["direction"]
        expiry_secs = sig["expiry_secs"]
        entry_time = sig.get("entry_time")

        # 1. Handle scheduling if entry time is specified
        if entry_time:
            now_tz = datetime.now(entry_time.tzinfo)
            delay = (entry_time - now_tz).total_seconds()

            if delay > 0 and delay <= 900:
                logger.info(f"⏳ [Polycarp] Signal scheduled in {int(delay)}s at {entry_time.strftime('%H:%M:%S')}")
                await self.notify(
                    f"⏳ [Polycarp VIP SIGNAL SCHEDULED]\n"
                    f"Pair     : {pair}\n"
                    f"Direction: {direction.upper()}\n"
                    f"Expiry   : {sig['expiry_mins']} min\n"
                    f"Entry in : {int(delay)}s ({entry_time.strftime('%H:%M:%S')})"
                )
                await asyncio.sleep(delay)
            elif delay < -120:
                logger.warning(f"⏰ [Polycarp] Signal arrived too late ({int(-delay)}s past entry). Skipping.")
                await self.notify(f"⚠️ [Polycarp] Skipped expired signal ({pair} was {int(-delay)}s ago).")
                return

        # 2. Match Asset in Blitz MCP
        asset = self.blitz.find_asset(pair)
        if not asset:
            logger.error(f"❌ [Polycarp] Asset not found in Blitz MCP for '{pair}'")
            await self.notify(f"❌ [Polycarp] Asset '{pair}' not found on Blitz Options engine.")
            return

        asset_id = asset.get("asset_id")
        profit_percent = asset.get("profit_percent", 80)
        is_open = asset.get("is_open", False)
        avail_expirations = asset.get("expiration_sizes_seconds", [300])

        if not is_open:
            logger.warning(f"⚠️ [Polycarp] Asset {pair} (ID: {asset_id}) is currently CLOSED on Blitz options.")
            await self.notify(f"⚠️ [Polycarp] Asset {pair} is currently CLOSED for trading.")
            return

        # Match closest expiration
        chosen_exp = expiry_secs
        if chosen_exp not in avail_expirations:
            # find closest available
            chosen_exp = min(avail_expirations, key=lambda x: abs(x - expiry_secs))
            logger.info(f"[Polycarp] Adjusted expiration from {expiry_secs}s to {chosen_exp}s based on asset limits.")

        # 3. Place Trade on Blitz MCP
        logger.info(f"🚀 [Polycarp] Executing Blitz trade: {pair} (ID: {asset_id}) {direction.upper()} ${self.stake_amount} ({chosen_exp}s)")
        await self.notify(
            f"⚡ [Polycarp VIP EXECUTING BLITZ TRADE]\n"
            f"Asset    : {asset.get('name')}\n"
            f"Direction: {direction.upper()}\n"
            f"Stake    : ${self.stake_amount:.2f}\n"
            f"Payout   : {profit_percent}%\n"
            f"Duration : {chosen_exp}s"
        )

        res = self.blitz.place_trade(
            balance_id=self.balance_id,
            asset_id=asset_id,
            direction=direction,
            amount=self.stake_amount,
            profit_percent=profit_percent,
            expiration_size=chosen_exp
        )

        if "position_id" in res:
            pos_id = res["position_id"]
            logger.info(f"✅ [Polycarp] Blitz Position Opened! ID: #{pos_id}")
            self.open_trades[pos_id] = {
                "position_id": pos_id,
                "asset_id": asset_id,
                "pair": pair,
                "direction": direction,
                "amount": self.stake_amount,
                "profit_percent": profit_percent,
                "expiration_size": chosen_exp,
                "opened_at": time.time()
            }
            await self.notify(f"✅ [Polycarp] Blitz Position Opened! ID: #{pos_id}")
            asyncio.create_task(self.monitor_settlement(pos_id, chosen_exp))
        else:
            logger.error(f"❌ [Polycarp] Blitz trade failed: {res}")
            await self.notify(f"❌ [Polycarp] Blitz Trade Failed: {res.get('error', res)}")

    async def monitor_settlement(self, pos_id: int, exp_secs: int):
        """Wait for position expiration and log final result."""
        await asyncio.sleep(exp_secs + 6) # wait until expired + buffer
        try:
            history = self.blitz.get_trade_history(limit=10)
            trade = next((h for h in history if h.get("position_id") == pos_id), None)

            trade_info = self.open_trades.pop(pos_id, {})
            pair = trade_info.get("pair", "Blitz Option")
            direction = trade_info.get("direction", "").upper()
            stake = trade_info.get("amount", self.stake_amount)

            if trade:
                win = trade.get("is_win", False)
                profit = float(trade.get("profit", 0.0))
                pnl = profit - stake if win else -stake
                status_emoji = "🏆 WIN" if win else "❌ LOSS"
                await self.notify(
                    f"{status_emoji} [Polycarp VIP SETTLED]\n"
                    f"Position : #{pos_id} ({pair} {direction})\n"
                    f"Outcome  : {'WIN' if win else 'LOSS'}\n"
                    f"PnL      : {'+' if pnl >= 0 else ''}${pnl:.2f}"
                )
            else:
                logger.info(f"[Polycarp] Position #{pos_id} settled.")
        except Exception as e:
            logger.error(f"[Polycarp] Settlement check error: {e}")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "channel_id": self.channel_id,
            "stake_amount": self.stake_amount,
            "open_trades_count": len(self.open_trades)
        }
