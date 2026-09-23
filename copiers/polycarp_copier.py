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
    CURRENCIES = r'(?:EUR|USD|GBP|JPY|AUD|NZD|CAD|CHF|THB|BRL|TRY|PLN|IDR|SAR|VND|MXN|COP)'

    @staticmethod
    def is_signal_message(text: str) -> bool:
        u = text.upper()
        has_dir = any(k in u for k in ["CALL", "PUT", "BUY", "SELL"])
        has_curr = bool(re.search(PolycarpSignalParser.CURRENCIES, u))
        return has_dir and has_curr

    @staticmethod
    def parse_signal(text: str, default_tz: str = "Africa/Lagos") -> Optional[Dict[str, Any]]:
        """
        Parses all Polycarp VIP room signal formats:
        - Standard format (Trade: AUD/JPY (OTC), Timer: 5 min, Direction: BUY)
        - Uplivon Special format (NZD/CAD OTC, Timeframe: 5 MIN, Direction: CALL)
        - Quick alert format (EUR/USD, 5m, CALL)
        """
        try:
            curr_pat = PolycarpSignalParser.CURRENCIES
            # 1. Extract Pair
            m = re.search(rf'({curr_pat})\s*/?\s*({curr_pat})', text, re.IGNORECASE)
            if not m:
                return None

            base, quote = m.group(1).upper(), m.group(2).upper()
            surrounding = text[max(0, m.start() - 10):min(len(text), m.end() + 20)].upper()
            otc = "OTC" in surrounding
            pair = f"{base}/{quote} (OTC)" if otc else f"{base}/{quote}"

            # 2. Extract Direction
            dir_m = re.search(r'(?:Direction:|Action:|Signal:|\b)(BUY|SELL|CALL|PUT)\b', text, re.IGNORECASE)
            if not dir_m:
                return None
            raw_dir = dir_m.group(1).upper()
            direction = "call" if raw_dir in ["BUY", "CALL"] else "put"

            # 3. Extract Timer / Expiry
            timer_m = re.search(r'(?:Timer:|Timeframe:|⏱️|⏳)?\s*(\d+)\s*(?:min|minute|m\b)', text, re.IGNORECASE)
            expiry_mins = int(timer_m.group(1)) if timer_m else 5
            expiry_secs = expiry_mins * 60

            # 4. Extract Entry Time (12h or 24h)
            entry_time = None
            try:
                tz = pytz.timezone(default_tz)
            except Exception:
                tz = pytz.timezone("Africa/Lagos")
            now_tz = datetime.now(tz)

            # 12-hour format e.g. 12:36 PM
            m12 = re.search(r'Entry:\s*(\d{1,2}):(\d{2})\s*(AM|PM)', text, re.IGNORECASE)
            if m12:
                hr, mn, ap = int(m12.group(1)), int(m12.group(2)), m12.group(3).upper()
                if ap == "PM" and hr < 12: hr += 12
                elif ap == "AM" and hr == 12: hr = 0
                res = now_tz.replace(hour=hr, minute=mn, second=0, microsecond=0)
                if (now_tz - res).total_seconds() > 43200: res += timedelta(days=1)
                entry_time = res
            else:
                # 24-hour format e.g. 15:38
                m24 = re.search(r'Entry:\s*(\d{1,2}):(\d{2})\b', text, re.IGNORECASE)
                if m24:
                    hr, mn = int(m24.group(1)), int(m24.group(2))
                    res = now_tz.replace(hour=hr, minute=mn, second=0, microsecond=0)
                    if (now_tz - res).total_seconds() > 43200: res += timedelta(days=1)
                    entry_time = res

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
                 stake_amount: float = 2.0, max_gales: int = 2,
                 martingale_multiplier: float = 2.2, enabled: bool = False):
        super().__init__("PolycarpVIP", channel_id, enabled)
        self.blitz = blitz_mcp
        self.stake_amount = stake_amount
        self.max_gales = max_gales
        self.martingale_multiplier = martingale_multiplier

        self.balance_id: Optional[int] = None
        self.account_type = "training"
        self.open_trades: Dict[int, Dict] = {}
        self.processed_msg_ids = set()

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type

    def set_stake(self, stake: float):
        self.stake_amount = max(1.0, round(float(stake), 2))
        logger.info(f"⚡ [Polycarp] Blitz base stake set to: ${self.stake_amount:.2f}")

    async def handle_message(self, text: str, message_id: int, event: Any = None, msg_date: Any = None):
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
        asyncio.create_task(self.schedule_and_execute(sig, gale_level=0))

    async def schedule_and_execute(self, sig: Dict[str, Any], gale_level: int = 0):
        pair = sig["pair"]
        direction = sig["direction"]
        expiry_secs = sig["expiry_secs"]
        entry_time = sig.get("entry_time")

        # Calculate stake based on martingale level
        stake = round(self.stake_amount * (self.martingale_multiplier ** gale_level), 2)

        # 1. Handle scheduling if entry time is specified (only on initial trade, not gales)
        if entry_time and gale_level == 0:
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
            chosen_exp = min(avail_expirations, key=lambda x: abs(x - expiry_secs))
            logger.info(f"[Polycarp] Adjusted expiration from {expiry_secs}s to {chosen_exp}s based on asset limits.")

        # 3. Place Trade on Blitz MCP
        gale_tag = f" [Gale {gale_level}]" if gale_level > 0 else ""
        logger.info(f"🚀 [Polycarp] Executing Blitz trade{gale_tag}: {pair} (ID: {asset_id}) {direction.upper()} ${stake} ({chosen_exp}s)")
        await self.notify(
            f"⚡ [Polycarp VIP EXECUTING BLITZ TRADE{gale_tag}]\n"
            f"Asset    : {asset.get('name')}\n"
            f"Direction: {direction.upper()}\n"
            f"Stake    : ${stake:.2f}\n"
            f"Payout   : {profit_percent}%\n"
            f"Duration : {chosen_exp}s"
        )

        res = self.blitz.place_trade(
            balance_id=self.balance_id,
            asset_id=asset_id,
            direction=direction,
            amount=stake,
            profit_percent=profit_percent,
            expiration_size=chosen_exp
        )

        if "position_id" in res:
            pos_id = res["position_id"]
            logger.info(f"✅ [Polycarp] Blitz Position Opened! ID: #{pos_id}{gale_tag}")
            self.open_trades[pos_id] = {
                "position_id": pos_id,
                "asset_id": asset_id,
                "pair": pair,
                "direction": direction,
                "amount": stake,
                "profit_percent": profit_percent,
                "expiration_size": chosen_exp,
                "gale_level": gale_level,
                "opened_at": time.time(),
                "sig": sig
            }
            await self.notify(f"✅ [Polycarp] Blitz Position Opened! ID: #{pos_id}{gale_tag}")
            asyncio.create_task(self.monitor_settlement(pos_id, chosen_exp, gale_level))
        else:
            logger.error(f"❌ [Polycarp] Blitz trade failed: {res}")
            await self.notify(f"❌ [Polycarp] Blitz Trade Failed: {res.get('error', res)}")

    async def monitor_settlement(self, pos_id: int, exp_secs: int, gale_level: int = 0):
        """Wait for position expiration, handle result, and execute Martingale if loss."""
        await asyncio.sleep(exp_secs + 4) # wait until expired + buffer

        trade = None
        for attempt in range(3):
            try:
                history = self.blitz.get_trade_history(limit=15)
                trade = next((h for h in history if h.get("position_id") == pos_id), None)
                if trade:
                    break
            except Exception as e:
                logger.warning(f"[Polycarp] Trade history fetch attempt {attempt+1} warning: {e}")
            await asyncio.sleep(2)

        trade_info = self.open_trades.pop(pos_id, {})
        pair = trade_info.get("pair", "Blitz Option")
        direction = trade_info.get("direction", "call")
        stake = trade_info.get("amount", self.stake_amount)
        sig = trade_info.get("sig", {"pair": pair, "direction": direction, "expiry_secs": exp_secs, "expiry_mins": exp_secs // 60})

        if trade:
            res_str = str(trade.get("result", "")).lower()
            profit = float(trade.get("profit", 0.0))
            is_win = (res_str == "win" or profit > 0)
            pnl = profit

            try:
                gsheet_logger.log_trade({
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "asset": pair,
                    "direction": direction.upper(),
                    "amount": stake,
                    "expiry": exp_secs,
                    "result": "WIN" if is_win else "LOSS",
                    "profit": pnl if is_win else -stake,
                    "gale_level": gale_level,
                    "signal_source": "Polycarp VIP",
                    "entry_price": trade.get("open_price", trade.get("entry_price", 0.0)),
                    "close": trade.get("close_price", 0.0)
                }, worksheet_name="Polycarp_Trades")
            except Exception as ge:
                logger.warning(f"[Polycarp] GSheet log error: {ge}")

            gale_label = f" (Gale {gale_level})" if gale_level > 0 else ""

            if is_win:
                logger.info(f"🏆 [Polycarp] WIN on #{pos_id}{gale_label}! Profit: +${pnl:.2f}")
                await self.notify(
                    f"🏆 [Polycarp VIP WIN{gale_label}]\n"
                    f"Position : #{pos_id} ({pair} {direction.upper()})\n"
                    f"Result   : WIN\n"
                    f"Net PnL  : +${pnl:.2f}\n"
                    f"🎯 Martingale reset to Base."
                )
            else:
                logger.info(f"❌ [Polycarp] LOSS on #{pos_id}{gale_label}! Net: -${abs(pnl):.2f}")
                if gale_level < self.max_gales:
                    next_gale = gale_level + 1
                    next_stake = round(self.stake_amount * (self.martingale_multiplier ** next_gale), 2)
                    await self.notify(
                        f"🔄 [Polycarp VIP MARTINGALE RECOVERY — GALE {next_gale}/{self.max_gales}]\n"
                        f"Loss on #{pos_id}{gale_label}.\n"
                        f"Re-entering {pair} {direction.upper()} with ${next_stake:.2f}..."
                    )
                    # Immediate re-entry on same pair and direction!
                    asyncio.create_task(self.schedule_and_execute(sig, gale_level=next_gale))
                else:
                    await self.notify(
                        f"❌ [Polycarp VIP MAX GALE REACHED]\n"
                        f"Position #{pos_id} ended in LOSS after {self.max_gales} recovery step(s).\n"
                        f"Stopping Martingale sequence for {pair}."
                    )
        else:
            logger.info(f"[Polycarp] Position #{pos_id} settled.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "channel_id": self.channel_id,
            "stake_amount": self.stake_amount,
            "max_gales": self.max_gales,
            "martingale_multiplier": self.martingale_multiplier,
            "open_trades_count": len(self.open_trades)
        }
