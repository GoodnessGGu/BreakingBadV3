"""
copiers/gold_pips_copier.py - Gold Pips Hunter Direct Signal Copier
Preserves 100% of the Gold Pips Hunter signal parsing, execution, breakeven, and close-all logic.
"""

import re
import time
import logging
import asyncio
from datetime import datetime
from typing import Dict, Any, Optional, List
from copiers.base_copier import BaseCopier
from clients.forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

logger = logging.getLogger("GoldPipsCopier")

GOLD_PIPS_DEFAULT_CHANNEL = -1003679078163
GOLD_ASSET_ID   = 74
GOLD_INSTRUMENT = "mcfd.74"

class GoldSignalParser:
    @staticmethod
    def parse_signal(text: str) -> Optional[Dict[str, Any]]:
        clean = text.lower()
        if not any(k in clean for k in ["gold", "xau", "xauusd"]):
            return None

        side = None
        if "buy" in clean or "long" in clean:
            side = "BUY"
        elif "sell" in clean or "short" in clean:
            side = "SELL"
        if not side:
            return None

        sl_m  = re.search(r"(?:sl|stop\s*loss)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp1_m = re.search(r"(?:tp1|take\s*profit\s*1?)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp2_m = re.search(r"(?:tp2|take\s*profit\s*2)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp3_m = re.search(r"(?:tp3|take\s*profit\s*3)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        tp_g  = re.search(r"(?:tp|take\s*profit)[\s:]*([0-9]+(?:\.[0-9]+)?)", clean)
        ent_m = re.search(r"(?:@|at|entry)[\s:]*([0-9]+(?:\.[0-9]+)?)(?:\s*-\s*([0-9]+(?:\.[0-9]+)?))?", clean)

        sl   = float(sl_m.group(1))  if sl_m  else None
        tp1  = float(tp1_m.group(1)) if tp1_m else (float(tp_g.group(1)) if tp_g else None)
        tp2  = float(tp2_m.group(1)) if tp2_m else None
        tp3  = float(tp3_m.group(1)) if tp3_m else None
        emin = float(ent_m.group(1)) if ent_m else None
        emax = float(ent_m.group(2)) if (ent_m and ent_m.group(2)) else emin

        return {
            "type": "SIGNAL", "side": side, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "entry_min": emin, "entry_max": emax, "raw": text
        }

    @staticmethod
    def parse_instruction(text: str) -> Optional[Dict[str, Any]]:
        clean = text.lower()
        if any(k in clean for k in ["breakeven", "break even", "move sl to entry", "sl to be", "secure profit"]):
            return {"type": "BREAKEVEN"}
        if any(k in clean for k in ["close all", "close positions", "close gold", "exit all"]):
            return {"type": "CLOSE_ALL"}
        return None

class GoldPipsCopier(BaseCopier):
    def __init__(self, mcp_client: IQForexMCPClient, channel_id: int = GOLD_PIPS_DEFAULT_CHANNEL,
                 lots: float = 1.0, leverage: int = 100, tp_target: int = 1,
                 max_slippage: float = 4.0, enabled: bool = True):
        super().__init__("GoldPipsHunter", channel_id, enabled)
        self.mcp = mcp_client
        self.lots = lots
        self.leverage = leverage
        self.tp_target = tp_target
        self.max_slippage = max_slippage

        self.balance_id: Optional[int] = None
        self.account_type = "training"
        self.open_positions: Dict[int, Dict] = {}
        self.processed_msg_ids = set()

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type

    def get_market_price(self) -> Dict[str, float]:
        try:
            p = self.mcp.calculate_order_size(
                asset_id=GOLD_ASSET_ID, balance_currency="USD",
                lots=self.lots, leverage=self.leverage
            )
            buy = float(p.get("buy_price", 0.0))
            sell = float(p.get("sell_price", 0.0))
            return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception as e:
            logger.warning(f"[GoldPips] Price fetch error: {e}")
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    async def handle_message(self, text: str, message_id: int, event: Any = None):
        if not self.is_enabled:
            return
        if message_id in self.processed_msg_ids:
            return
        self.processed_msg_ids.add(message_id)

        # 1. Check for instruction
        inst = GoldSignalParser.parse_instruction(text)
        if inst:
            itype = inst["type"]
            if itype == "BREAKEVEN":
                await self.apply_breakeven()
            elif itype == "CLOSE_ALL":
                await self.close_all_positions()
            return

        # 2. Check for signal
        sig = GoldSignalParser.parse_signal(text)
        if sig:
            await self.execute_signal(sig)

    async def execute_signal(self, sig: Dict[str, Any]):
        side = sig["side"]
        prices = self.get_market_price()
        cur_mid = prices["mid"]

        if cur_mid <= 0:
            logger.error("[GoldPips] Cannot execute: price unavailable.")
            return

        # Slippage check
        emin = sig.get("entry_min")
        emax = sig.get("entry_max")
        if emin is not None:
            zone_low = min(emin, emax or emin)
            zone_high = max(emin, emax or emin)
            if side == "BUY" and cur_mid > (zone_high + self.max_slippage):
                logger.warning(f"[GoldPips] Price {cur_mid:.2f} slipped too far above entry {zone_high:.2f}. Skipping.")
                await self.notify(f"⚠️ [Gold Pips] BUY skipped — Price slipped too far ({cur_mid:.2f} vs {zone_high:.2f})")
                return
            elif side == "SELL" and cur_mid < (zone_low - self.max_slippage):
                logger.warning(f"[GoldPips] Price {cur_mid:.2f} slipped too far below entry {zone_low:.2f}. Skipping.")
                await self.notify(f"⚠️ [Gold Pips] SELL skipped — Price slipped too far ({cur_mid:.2f} vs {zone_low:.2f})")
                return

        # Determine TP
        tp = None
        if self.tp_target == 1 and sig.get("tp1"):
            tp = sig["tp1"]
        elif self.tp_target == 2 and sig.get("tp2"):
            tp = sig["tp2"]
        elif self.tp_target == 3 and sig.get("tp3"):
            tp = sig["tp3"]
        elif sig.get("tp1"):
            tp = sig["tp1"]

        sl = sig.get("sl")
        exec_price = prices["buy"] if side == "BUY" else prices["sell"]

        await self.notify(
            f"⚡ [Gold Pips Hunter SIGNAL]\n"
            f"Side : {side}\n"
            f"Entry: ~{exec_price:.2f}\n"
            f"SL   : {sl if sl else 'None'}\n"
            f"TP   : {tp if tp else 'None'} (Target {self.tp_target})\n"
            f"Lots : {self.lots} (Lev {self.leverage}x)"
        )

        res = self.mcp.place_market_order(
            side=side.lower(),
            balance_id=self.balance_id,
            instrument_id=GOLD_INSTRUMENT,
            asset_id=GOLD_ASSET_ID,
            lots=self.lots,
            leverage=self.leverage,
            stop_loss=sl,
            take_profit=tp,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if "order_id" in res:
            order_id = res["order_id"]
            logger.info(f"✅ [GoldPips] Order placed! ID: #{order_id}")
            self.open_positions[order_id] = {
                "order_id": order_id,
                "position_id": None,
                "side": side,
                "entry_price": exec_price,
                "sl": sl,
                "tp": tp,
                "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            await self.notify(f"✅ [Gold Pips] Order Filled! ID: #{order_id}")
        else:
            logger.error(f"❌ [GoldPips] Order failed: {res}")
            await self.notify(f"❌ [Gold Pips] Order Failed: {res.get('error', res)}")

    async def apply_breakeven(self):
        logger.info("🛡️ [GoldPips] Breakeven instruction received!")
        positions = self.mcp.list_positions(balance_id=self.balance_id)
        count = 0
        for p in positions:
            if p.get("asset_id") == GOLD_ASSET_ID:
                pos_id = p.get("position_id") or p.get("id")
                open_quote = float(p.get("open_quote", 0.0))
                if pos_id and open_quote > 0:
                    self.mcp.change_position_stop_loss(position_id=pos_id, level=open_quote)
                    count += 1
        await self.notify(f"🛡️ [Gold Pips] Breakeven applied to {count} open Gold position(s).")

    async def close_all_positions(self):
        logger.info("🛑 [GoldPips] Close All instruction received!")
        positions = self.mcp.list_positions(balance_id=self.balance_id)
        count = 0
        for p in positions:
            if p.get("asset_id") == GOLD_ASSET_ID:
                pos_id = p.get("position_id") or p.get("id")
                if pos_id:
                    self.mcp.close_position(position_id=pos_id)
                    count += 1
        self.open_positions.clear()
        await self.notify(f"🛑 [Gold Pips] Closed all {count} open Gold position(s).")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "channel_id": self.channel_id,
            "tp_target": self.tp_target,
            "open_positions_count": len(self.open_positions)
        }
