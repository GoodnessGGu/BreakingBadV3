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
    REJECT_KEYWORDS = [
        "performance", "weekly performance", "daily performance", "recap", "recaps",
        "win rate", "pips won", "total pips", "market analysis", "giveaway",
        "free capital", "happy sunday", "happy saturday", "be careful who you trust",
        "scammers", "waiting for the", "session is coming", "double profit"
    ]

    @staticmethod
    def parse_signal(text: str) -> Optional[Dict[str, Any]]:
        clean = text.lower()
        
        # 1. Noise / non-signal filter
        if any(k in clean for k in GoldSignalParser.REJECT_KEYWORDS):
            return None

        if not any(k in clean for k in ["gold", "xau", "xauusd"]):
            return None

        # 2. Determine side (require explicit buy/sell action)
        side = None
        if re.search(r'(?:^|\b)(?:gold\s+|xauusd\s+|xau\s+)?(?:buy\s*now|buy\b|long\b)', clean):
            side = "BUY"
        elif re.search(r'(?:^|\b)(?:gold\s+|xauusd\s+|xau\s+)?(?:sell\s*now|sell\b|short\b)', clean):
            side = "SELL"
        if not side:
            return None

        # 3. Stop Loss / Cut Loss (mandatory for execution safety)
        sl_m = re.search(r'(?:cut\s*loss|cutloss|stop\s*loss|\bsl\b)[^\d\n\r]*([0-9]+(?:\.[0-9]+)?)', clean)
        if not sl_m:
            return None
        sl = float(sl_m.group(1))

        # 4. Entry Zone / Range (mandatory to avoid entering on random teaser comments)
        ent_m = re.search(r'(?:entry\s*zone|entryzone|entry|@|at)[^\d\n\r]*([0-9]+(?:\.[0-9]+)?)(?:\s*[-–—/]\s*([0-9]+(?:\.[0-9]+)?))?', clean)
        if not ent_m:
            return None
        emin = float(ent_m.group(1))
        emax = float(ent_m.group(2)) if ent_m.group(2) else emin

        # 5. Take Profit levels (support multiple Take Profit lines with emojis)
        tp_matches = re.findall(r'(?:take\s*profit|tp\s*[123]?)[^\d\n\r]*([0-9]+(?:\.[0-9]+)?)', clean)
        tps = [float(x) for x in tp_matches] if tp_matches else []
        tp1 = tps[0] if len(tps) > 0 else None
        tp2 = tps[1] if len(tps) > 1 else None
        tp3 = tps[2] if len(tps) > 2 else None

        return {
            "type": "SIGNAL", "side": side, "sl": sl,
            "tp1": tp1, "tp2": tp2, "tp3": tp3,
            "entry_min": min(emin, emax), "entry_max": max(emin, emax), "raw": text
        }

    @staticmethod
    def parse_instruction(text: str) -> Optional[Dict[str, Any]]:
        clean = text.lower()
        # Milestone template gives a choice (Close all vs Breakeven); prioritize Breakeven to let winning positions run risk-free
        if any(k in clean for k in ["breakeven", "break even", "move sl to entry", "sl to be", "secure profit", "hold the positions with breakeven"]):
            return {"type": "BREAKEVEN"}
        if any(k in clean for k in ["close all", "close positions", "close gold", "exit all", "close out this trading week"]):
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

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))
        logger.info(f"📊 [GoldPips] Lot size set to: {self.lots}")

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)
        logger.info(f"⚡ [GoldPips] Leverage set to: {self.leverage}x")

    def get_market_price(self) -> Dict[str, float]:
        try:
            candles = self.mcp.get_candles(asset_id=GOLD_ASSET_ID, size=60, count=2)
            if candles and len(candles) > 0:
                px = float(candles[-1].get("close", 0.0))
                if px > 0:
                    return {"buy": px, "sell": px, "mid": px}
        except Exception:
            pass

        try:
            p = self.mcp.calculate_order_size(
                asset_id=GOLD_ASSET_ID, balance_currency="USD",
                lots=self.lots, leverage=self.leverage
            )
            if isinstance(p, dict) and "buy_price" in p:
                buy = float(p.get("buy_price", 0.0))
                sell = float(p.get("sell_price", 0.0))
                return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception:
            pass
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
