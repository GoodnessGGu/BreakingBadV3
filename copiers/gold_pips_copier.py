"""
copiers/gold_pips_copier.py - Gold Pips Hunter Direct Signal Copier
Preserves 100% of the Gold Pips Hunter signal parsing, execution, breakeven, and close-all logic.
"""

import re
import time
import logging
import asyncio
from datetime import datetime, timezone
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

    async def handle_message(self, text: str, message_id: int, event: Any = None, msg_date: Any = None):
        if not self.is_enabled:
            return
        if message_id in self.processed_msg_ids:
            return
        self.processed_msg_ids.add(message_id)

        # 1. Check message freshness for instant market execution
        if msg_date:
            try:
                now_utc = datetime.now(timezone.utc)
                msg_utc = msg_date if msg_date.tzinfo else msg_date.replace(tzinfo=timezone.utc)
                age_sec = (now_utc - msg_utc).total_seconds()
                if age_sec > 180:
                    logger.info(f"⏰ [GoldPips] Skipped historical signal #{message_id} ({int(age_sec)}s old during lookback)")
                    return
            except Exception as e:
                logger.warning(f"[GoldPips] Error checking message date: {e}")

        # 2. Check for instruction
        inst = GoldSignalParser.parse_instruction(text)
        if inst:
            itype = inst["type"]
            if itype == "BREAKEVEN":
                await self.apply_breakeven()
            elif itype == "CLOSE_ALL":
                await self.close_all_positions()
            return

        # 3. Check for signal
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

        # Check for conflicting opposite position on Gold
        try:
            positions = self.mcp.list_positions(balance_id=self.balance_id)
            opp_side = "short" if side == "BUY" else "long"
            opp_pos = next((p for p in positions if p.get("asset_id") == GOLD_ASSET_ID and p.get("type", "").lower() == opp_side), None)
            if opp_pos:
                pos_id = opp_pos.get("position_id") or opp_pos.get("id")
                logger.warning(f"⚠️ [GoldPips] Skipped {side} — An active {opp_side.upper()} position (#{pos_id}) already exists on Gold.")
                await self.notify(f"⚠️ [Gold Pips] Skipped {side} — Opposing {opp_side.upper()} position #{pos_id} already active on Gold.")
                return
        except Exception as e:
            logger.warning(f"[GoldPips] Error checking open positions: {e}")

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
                "initial_sl": sl,
                "tp": tp,
                "moved_to_be": False,
                "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
            await self.notify(f"✅ [Gold Pips] Order Filled: #{order_id} {side} @ {exec_price:.2f}")
            asyncio.create_task(self._monitor_position(order_id))
        else:
            logger.error(f"❌ [GoldPips] Order failed: {res}")
            await self.notify(f"❌ [Gold Pips] Order Failed: {res.get('error', res)}")

    async def _monitor_position(self, order_id: int):
        """Monitors active Gold Pips trade and logs settlement."""
        await asyncio.sleep(5)
        pos = self.open_positions.get(order_id)
        if not pos:
            return

        resolved = False
        for _ in range(8):
            if pos.get("position_id"):
                resolved = True
                break
            try:
                positions = self.mcp.list_positions(balance_id=self.balance_id)
                assigned_pos_ids = {p.get("position_id") for oid, p in self.open_positions.items() if oid != order_id and p.get("position_id")}
                for p in positions:
                    if p.get("asset_id") == GOLD_ASSET_ID:
                        p_id = p.get("position_id") or p.get("id")
                        if p_id in assigned_pos_ids:
                            continue
                        pos["position_id"] = p_id
                        resolved = True
                        break
            except Exception as e:
                logger.warning(f"[GoldPips] Position lookup error: {e}")
            if not pos.get("position_id"):
                await asyncio.sleep(4)

        if not resolved or not pos.get("position_id"):
            logger.warning(f"⚠️ [GoldPips] Order #{order_id} failed to map to active position. Pruning.")
            self.open_positions.pop(order_id, None)
            return

        pos_id = pos["position_id"]
        sl = pos.get("sl")
        tp = pos.get("tp")

        # Firmly attach SL and TP if not already bound by broker
        if pos_id:
            try:
                if sl and sl > 0:
                    self.mcp.change_position_stop_loss(position_id=pos_id, level=sl)
                if tp and tp > 0:
                    self.mcp.change_position_take_profit(position_id=pos_id, level=tp)
            except Exception as e:
                logger.warning(f"[GoldPips] Notice setting post-fill SL/TP on #{pos_id}: {e}")

        logger.info(f"🛡️ [GoldPips] Monitoring position #{pos_id} for settlement.")

        while order_id in self.open_positions:
            await asyncio.sleep(15)
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                is_still_open = any((p.get("position_id") or p.get("id")) == pos_id for p in open_positions)
            except Exception as e:
                logger.warning(f"[GoldPips] Error listing positions: {e}")
                continue

            if not is_still_open:
                logger.info(f"📊 [GoldPips] Position #{pos_id} closed! Resolving settlement...")
                await self._log_trade_closure(order_id)
                self.open_positions.pop(order_id, None)
                break

    async def _log_trade_closure(self, order_id: int):
        pos = self.open_positions.get(order_id, {})
        pos_id = pos.get("position_id")
        side = pos.get("side", "BUY")
        entry_px = pos.get("entry_price", 0.0)
        tp = pos.get("tp", 0.0)
        sl = pos.get("sl", 0.0)

        pnl, exit_px, reason = 0.0, 0.0, "closed"
        try:
            history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=10)
            matched = next((h for h in history if (pos_id and h.get("position_id") == pos_id) or h.get("asset_id") == GOLD_ASSET_ID), None)
            if matched:
                pnl = float(matched.get("pnl", 0.0))
                exit_px = float(matched.get("close_price", 0.0))
                reason = matched.get("close_reason", "closed")
        except Exception as e:
            logger.warning(f"[GoldPips] History lookup error: {e}")

        bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
        eq = bal.get("equity", 0.0) if bal else 0.0

        try:
            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": "Gold Pips Hunter (XAUUSD)",
                "side": side,
                "lots": self.lots,
                "entry_price": entry_px,
                "stop_loss": sl,
                "take_profit": tp,
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * 10, 1) if exit_px else 0.0,
                "risk_reward": f"Target {self.tp_target}",
                "exit_reason": reason,
                "position_id": pos_id or order_id,
                "balance_equity": eq
            })
        except Exception as e:
            logger.warning(f"[GoldPips] GSheet log error: {e}")

        emoji = "🏆 WIN" if pnl > 0 else "❌ LOSS"
        await self.notify(
            f"{emoji} [Gold Pips SETTLED]\n"
            f"Side    : {side} ({self.lots} Lots)\n"
            f"Entry   : {entry_px:.2f}\n"
            f"Exit    : {exit_px:.2f}\n"
            f"PnL     : ${pnl:+.2f}\n"
            f"Reason  : {reason}"
        )

    async def apply_breakeven(self):
        if not self.open_positions:
            logger.debug("[GoldPips] No active Gold Pips positions open to apply Breakeven. Ignoring.")
            return

        prices = self.get_market_price()
        mid = prices.get("mid", 0.0)
        count = 0

        for order_id, pos in list(self.open_positions.items()):
            pos_id = pos.get("position_id")
            if not pos_id or pos.get("moved_to_be"):
                continue

            side = pos["side"]
            entry = pos["entry_price"]

            # Prevent premature stopout if currently in drawdown
            if mid > 0:
                if side == "BUY" and mid < (entry - 0.50):
                    logger.warning(f"⚠️ [GoldPips] Position #{pos_id} is below entry ({mid:.2f} < {entry:.2f}). Skipping premature BE.")
                    continue
                elif side == "SELL" and mid > (entry + 0.50):
                    logger.warning(f"⚠️ [GoldPips] Position #{pos_id} is above entry ({mid:.2f} > {entry:.2f}). Skipping premature BE.")
                    continue

            be_buf = 0.30
            be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)
            res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
            if not res.get("error"):
                pos["sl"] = be_level
                pos["moved_to_be"] = True
                count += 1
                await self.notify(f"🛡️ [Gold Pips BREAKEVEN]\nPosition: #{pos_id} ({side})\nSL shifted to: {be_level:.2f}")

    async def close_all_positions(self):
        if not self.open_positions:
            logger.debug("[GoldPips] No active Gold Pips positions open to close.")
            return

        count = 0
        for order_id, pos in list(self.open_positions.items()):
            pos_id = pos.get("position_id")
            if pos_id:
                res = self.mcp.close_position(position_id=pos_id)
                if not res.get("error"):
                    count += 1
        self.open_positions.clear()
        if count > 0:
            await self.notify(f"🛑 [Gold Pips] Closed {count} active Gold Pips position(s).")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "channel_id": self.channel_id,
            "tp_target": self.tp_target,
            "open_positions_count": len(self.open_positions)
        }
