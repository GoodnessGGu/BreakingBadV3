"""
copiers/callisto_copier.py - CallistoFx Dual-Zone Confirmation Copier
Preserves 100% of the Callisto zone parsing, candle confirmation, and dual-zone watching logic.
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

logger = logging.getLogger("CallistoCopier")

CALLISTO_DEFAULT_CHANNEL = -1002848189989
GOLD_ASSET_ID   = 74
GOLD_INSTRUMENT = "mcfd.74"

class CallistoZoneParser:
    @staticmethod
    def is_zone_message(text: str) -> bool:
        u = text.upper()
        return "BUY ZONE" in u or "SELL ZONE" in u

    @staticmethod
    def parse_all(text: str) -> List[Dict[str, Any]]:
        upper   = text.upper()
        results = []
        for inv in ("BUY", "SELL"):
            if inv + " ZONE INVALIDATED" in upper:
                results.append({"type": "INVALIDATE", "side": inv})

        for m in re.finditer(
            r"(?:NEW\s+)?(BUY|SELL)\s+ZONE[:\s]+([0-9]+(?:\.[0-9]+)?)\s*[-\u2013]\s*([0-9]+(?:\.[0-9]+)?)",
            upper
        ):
            side = m.group(1)
            a, b = float(m.group(2)), float(m.group(3))
            zhi, zlo = max(a, b), min(a, b)

            t = re.search(r"towards\s+(?:the\s+)?([0-9]+(?:\.[0-9]+)?)", text, re.IGNORECASE)
            target = float(t.group(1)) if t else None
            if not target:
                for lm in re.finditer(r"([0-9]{4,}(?:\.[0-9]+)?)\s*level", text, re.IGNORECASE):
                    lv = float(lm.group(1))
                    if not (zlo - 1 <= lv <= zhi + 1):
                        target = lv
                        break

            results.append({
                "type": "ZONE",
                "side": side,
                "zone_high": zhi,
                "zone_low": zlo,
                "target": target
            })
        return results

class CallistoCopier(BaseCopier):
    def __init__(self, mcp_client: IQForexMCPClient, channel_id: int = CALLISTO_DEFAULT_CHANNEL,
                 lots: float = 1.0, leverage: int = 100, sl_buffer: float = 3.0,
                 poll_interval: int = 15, confirm_interval: int = 5, max_zone_wait_hrs: int = 8,
                 enabled: bool = True):
        super().__init__("CallistoFx", channel_id, enabled)
        self.mcp = mcp_client
        self.lots = lots
        self.leverage = leverage
        self.sl_buffer = sl_buffer
        self.poll_interval = poll_interval
        self.confirm_interval = confirm_interval
        self.max_zone_wait_hrs = max_zone_wait_hrs

        self.balance_id: Optional[int] = None
        self.account_type = "training"
        self.active_zones: Dict[str, Dict] = {}  # keyed by "BUY" / "SELL"
        self.zone_tasks: Dict[str, asyncio.Task] = {}
        self.open_positions: Dict[int, Dict] = {}
        self.processed_msg_ids = set()

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))
        logger.info(f"📊 [CallistoFx] Lot size set to: {self.lots}")

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)
        logger.info(f"⚡ [CallistoFx] Leverage set to: {self.leverage}x")

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

    def check_confirmation(self, side: str) -> Optional[Dict[str, Any]]:
        zone = self.active_zones.get(side)
        if not zone:
            return None
        zhi, zlo = zone["zone_high"], zone["zone_low"]

        for tf in [60, 120, 300]:
            try:
                candles = self.mcp.get_candles(asset_id=GOLD_ASSET_ID, size=tf, count=3)
                if not candles or len(candles) < 2:
                    continue
                last = candles[-1]
                op  = float(last.get("open",  0.0))
                cl  = float(last.get("close", 0.0))
                hi  = float(last.get("max",   0.0))
                lo  = float(last.get("min",   0.0))
                body = abs(cl - op)

                if side == "BUY":
                    in_zone = lo <= zhi and cl >= zlo
                    bullish = cl > op and body >= 0.4 and cl >= (hi - body * 0.35)
                    if in_zone and bullish:
                        return {"confirmed": True, "tf": tf, "candle_close": cl, "candle_low": lo}
                elif side == "SELL":
                    in_zone = hi >= zlo and cl <= zhi
                    bearish = cl < op and body >= 0.4 and cl <= (lo + body * 0.35)
                    if in_zone and bearish:
                        return {"confirmed": True, "tf": tf, "candle_close": cl, "candle_high": hi}
            except Exception as e:
                logger.warning(f"[Callisto] Error checking {tf}s candle for {side}: {e}")
        return None

    def set_zone(self, side: str, zone_data: Dict[str, Any]):
        old_task = self.zone_tasks.get(side)
        if old_task and not old_task.done():
            old_task.cancel()

        self.active_zones[side] = zone_data
        logger.info(f"📍 [Callisto] Active {side} ZONE set: {zone_data['zone_low']:.2f} - {zone_data['zone_high']:.2f}")
        self.zone_tasks[side] = asyncio.ensure_future(self._watch_zone(side))

    def invalidate_zone(self, side: str):
        if side in self.active_zones:
            del self.active_zones[side]
        task = self.zone_tasks.pop(side, None)
        if task and not task.done():
            task.cancel()
        logger.info(f"🚫 [Callisto] {side} ZONE INVALIDATED.")

    async def _watch_zone(self, side: str):
        zone = self.active_zones.get(side)
        if not zone:
            return
        zhi, zlo = zone["zone_high"], zone["zone_low"]
        logger.info(f"👀 [Callisto] Started watching {side} zone [{zlo:.2f} - {zhi:.2f}]")

        try:
            while side in self.active_zones:
                if time.time() - zone["created_at"] > (self.max_zone_wait_hrs * 3600):
                    logger.info(f"⏰ [Callisto] {side} zone expired after {self.max_zone_wait_hrs}h.")
                    self.invalidate_zone(side)
                    break

                prices = self.get_market_price()
                mid = prices["mid"]

                if mid > 0:
                    in_zone = (zlo <= mid <= zhi)
                    if in_zone:
                        logger.info(f"🎯 [Callisto] Price {mid:.2f} IN {side} ZONE [{zlo:.2f} - {zhi:.2f}]. Checking confirmation...")
                        for _ in range(6):
                            conf = self.check_confirmation(side)
                            if conf:
                                await self.execute_trade(side, prices, conf)
                                self.invalidate_zone(side)
                                return
                            await asyncio.sleep(self.confirm_interval)
                await asyncio.sleep(self.poll_interval)
        except asyncio.CancelledError:
            pass

    async def execute_trade(self, side: str, prices: Dict[str, float], conf: Dict[str, Any]):
        zone = self.active_zones.get(side, {})
        exec_price = prices["buy"] if side == "BUY" else prices["sell"]

        if side == "BUY":
            sl = round(zone.get("zone_low", exec_price) - self.sl_buffer, 2)
            target = zone.get("target")
            tp = round(target, 2) if target and target > exec_price else round(exec_price + abs(exec_price - sl) * 2.0, 2)
        else:
            sl = round(zone.get("zone_high", exec_price) + self.sl_buffer, 2)
            target = zone.get("target")
            tp = round(target, 2) if target and target < exec_price else round(exec_price - abs(exec_price - sl) * 2.0, 2)

        msg = (
            f"⚡ [CallistoFx CONFIRMATION]\n"
            f"Side : {side}\n"
            f"Entry: {exec_price:.2f} (TF {conf.get('tf')}s)\n"
            f"SL   : {sl:.2f}\n"
            f"TP   : {tp:.2f}\n"
            f"Zone : {zone.get('zone_low', 0):.2f} – {zone.get('zone_high', 0):.2f}"
        )
        await self.notify(msg)

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
            logger.info(f"✅ [Callisto] Order placed! ID: #{order_id}")
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
            await self.notify(f"✅ [Callisto] Trade Executed: #{order_id} {side} @ {exec_price:.2f}")
            asyncio.create_task(self._monitor_position(order_id))
        else:
            logger.error(f"❌ [Callisto] Order failed: {res}")
            await self.notify(f"❌ [Callisto] Order Failed: {res.get('error', res)}")

    async def _monitor_position(self, order_id: int):
        """Monitors active Callisto trade for 1.0R Breakeven and logs settlement when closed."""
        await asyncio.sleep(5)
        pos = self.open_positions.get(order_id)
        if not pos:
            return

        # 1. Resolve position_id from list_positions
        for _ in range(6):
            if pos.get("position_id"):
                break
            try:
                positions = self.mcp.list_positions(balance_id=self.balance_id)
                for p in positions:
                    if p.get("asset_id") == GOLD_ASSET_ID:
                        pos["position_id"] = p.get("position_id") or p.get("id")
                        break
            except Exception as e:
                logger.warning(f"[Callisto] Position lookup error: {e}")
            if not pos.get("position_id"):
                await asyncio.sleep(5)

        pos_id = pos.get("position_id")
        side = pos["side"]
        entry = pos["entry_price"]
        sl = pos["initial_sl"]
        risk_dist = abs(entry - sl)

        logger.info(f"🛡️ [Callisto] Monitoring position #{pos_id or order_id} for Breakeven & Settlement.")

        while order_id in self.open_positions:
            await asyncio.sleep(self.poll_interval)
            try:
                open_positions = self.mcp.list_positions(balance_id=self.balance_id)
                is_still_open = False
                if pos_id:
                    is_still_open = any((p.get("position_id") or p.get("id")) == pos_id for p in open_positions)
                else:
                    is_still_open = any(p.get("asset_id") == GOLD_ASSET_ID for p in open_positions)
            except Exception as e:
                logger.warning(f"[Callisto] Error listing positions: {e}")
                continue

            if not is_still_open:
                logger.info(f"📊 [Callisto] Position #{pos_id or order_id} closed! Resolving settlement...")
                await self._log_trade_closure(order_id)
                self.open_positions.pop(order_id, None)
                break

            # 1.0R Breakeven shift
            if not pos.get("moved_to_be") and pos_id:
                prices = self.get_market_price()
                mid = prices["mid"]
                if mid > 0:
                    hit_1r = (mid - entry >= risk_dist) if side == "BUY" else (entry - mid >= risk_dist)
                    if hit_1r:
                        be_buf = 0.25
                        be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)
                        logger.info(f"🛡️ [Callisto BREAKEVEN] 1.0R reached on #{pos_id}! Moving SL to {be_level}")
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
                        if not res.get("error"):
                            pos["sl"] = be_level
                            pos["moved_to_be"] = True
                            await self.notify(
                                f"🛡️ [Callisto BREAKEVEN ACTIVATED]\n"
                                f"Position: #{pos_id} ({side})\n"
                                f"SL shifted to: {be_level:.2f}"
                            )

    async def _log_trade_closure(self, order_id: int):
        pos = self.open_positions.get(order_id, {})
        pos_id = pos.get("position_id")
        side = pos.get("side", "BUY")
        entry_px = pos.get("entry_price", 0.0)
        tp = pos.get("tp", 0.0)
        sl = pos.get("sl", 0.0)

        pnl = 0.0
        exit_px = 0.0
        reason = "closed"

        try:
            history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=10)
            matched = next((h for h in history if (pos_id and h.get("position_id") == pos_id) or h.get("asset_id") == GOLD_ASSET_ID), None)
            if matched:
                pnl = float(matched.get("pnl", 0.0))
                exit_px = float(matched.get("close_price", 0.0))
                reason = matched.get("close_reason", "closed")
        except Exception as e:
            logger.warning(f"[Callisto] Trade history lookup error: {e}")

        bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
        eq = bal.get("equity", 0.0) if bal else 0.0

        try:
            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": "Callisto Gold (XAUUSD)",
                "side": side,
                "lots": self.lots,
                "entry_price": entry_px,
                "stop_loss": sl,
                "take_profit": tp,
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * 10, 1) if exit_px else 0.0,
                "risk_reward": "Callisto Zone",
                "exit_reason": reason,
                "position_id": pos_id or order_id,
                "balance_equity": eq
            })
        except Exception as e:
            logger.warning(f"[Callisto] GSheet log error: {e}")

        emoji = "🏆 WIN" if pnl > 0 else "❌ LOSS"
        await self.notify(
            f"{emoji} [Callisto TRADE SETTLED]\n"
            f"Side    : {side}\n"
            f"Entry   : {entry_px:.2f}\n"
            f"Exit    : {exit_px:.2f}\n"
            f"PnL     : ${pnl:+.2f}\n"
            f"Reason  : {reason}"
        )

    async def handle_message(self, text: str, message_id: int, event: Any = None):
        if not self.is_enabled:
            return
        if message_id in self.processed_msg_ids:
            return
        self.processed_msg_ids.add(message_id)

        parsed_items = CallistoZoneParser.parse_all(text)
        if not parsed_items:
            return

        for item in parsed_items:
            itype = item.get("type")
            side  = item.get("side")
            if itype == "INVALIDATE":
                self.invalidate_zone(side)
                await self.notify(f"🚫 [Callisto] {side} ZONE INVALIDATED by channel.")
            elif itype == "ZONE":
                zone_data = {
                    "side": side,
                    "zone_high": item["zone_high"],
                    "zone_low": item["zone_low"],
                    "target": item.get("target"),
                    "created_at": time.time()
                }
                self.set_zone(side, zone_data)
                target_str = f" | Target: {zone_data['target']:.2f}" if zone_data['target'] else ""
                await self.notify(
                    f"📍 [Callisto] NEW {side} ZONE DETECTED\n"
                    f"Range: {zone_data['zone_low']:.2f} – {zone_data['zone_high']:.2f}{target_str}\n"
                    f"👀 Bot is monitoring for candle confirmation..."
                )

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "channel_id": self.channel_id,
            "active_zones": {
                s: f"{z['zone_low']:.2f} - {z['zone_high']:.2f}" for s, z in self.active_zones.items()
            },
            "open_positions_count": len(self.open_positions)
        }
