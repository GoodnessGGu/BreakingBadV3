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
        return any(kw in u for kw in ["BUY ZONE", "SELL ZONE", "BREAK EVEN", "BREAKEVEN", "SECURE PROFIT"])

    @staticmethod
    def parse_all(text: str) -> List[Dict[str, Any]]:
        upper   = text.upper()
        results = []
        for inv in ("BUY", "SELL"):
            if f"{inv} ZONE INVALIDATED" in upper or f"{inv} ZONE CANCELLED" in upper:
                results.append({"type": "INVALIDATE", "side": inv})

        if any(kw in upper for kw in ["BREAK EVEN", "BREAKEVEN", "SET STOPS TO BREAK", "MOVE SL TO ENTRY", "MOVE STOPS TO BE", "SECURE PROFIT", "SECURE MORE PROFIT"]):
            results.append({"type": "BREAKEVEN"})

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
            macro_tp = round(target, 2) if target and target > exec_price else round(exec_price + abs(exec_price - sl) * 2.0, 2)
        else:
            sl = round(zone.get("zone_high", exec_price) + self.sl_buffer, 2)
            target = zone.get("target")
            macro_tp = round(target, 2) if target and target < exec_price else round(exec_price - abs(exec_price - sl) * 2.0, 2)

        orders_to_place = []
        # Respect broker min_quantity (1.0 for Gold CFD on IQ Option)
        min_qty = 1.0
        try:
            inst = self.mcp.get_instruments(GOLD_ASSET_ID)
            if inst and isinstance(inst, dict) and "instruments" in inst and len(inst["instruments"]) > 0:
                min_qty = float(inst["instruments"][0].get("min_quantity", 1.0))
        except Exception:
            min_qty = 1.0

        if self.lots >= (min_qty * 2.0):
            lot1 = round(self.lots / 2, 2)
            lot2 = round(self.lots - lot1, 2)
            tp1 = round(exec_price + 4.50 if side == "BUY" else exec_price - 4.50, 2)
            orders_to_place.append({
                "tag": "TP1 Scalper (50%)",
                "lots": lot1,
                "sl": sl,
                "tp": tp1,
                "is_tp1": True
            })
            orders_to_place.append({
                "tag": "Macro Runner (50%)",
                "lots": lot2,
                "sl": sl,
                "tp": macro_tp,
                "is_tp1": False
            })
        else:
            trade_lots = max(min_qty, self.lots)
            orders_to_place.append({
                "tag": "Standard Position (100%)",
                "lots": trade_lots,
                "sl": sl,
                "tp": macro_tp,
                "is_tp1": False
            })

        plan_desc = "\n".join([f"  • {o['tag']}: {o['lots']}L | TP: {o['tp']:.2f}" for o in orders_to_place])
        msg = (
            f"⚡ [CallistoFx CONFIRMATION — SPLIT ENTRY]\n"
            f"Side : {side}\n"
            f"Entry: {exec_price:.2f} (TF {conf.get('tf')}s)\n"
            f"SL   : {sl:.2f}\n"
            f"Zone : {zone.get('zone_low', 0):.2f} – {zone.get('zone_high', 0):.2f}\n"
            f"Orders:\n{plan_desc}"
        )
        await self.notify(msg)

        for o in orders_to_place:
            res = self.mcp.place_market_order(
                side=side.lower(),
                balance_id=self.balance_id,
                instrument_id=GOLD_INSTRUMENT,
                asset_id=GOLD_ASSET_ID,
                lots=o["lots"],
                leverage=self.leverage,
                stop_loss=o["sl"],
                take_profit=o["tp"],
                is_margin_isolated=True,
                keep_position_open=False
            )

            if "order_id" in res:
                order_id = res["order_id"]
                logger.info(f"✅ [Callisto] {o['tag']} placed! ID: #{order_id}")
                self.open_positions[order_id] = {
                    "order_id": order_id,
                    "position_id": None,
                    "tag": o["tag"],
                    "is_tp1": o["is_tp1"],
                    "lots": o["lots"],
                    "side": side,
                    "entry_price": exec_price,
                    "sl": o["sl"],
                    "initial_sl": o["sl"],
                    "tp": o["tp"],
                    "moved_to_be": False,
                    "trailing_stage": 0,
                    "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                await self.notify(f"✅ [Callisto] {o['tag']} Executed: #{order_id} {side} {o['lots']}L @ {exec_price:.2f}")
                asyncio.create_task(self._monitor_position(order_id))
            else:
                logger.error(f"❌ [Callisto] {o['tag']} order failed: {res}")
                await self.notify(f"❌ [Callisto] {o['tag']} Order Failed: {res.get('error', res)}")

    async def trigger_manual_breakeven(self, reason: str = "Channel Broadcast"):
        """Shifts all active Callisto positions to Breakeven."""
        if not self.open_positions:
            logger.debug("[Callisto] No active Callisto positions open to apply Breakeven. Ignoring.")
            return

        prices = self.get_market_price()
        mid = prices.get("mid", 0.0)

        for order_id, pos in list(self.open_positions.items()):
            pos_id = pos.get("position_id")
            if not pos_id:
                continue
            if pos.get("moved_to_be"):
                continue

            side = pos["side"]
            entry = pos["entry_price"]

            # Prevent premature stopout if currently in drawdown
            if mid > 0:
                if side == "BUY" and mid < (entry - 0.50):
                    logger.warning(f"⚠️ [Callisto] Position #{pos_id} is below entry ({mid:.2f} < {entry:.2f}). Skipping premature BE.")
                    continue
                elif side == "SELL" and mid > (entry + 0.50):
                    logger.warning(f"⚠️ [Callisto] Position #{pos_id} is above entry ({mid:.2f} > {entry:.2f}). Skipping premature BE.")
                    continue

            be_buf = 0.30
            be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)

            logger.info(f"🛡️ [Callisto BREAKEVEN] ({reason}) Moving SL to {be_level} for #{pos_id}")
            res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
            if not res.get("error"):
                pos["sl"] = be_level
                pos["moved_to_be"] = True
                pos["trailing_stage"] = max(pos.get("trailing_stage", 0), 2)
                await self.notify(
                    f"🛡️ [Callisto BREAKEVEN ACTIVATED]\n"
                    f"Trigger: {reason}\n"
                    f"Position: #{pos_id} ({side} - {pos.get('tag', 'Order')})\n"
                    f"SL shifted to: {be_level:.2f}"
                )

    async def _monitor_position(self, order_id: int):
        """Monitors active Callisto trade with pip milestone scaling, breakeven, and profit trailing."""
        await asyncio.sleep(5)
        pos = self.open_positions.get(order_id)
        if not pos:
            return

        # 1. Resolve position_id from list_positions (avoiding collisions across split tickets)
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
                        p_tp = float(p.get("take_profit", 0) or p.get("tp", 0) or 0)
                        if p_tp > 0 and abs(p_tp - pos["tp"]) < 0.2:
                            pos["position_id"] = p_id
                            resolved = True
                            break
                        elif not pos.get("position_id"):
                            pos["position_id"] = p_id
                            resolved = True
            except Exception as e:
                logger.warning(f"[Callisto] Position lookup error: {e}")
            if not pos.get("position_id"):
                await asyncio.sleep(4)

        if not resolved or not pos.get("position_id"):
            logger.warning(f"⚠️ [Callisto] Order #{order_id} failed to map to an active position. Pruning from tracking.")
            self.open_positions.pop(order_id, None)
            return

        pos_id = pos.get("position_id")
        side = pos["side"]
        entry = pos["entry_price"]
        sl = pos["initial_sl"]
        tp = pos.get("tp")
        risk_dist = abs(entry - sl)
        tag = pos.get("tag", "Standard")

        # Firmly attach SL and TP if not already bound by broker
        if pos_id:
            try:
                if sl and sl > 0:
                    self.mcp.change_position_stop_loss(position_id=pos_id, level=sl)
                if tp and tp > 0:
                    self.mcp.change_position_take_profit(position_id=pos_id, level=tp)
            except Exception as e:
                logger.warning(f"[Callisto] Notice setting post-fill SL/TP on #{pos_id}: {e}")

        logger.info(f"🛡️ [Callisto] Monitoring position #{pos_id or order_id} ({tag}) with milestone trailing & Breakeven.")

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
                logger.info(f"📊 [Callisto] Position #{pos_id or order_id} ({tag}) closed! Resolving settlement...")
                await self._log_trade_closure(order_id)
                self.open_positions.pop(order_id, None)
                break

            # Multi-tier Milestone Trailing and Breakeven
            if pos_id:
                prices = self.get_market_price()
                mid = prices["mid"]
                if mid > 0:
                    gain = (mid - entry) if side == "BUY" else (entry - mid)
                    gain_pips = gain * 10.0
                    stage = pos.get("trailing_stage", 0)

                    # Stage 1: +30 Pips ($3.00) -> Cut risk by 50%
                    if gain_pips >= 30.0 and stage < 1:
                        half_risk_sl = round(entry - (risk_dist * 0.5) if side == "BUY" else entry + (risk_dist * 0.5), 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=half_risk_sl)
                        if not res.get("error"):
                            pos["sl"] = half_risk_sl
                            pos["trailing_stage"] = 1
                            logger.info(f"🛡️ [Callisto +30 Pips] Risk cut 50% on #{pos_id}! SL: {half_risk_sl}")
                            await self.notify(
                                f"🛡️ [Callisto DEFENSE +30 PIPS]\n"
                                f"Position #{pos_id} ({tag})\n"
                                f"Risk reduced by 50% | New SL: {half_risk_sl:.2f}"
                            )

                    # Stage 2: +50 Pips ($5.00) or 1.0R -> Move to Breakeven (+0.30 buffer)
                    if (gain_pips >= 50.0 or gain >= risk_dist) and stage < 2:
                        be_buf = 0.30
                        be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
                        if not res.get("error"):
                            pos["sl"] = be_level
                            pos["moved_to_be"] = True
                            pos["trailing_stage"] = 2
                            logger.info(f"🛡️ [Callisto +50 Pips / 1R] Breakeven activated on #{pos_id}! SL: {be_level}")
                            await self.notify(
                                f"🛡️ [Callisto BREAKEVEN +50 PIPS]\n"
                                f"Position #{pos_id} ({tag})\n"
                                f"Trade is now Risk-Free! SL shifted to: {be_level:.2f}"
                            )

                    # Stage 3: +100 Pips ($10.00) -> Lock in +50 Pips profit (Runner only)
                    if gain_pips >= 100.0 and stage < 3:
                        lock_50 = round(entry + 5.00 if side == "BUY" else entry - 5.00, 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=lock_50)
                        if not res.get("error"):
                            pos["sl"] = lock_50
                            pos["trailing_stage"] = 3
                            logger.info(f"💰 [Callisto +100 Pips] Secured +50 Pips on #{pos_id}! SL: {lock_50}")
                            await self.notify(
                                f"💰 [Callisto PROFIT LOCK +100 PIPS]\n"
                                f"Position #{pos_id} ({tag})\n"
                                f"Banked +50 Pips profit! New SL: {lock_50:.2f}"
                            )

                    # Stage 4: +150 Pips ($15.00) -> Lock in +100 Pips profit (Runner only)
                    if gain_pips >= 150.0 and stage < 4:
                        lock_100 = round(entry + 10.00 if side == "BUY" else entry - 10.00, 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=lock_100)
                        if not res.get("error"):
                            pos["sl"] = lock_100
                            pos["trailing_stage"] = 4
                            logger.info(f"💰 [Callisto +150 Pips] Secured +100 Pips on #{pos_id}! SL: {lock_100}")
                            await self.notify(
                                f"💰 [Callisto PROFIT LOCK +150 PIPS]\n"
                                f"Position #{pos_id} ({tag})\n"
                                f"Banked +100 Pips profit! New SL: {lock_100:.2f}"
                            )

                    # Stage 5: +200+ Pips ($20.00+) -> Dynamic 60 Pip Trailing Stop
                    if gain_pips >= 200.0:
                        trail_sl = round(mid - 6.00 if side == "BUY" else mid + 6.00, 2)
                        current_sl = pos.get("sl", sl)
                        should_update = (trail_sl > current_sl + 0.80) if side == "BUY" else (trail_sl < current_sl - 0.80)
                        if should_update:
                            res = self.mcp.change_position_stop_loss(position_id=pos_id, level=trail_sl)
                            if not res.get("error"):
                                pos["sl"] = trail_sl
                                pos["trailing_stage"] = 5
                                logger.info(f"🎯 [Callisto Trailing Stop] Trailing SL updated to {trail_sl:.2f} on #{pos_id}")

    async def _log_trade_closure(self, order_id: int):
        pos = self.open_positions.get(order_id, {})
        pos_id = pos.get("position_id")
        tag = pos.get("tag", "Standard")
        side = pos.get("side", "BUY")
        lots = pos.get("lots", self.lots)
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
                "asset": f"Callisto Gold ({tag})",
                "side": side,
                "lots": lots,
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
            f"{emoji} [Callisto SETTLED — {tag}]\n"
            f"Side    : {side} ({lots} Lots)\n"
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
            elif itype == "BREAKEVEN":
                logger.info("📢 [Callisto] Received BREAKEVEN / SECURE PROFIT broadcast from channel!")
                await self.trigger_manual_breakeven(reason="Channel Broadcast")
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
