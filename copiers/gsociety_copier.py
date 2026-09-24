"""
copiers/gsociety_copier.py - G Society Telegram Signal Copier
Direct Gold (XAUUSD) signal parsing with Split-Order partial profit-taking,
milestone profit trailing, and channel Breakeven / Secure Profit commands.
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

logger = logging.getLogger("GSocietyCopier")

GSOCIETY_DEFAULT_CHANNEL = -1001883957585
GOLD_ASSET_ID   = 74
GOLD_INSTRUMENT = "mcfd.74"

class GSocietyParser:
    REJECT_KEYWORDS = [
        "performance", "before vs after", "round 1", "round 2", "round 3", "round 4", "round 5",
        "performance for today", "how to get access", "support @aaron", "just enjoy okay",
        "what a good ending", "pips settle", "where to buy", "just wait for my instructions"
    ]

    @staticmethod
    def parse_signal(text: str) -> Optional[Dict[str, Any]]:
        clean = text.strip()
        lower = clean.lower()

        # Reject pure commentary / recaps unless an explicit buy/sell action exists
        if any(k in lower for k in GSocietyParser.REJECT_KEYWORDS) and not any(k in lower for k in ["buy gold now", "sell gold now", "i buy gold", "i sell gold"]):
            return None

        # Determine side
        side = None
        if re.search(r'\b(?:i\s+)?buy\s+gold(?:\s+now)?\b', lower) or (re.search(r'\bbuy\b', lower) and "gold" in lower):
            side = "BUY"
        elif re.search(r'\b(?:i\s+)?sell\s+gold(?:\s+now)?\b', lower) or (re.search(r'\bsell\b', lower) and "gold" in lower):
            side = "SELL"
        elif "if sl just reentry" in lower or "reentry" in lower:
            side = "AUTO"

        if not side:
            return None

        # Stop Loss
        sl_m = re.search(r'\bsl[^\d\n\r]*([0-9]{4}(?:\.[0-9]+)?)', lower)
        if not sl_m:
            return None
        sl = float(sl_m.group(1))

        # Entry Range
        ent_m = re.search(r'([0-9]{4}(?:\.[0-9]+)?)\s*[-\u2013\u2014]\s*([0-9]{4}(?:\.[0-9]+)?)', clean)
        emin, emax = None, None
        if ent_m:
            a, b = float(ent_m.group(1)), float(ent_m.group(2))
            emin, emax = min(a, b), max(a, b)

        # TP levels
        tps = []
        tp_slash = re.search(r'\btp[^\d\n\r]*([0-9]{4}(?:\.[0-9]+)?)\s*[/\\&]\s*([0-9]{4}(?:\.[0-9]+)?)', lower)
        if tp_slash:
            tps = [float(tp_slash.group(1)), float(tp_slash.group(2))]
        else:
            tp_matches = re.findall(r'\btp[^\d\n\r]*([0-9]{4}(?:\.[0-9]+)?)', lower)
            if tp_matches:
                tps = [float(x) for x in tp_matches]

        tp1 = tps[0] if len(tps) > 0 else None
        tp2 = tps[1] if len(tps) > 1 else None

        if side == "AUTO" and tp1:
            side = "BUY" if tp1 > sl else "SELL"

        return {
            "type": "SIGNAL",
            "side": side,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "entry_min": emin,
            "entry_max": emax,
            "raw": text
        }

    @staticmethod
    def parse_instruction(text: str) -> Optional[Dict[str, Any]]:
        lower = text.lower()
        if any(k in lower for k in ["secure some", "cans secure", "xan secure", "secure profit", "breakeven", "break even", "set stops to be", "sl to entry"]):
            return {"type": "BREAKEVEN"}
        return None

class GSocietyCopier(BaseCopier):
    def __init__(self, mcp_client: IQForexMCPClient, channel_id: int = GSOCIETY_DEFAULT_CHANNEL,
                 lots: float = 1.0, leverage: int = 100, max_slippage: float = 4.0,
                 poll_interval: int = 15, enabled: bool = True):
        super().__init__("GSociety", channel_id, enabled)
        self.mcp = mcp_client
        self.lots = lots
        self.leverage = leverage
        self.max_slippage = max_slippage
        self.poll_interval = poll_interval

        self.balance_id: Optional[int] = None
        self.account_type = "training"
        self.open_positions: Dict[int, Dict] = {}
        self.processed_msg_ids = set()

    def set_balance(self, balance_id: int, account_type: str = "training"):
        self.balance_id = balance_id
        self.account_type = account_type

    def set_lots(self, lots: float):
        self.lots = max(0.01, round(float(lots), 2))
        logger.info(f"📊 [GSociety] Lot size set to: {self.lots}")

    def set_leverage(self, leverage: int):
        self.leverage = int(leverage)
        logger.info(f"⚡ [GSociety] Leverage set to: {self.leverage}x")

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
                lots=max(1.0, self.lots), leverage=self.leverage
            )
            if isinstance(p, dict) and "buy_price" in p:
                buy = float(p.get("buy_price", 0.0))
                sell = float(p.get("sell_price", 0.0))
                return {"buy": buy, "sell": sell, "mid": (buy + sell) / 2}
        except Exception:
            pass
        return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    async def execute_signal(self, sig: Dict[str, Any]):
        prices = self.get_market_price()
        side = sig["side"]
        exec_price = prices["buy"] if side == "BUY" else prices["sell"]

        if exec_price <= 0:
            logger.error("❌ [GSociety] Could not fetch market price.")
            return

        # Check for conflicting opposite position on Gold
        try:
            positions = self.mcp.list_positions(balance_id=self.balance_id)
            opp_side = "short" if side == "BUY" else "long"
            opp_pos = next((p for p in positions if p.get("asset_id") == GOLD_ASSET_ID and p.get("type", "").lower() == opp_side), None)
            if opp_pos:
                pos_id = opp_pos.get("position_id") or opp_pos.get("id")
                logger.warning(f"⚠️ [GSociety] Skipped {side} — An active {opp_side.upper()} position (#{pos_id}) already exists on Gold.")
                await self.notify(f"⚠️ [G Society] Skipped {side} — Opposing {opp_side.upper()} position #{pos_id} already active on Gold.")
                return
        except Exception as e:
            logger.warning(f"[GSociety] Error checking open positions: {e}")

        # Slippage check against entry range if provided
        emin = sig.get("entry_min")
        emax = sig.get("entry_max")
        if emin and emax:
            if side == "BUY" and exec_price > (emax + self.max_slippage):
                logger.warning(f"⚠️ [GSociety] Market price {exec_price:.2f} slipped too far above entry max {emax:.2f}")
                await self.notify(f"⚠️ [GSociety] Skipped BUY — Price {exec_price:.2f} slipped > {self.max_slippage} above max entry {emax:.2f}")
                return
            elif side == "SELL" and exec_price < (emin - self.max_slippage):
                logger.warning(f"⚠️ [GSociety] Market price {exec_price:.2f} slipped too far below entry min {emin:.2f}")
                await self.notify(f"⚠️ [GSociety] Skipped SELL — Price {exec_price:.2f} slipped > {self.max_slippage} below min entry {emin:.2f}")
                return

        sl = sig["sl"]
        tp1 = sig.get("tp1") or round(exec_price + 4.50 if side == "BUY" else exec_price - 4.50, 2)
        tp2 = sig.get("tp2") or round(exec_price + abs(exec_price - sl) * 2.0, 2)

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
                "tp": tp2,
                "is_tp1": False
            })
        else:
            trade_lots = max(min_qty, self.lots)
            orders_to_place.append({
                "tag": "Standard Position (100%)",
                "lots": trade_lots,
                "sl": sl,
                "tp": tp2 or tp1,
                "is_tp1": False
            })

        range_desc = f"{emin:.2f} – {emax:.2f}" if emin and emax else f"~{exec_price:.2f}"
        plan_desc = "\n".join([f"  • {o['tag']}: {o['lots']}L | TP: {o['tp']:.2f}" for o in orders_to_place])
        msg = (
            f"⚡ [G Society SIGNAL — SPLIT ENTRY]\n"
            f"Side : {side}\n"
            f"Entry: {exec_price:.2f} (Range: {range_desc})\n"
            f"SL   : {sl:.2f}\n"
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
                logger.info(f"✅ [GSociety] {o['tag']} placed! ID: #{order_id}")
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
                await self.notify(f"✅ [GSociety] {o['tag']} Live: #{order_id} {side} {o['lots']}L @ {exec_price:.2f}")
                asyncio.create_task(self._monitor_position(order_id))
            else:
                logger.error(f"❌ [GSociety] {o['tag']} failed: {res}")
                await self.notify(f"❌ [GSociety] {o['tag']} Order Failed: {res.get('error', res)}")

    async def trigger_manual_breakeven(self, reason: str = "Channel Broadcast"):
        """Shifts all active G Society positions to Breakeven."""
        if not self.open_positions:
            logger.debug("[GSociety] No active G Society positions open to apply Breakeven. Ignoring.")
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
                    logger.warning(f"⚠️ [GSociety] Position #{pos_id} is below entry ({mid:.2f} < {entry:.2f}). Skipping premature BE.")
                    continue
                elif side == "SELL" and mid > (entry + 0.50):
                    logger.warning(f"⚠️ [GSociety] Position #{pos_id} is above entry ({mid:.2f} > {entry:.2f}). Skipping premature BE.")
                    continue

            be_buf = 0.30
            be_level = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)

            logger.info(f"🛡️ [GSociety BREAKEVEN] ({reason}) Moving SL to {be_level} for #{pos_id}")
            res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
            if not res.get("error"):
                pos["sl"] = be_level
                pos["moved_to_be"] = True
                pos["trailing_stage"] = max(pos.get("trailing_stage", 0), 2)
                await self.notify(
                    f"🛡️ [GSociety BREAKEVEN ACTIVATED]\n"
                    f"Trigger: {reason}\n"
                    f"Position: #{pos_id} ({side} - {pos.get('tag', 'Order')})\n"
                    f"SL shifted to: {be_level:.2f}"
                )

    async def _monitor_position(self, order_id: int):
        """Monitors active G Society trade with pip milestone scaling, breakeven, and profit trailing."""
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
                        p_tp = float(p.get("take_profit", 0) or p.get("tp", 0) or 0)
                        if p_tp > 0 and abs(p_tp - pos["tp"]) < 0.2:
                            pos["position_id"] = p_id
                            resolved = True
                            break
                        elif not pos.get("position_id"):
                            pos["position_id"] = p_id
                            resolved = True
            except Exception as e:
                logger.warning(f"[GSociety] Position lookup error: {e}")
            if not pos.get("position_id"):
                await asyncio.sleep(4)

        if not resolved or not pos.get("position_id"):
            logger.warning(f"⚠️ [GSociety] Order #{order_id} failed to map to an active position. Pruning from tracking.")
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
                logger.warning(f"[GSociety] Notice setting post-fill SL/TP on #{pos_id}: {e}")

        logger.info(f"🛡️ [GSociety] Monitoring position #{pos_id or order_id} ({tag}) with milestone trailing & Breakeven.")

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
                logger.warning(f"[GSociety] Error listing positions: {e}")
                continue

            if not is_still_open:
                logger.info(f"📊 [GSociety] Position #{pos_id or order_id} ({tag}) closed! Resolving settlement...")
                await self._log_trade_closure(order_id)
                self.open_positions.pop(order_id, None)
                break

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
                            logger.info(f"🛡️ [GSociety +30 Pips] Risk cut 50% on #{pos_id}! SL: {half_risk_sl}")
                            await self.notify(
                                f"🛡️ [GSociety DEFENSE +30 PIPS]\n"
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
                            logger.info(f"🛡️ [GSociety +50 Pips / 1R] Breakeven activated on #{pos_id}! SL: {be_level}")
                            await self.notify(
                                f"🛡️ [GSociety BREAKEVEN +50 PIPS]\n"
                                f"Position #{pos_id} ({tag})\n"
                                f"Trade is now Risk-Free! SL shifted to: {be_level:.2f}"
                            )

                    # Stage 3: +100 Pips ($10.00) -> Lock in +50 Pips profit
                    if gain_pips >= 100.0 and stage < 3:
                        lock_50 = round(entry + 5.00 if side == "BUY" else entry - 5.00, 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=lock_50)
                        if not res.get("error"):
                            pos["sl"] = lock_50
                            pos["trailing_stage"] = 3
                            logger.info(f"💰 [GSociety +100 Pips] Secured +50 Pips on #{pos_id}! SL: {lock_50}")
                            await self.notify(
                                f"💰 [GSociety PROFIT LOCK +100 PIPS]\n"
                                f"Position #{pos_id} ({tag})\n"
                                f"Banked +50 Pips profit! New SL: {lock_50:.2f}"
                            )

                    # Stage 4: +150 Pips ($15.00) -> Lock in +100 Pips profit
                    if gain_pips >= 150.0 and stage < 4:
                        lock_100 = round(entry + 10.00 if side == "BUY" else entry - 10.00, 2)
                        res = self.mcp.change_position_stop_loss(position_id=pos_id, level=lock_100)
                        if not res.get("error"):
                            pos["sl"] = lock_100
                            pos["trailing_stage"] = 4
                            logger.info(f"💰 [GSociety +150 Pips] Secured +100 Pips on #{pos_id}! SL: {lock_100}")
                            await self.notify(
                                f"💰 [GSociety PROFIT LOCK +150 PIPS]\n"
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
                                logger.info(f"🎯 [GSociety Trailing Stop] Trailing SL updated to {trail_sl:.2f} on #{pos_id}")

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
            logger.warning(f"[GSociety] Trade history lookup error: {e}")

        bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
        eq = bal.get("equity", 0.0) if bal else 0.0

        try:
            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": f"G Society Gold ({tag})",
                "side": side,
                "lots": lots,
                "entry_price": entry_px,
                "stop_loss": sl,
                "take_profit": tp,
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * 10, 1) if exit_px else 0.0,
                "risk_reward": "G Society Signal",
                "exit_reason": reason,
                "position_id": pos_id or order_id,
                "balance_equity": eq
            })
        except Exception as e:
            logger.warning(f"[GSociety] GSheet log error: {e}")

        emoji = "🏆 WIN" if pnl > 0 else "❌ LOSS"
        await self.notify(
            f"{emoji} [GSociety SETTLED — {tag}]\n"
            f"Side    : {side} ({lots} Lots)\n"
            f"Entry   : {entry_px:.2f}\n"
            f"Exit    : {exit_px:.2f}\n"
            f"PnL     : ${pnl:+.2f}\n"
            f"Reason  : {reason}"
        )

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
                if age_sec > 300:
                    logger.info(f"⏰ [GSociety] Skipped historical signal #{message_id} ({int(age_sec)}s old during lookback)")
                    return
            except Exception as e:
                logger.warning(f"[GSociety] Error checking message date: {e}")

        # 2. Check for Breakeven instructions
        inst = GSocietyParser.parse_instruction(text)
        if inst and inst["type"] == "BREAKEVEN":
            logger.info("📢 [GSociety] Received Secure Profit / Breakeven command from channel!")
            await self.trigger_manual_breakeven(reason="Channel Broadcast")
            return

        # 3. Check for trade signal
        sig = GSocietyParser.parse_signal(text)
        if sig and sig.get("type") == "SIGNAL":
            logger.info(f"🎯 [GSociety] New signal parsed: {sig['side']} | SL: {sig['sl']} | TP1: {sig['tp1']} | TP2: {sig['tp2']}")
            await self.execute_signal(sig)
        else:
            logger.debug(f"ℹ️ [GSociety] Message #{message_id} did not match trade signal syntax.")

    def get_status(self) -> Dict[str, Any]:
        return {
            "name": self.name,
            "enabled": self.is_enabled,
            "channel_id": self.channel_id,
            "open_positions_count": len(self.open_positions)
        }
