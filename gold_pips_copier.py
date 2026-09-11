"""
gold_pips_copier.py - Automated Telegram Signal Copier for Gold Pips Hunter

Listens to Telegram Channel 'Gold Pips Hunter' (-1003679078163) in real-time,
parses Gold/XAUUSD signals, and executes them on IQ Option's Marginal CFD engine
with Stop Loss, Take Profit, and automatic Breakeven management.

Features:
- Telethon integration with authenticated user session ('user_desktop_session').
- Direct execution via IQ Option Marginal CFD MCP Gateway (https://marginal-cfd.mcp.iqoption.com).
- Asset: Gold / XAUUSD (Asset ID: 74, Instrument: mcfd.74).
- Automatic Stop Loss (SL) and Take Profit (TP1/TP2/TP3) enforcement.
- Real-time Breakeven / Partial Close instruction handling.
- Integrated Google Sheets logging to 'Forex_Margin_Trades'.
"""

import os
import sys
import time
import re
import json
import logging
import asyncio
import argparse
from datetime import datetime, timezone
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Ensure root directory is on path
sys.path.append(os.getcwd())

# Force UTF-8 on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")

load_dotenv()

from telethon import TelegramClient, events
from forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GoldPipsCopier")

API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")
SESSION_NAME = "user_desktop_session"
GOLD_CHANNEL_ID = -1003679078163  # Gold Pips Hunter

# IQ Option CFD details for Gold
GOLD_ASSET_ID = 74
GOLD_INSTRUMENT_ID = "mcfd.74"

class GoldSignalParser:
    """Parses trade signals and management instructions from Gold Pips Hunter."""

    @staticmethod
    def parse_signal(text: str) -> Optional[Dict[str, Any]]:
        """
        Parses signals matching:
        🟢 Gold Buy Now @ 4394-4390
        SL: 4382
        TP1: 4397
        TP2: 4400
        TP3: 4403
        """
        clean = text.lower()
        if not any(k in clean for k in ["gold", "xau", "xauusd"]):
            return None

        # Determine side
        side = None
        if "buy" in clean or "long" in clean:
            side = "BUY"
        elif "sell" in clean or "short" in clean:
            side = "SELL"

        if not side:
            return None

        # Stop Loss
        sl_match = re.search(r'(?:sl|stop\s*loss)[:\s]*([0-9]+(?:\.[0-9]+)?)', clean)
        sl = float(sl_match.group(1)) if sl_match else None

        # Take Profit levels
        tp1_match = re.search(r'(?:tp1|take\s*profit\s*1?)[:\s]*([0-9]+(?:\.[0-9]+)?)', clean)
        tp1 = float(tp1_match.group(1)) if tp1_match else None

        tp2_match = re.search(r'(?:tp2|take\s*profit\s*2)[:\s]*([0-9]+(?:\.[0-9]+)?)', clean)
        tp2 = float(tp2_match.group(1)) if tp2_match else None

        tp3_match = re.search(r'(?:tp3|take\s*profit\s*3)[:\s]*([0-9]+(?:\.[0-9]+)?)', clean)
        tp3 = float(tp3_match.group(1)) if tp3_match else None

        # Generic TP fallback if TP1 wasn't specified
        if not tp1:
            tp_gen = re.search(r'(?:tp|take\s*profit)[:\s]*([0-9]+(?:\.[0-9]+)?)', clean)
            if tp_gen:
                tp1 = float(tp_gen.group(1))

        # Entry Range
        entry_match = re.search(r'(?:@|at|entry)[:\s]*([0-9]+(?:\.[0-9]+)?)(?:\s*-\s*([0-9]+(?:\.[0-9]+)?))?', clean)
        entry_min = float(entry_match.group(1)) if entry_match and entry_match.group(1) else None
        entry_max = float(entry_match.group(2)) if entry_match and entry_match.group(2) else entry_min

        return {
            "type": "NEW_SIGNAL",
            "asset": "Gold",
            "side": side,
            "sl": sl,
            "tp1": tp1,
            "tp2": tp2,
            "tp3": tp3,
            "entry_min": entry_min,
            "entry_max": entry_max,
            "raw_text": text
        }

    @staticmethod
    def parse_instruction(text: str) -> Optional[Dict[str, str]]:
        """
        Parses mid-trade updates:
        - 'Hold the positions with breakeven' / 'Move SL to BE'
        - 'Close all positions' / 'Close Gold'
        """
        clean = text.lower()
        if any(k in clean for k in ["breakeven", "break even", "move sl to entry", "sl to be"]):
            return {"type": "BREAKEVEN", "raw_text": text}
        if any(k in clean for k in ["close all", "close positions", "close gold", "exit all"]):
            return {"type": "CLOSE_ALL", "raw_text": text}
        return None

class GoldPipsCopier:
    def __init__(self, account_type: str = "training", lots: float = 1.0,
                 leverage: int = 100, tp_target: int = 1, max_slippage: float = 4.0):
        self.account_type = account_type.lower()
        self.lots = lots
        self.leverage = leverage
        self.tp_target = tp_target  # 1 for TP1, 2 for TP2, 3 for TP3
        self.max_slippage = max_slippage
        
        # Connect to IQ Option Marginal CFD MCP Gateway
        self.mcp_client = IQForexMCPClient(base_url="https://marginal-cfd.mcp.iqoption.com")
        self.balance_id: Optional[int] = None
        self.tele_client: Optional[TelegramClient] = None
        self.active_gold_positions: Dict[int, Dict[str, Any]] = {}

    def init_iq(self) -> bool:
        """Initialize IQ Option Marginal CFD session."""
        logger.info("Connecting to IQ Option Marginal CFD MCP Gateway...")
        if not self.mcp_client.initialize():
            logger.error("❌ Failed to initialize IQ Option CFD MCP session.")
            return False

        bal = self.mcp_client.get_training_balance() if self.account_type == "training" else self.mcp_client.get_real_balance()
        if not bal:
            logger.error("❌ Failed to retrieve balance.")
            return False

        self.balance_id = bal["balance_id"]
        logger.info(f"✅ IQ Option CFD Connected! Balance ID: {self.balance_id} | Equity: ${bal['equity']:.2f} | Free Margin: ${bal['free_margin']:.2f}")
        return True

    def get_gold_price(self) -> Dict[str, float]:
        """Fetch current bid/ask price for Gold."""
        preview = self.mcp_client.calculate_order_size(
            asset_id=GOLD_ASSET_ID,
            balance_currency="USD",
            lots=self.lots,
            leverage=self.leverage
        )
        return {
            "buy": float(preview.get("buy_price", 0.0)),
            "sell": float(preview.get("sell_price", 0.0))
        }

    def execute_signal(self, sig: Dict[str, Any]):
        """Place the Gold trade on IQ Option."""
        side = sig["side"].lower()
        prices = self.get_gold_price()
        current_px = prices["buy"] if side == "buy" else prices["sell"]
        if current_px <= 0:
            logger.error("Failed to get current Gold price.")
            return

        # Check entry zone validity if specified
        e_min = sig.get("entry_min")
        e_max = sig.get("entry_max")
        if e_min and e_max:
            low_bound = min(e_min, e_max) - self.max_slippage
            high_bound = max(e_min, e_max) + self.max_slippage
            if not (low_bound <= current_px <= high_bound):
                logger.warning(f"⚠️ [SLIPPAGE SKIP] Current Gold price ({current_px:.2f}) is outside entry zone ({low_bound:.2f}-{high_bound:.2f}).")
                return

        # Select Take Profit based on configuration
        tp = sig.get(f"tp{self.tp_target}") or sig.get("tp1")
        sl = sig.get("sl")

        # Stop loss validation: minimum distance on IQ Option Gold is $1.00
        if sl:
            dist = abs(current_px - sl)
            if dist < 1.0:
                sl = round(current_px - 1.5 if side == "buy" else current_px + 1.5, 2)
        if tp:
            dist = abs(current_px - tp)
            if dist < 1.0:
                tp = round(current_px + 1.5 if side == "buy" else current_px - 1.5, 2)

        logger.info("=" * 60)
        logger.info(f"⚡ [EXECUTING SIGNAL] Gold {side.upper()} @ {current_px:.2f}")
        logger.info(f"   Lots: {self.lots} | Leverage: {self.leverage}x")
        logger.info(f"   Stop Loss:   {sl}")
        logger.info(f"   Take Profit: {tp} (Target: TP{self.tp_target})")
        logger.info("=" * 60)

        order_res = self.mcp_client.place_market_order(
            side=side,
            balance_id=self.balance_id,
            instrument_id=GOLD_INSTRUMENT_ID,
            asset_id=GOLD_ASSET_ID,
            lots=self.lots,
            leverage=self.leverage,
            stop_loss=sl,
            take_profit=tp,
            is_margin_isolated=True,
            keep_position_open=False
        )

        if "order_id" in order_res:
            order_id = order_res["order_id"]
            logger.info(f"✅ [ORDER FILLED] Gold {side.upper()} | Order ID: #{order_id}")
            time.sleep(2)
            self._sync_positions()
        else:
            logger.error(f"❌ Order failed: {order_res}")

    def apply_breakeven(self):
        """Move Stop Loss to Entry Price for all open Gold positions."""
        self._sync_positions()
        if not self.active_gold_positions:
            logger.info("No open Gold positions to move to Breakeven.")
            return

        for pos_id, p in list(self.active_gold_positions.items()):
            open_px = float(p.get("open_price", 0.0))
            if open_px > 0:
                logger.info(f"🛡️ [BREAKEVEN] Moving SL on Position #{pos_id} to Entry: {open_px:.2f}")
                res = self.mcp_client.change_position_stop_loss(position_id=pos_id, level=open_px)
                if not res.get("error"):
                    p["stop_loss"] = open_px

    def close_all_gold(self):
        """Close all open Gold positions."""
        self._sync_positions()
        if not self.active_gold_positions:
            logger.info("No open Gold positions to close.")
            return

        for pos_id in list(self.active_gold_positions.keys()):
            logger.info(f"🔒 [MANUAL CLOSE] Closing Gold Position #{pos_id}...")
            res = self.mcp_client.close_position(position_id=pos_id)
            logger.info(f"Close result: {res}")
            
        time.sleep(2)
        self._sync_positions()

    def _sync_positions(self):
        """Check open positions on CFD and update local dictionary."""
        positions = self.mcp_client.list_positions(balance_id=self.balance_id)
        current_ids = set()
        for p in positions:
            if p.get("asset_id") == GOLD_ASSET_ID:
                pos_id = p.get("position_id") or p.get("id")
                current_ids.add(pos_id)
                if pos_id not in self.active_gold_positions:
                    self.active_gold_positions[pos_id] = p
                    
        # Check closed positions
        closed = [pid for pid in self.active_gold_positions if pid not in current_ids]
        for pid in closed:
            pos_data = self.active_gold_positions.pop(pid, {})
            self._log_closed_position(pid, pos_data)

    def _log_closed_position(self, position_id: int, pos_data: Dict[str, Any]):
        """Log closed trade to Google Sheets."""
        history = self.mcp_client.get_trade_history(balance_id=self.balance_id, limit=10)
        matched = None
        for h in history:
            if h.get("position_id") == position_id:
                matched = h
                break

        pnl = float(matched.get("pnl", 0.0)) if matched else 0.0
        exit_price = float(matched.get("close_price", 0.0)) if matched else 0.0
        reason = matched.get("close_reason", "closed") if matched else "closed"
        entry_price = float(pos_data.get("open_price", 0.0))
        side = pos_data.get("type", "buy").upper()

        logger.info(f"🏆 [TRADE FINISHED] Gold #{position_id} | PnL: ${pnl:.2f} | Reason: {reason} @ {exit_price}")

        gsheet_logger.log_forex_margin_trade({
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "asset": "Gold (XAUUSD)",
            "side": side,
            "lots": self.lots,
            "entry_price": entry_price,
            "stop_loss": pos_data.get("stop_lose_price", 0.0),
            "take_profit": pos_data.get("take_profit_price", 0.0),
            "exit_price": exit_price,
            "pnl": pnl,
            "pips": round(abs(exit_price - entry_price) * 10, 1),
            "risk_reward": f"TP{self.tp_target}",
            "exit_reason": reason,
            "position_id": position_id,
            "balance_equity": self.mcp_client.get_training_balance().get("equity", 0.0)
        })

    async def start_telegram_listener(self):
        """Start listening for incoming posts in Gold Pips Hunter channel with auto-reconnect."""
        logger.info(f"📡 Initializing Telethon listener for 'Gold Pips Hunter' ({GOLD_CHANNEL_ID})...")
        
        while True:
            try:
                self.tele_client = TelegramClient(
                    SESSION_NAME, 
                    API_ID, 
                    API_HASH,
                    connection_retries=None,  # Retry indefinitely
                    retry_delay=5,
                    auto_reconnect=True
                )
                await self.tele_client.connect()

                if not await self.tele_client.is_user_authorized():
                    logger.error("❌ Telegram user session is not authorized. Please run telegram_auth.py first.")
                    return

                me = await self.tele_client.get_me()
                logger.info(f"✅ Telegram Listener Active! Logged in as: {me.first_name} (@{me.username})")
                logger.info(f"🎧 Listening for signals in channel: {GOLD_CHANNEL_ID}...")

                @self.tele_client.on(events.NewMessage(chats=GOLD_CHANNEL_ID))
                async def handler(event):
                    text = event.message.text
                    if not text:
                        return

                    logger.info(f"\n📩 [NEW MESSAGE from Gold Pips Hunter]:\n{text}\n")

                    # 1. Check for management instructions (Breakeven / Close)
                    instr = GoldSignalParser.parse_instruction(text)
                    if instr:
                        if instr["type"] == "BREAKEVEN":
                            logger.info("🛡️ Received Breakeven Instruction!")
                            self.apply_breakeven()
                        elif instr["type"] == "CLOSE_ALL":
                            logger.info("🔒 Received Close All Instruction!")
                            self.close_all_gold()
                        return

                    # 2. Check for trade signals
                    sig = GoldSignalParser.parse_signal(text)
                    if sig:
                        logger.info(f"🎯 Valid Gold Signal Detected: {sig['side']} | SL: {sig['sl']} | TP1: {sig['tp1']}")
                        self.execute_signal(sig)
                    else:
                        logger.info("Message was commentary or update (no actionable signal).")

                # Run until disconnected, then loop will reconnect
                await self.tele_client.run_until_disconnected()

            except (ConnectionError, OSError) as e:
                logger.warning(f"⚠️ Telegram network connection dropped: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
            except Exception as e:
                logger.error(f"❌ Unexpected error in Telegram listener: {e}. Reconnecting in 10s...")
                await asyncio.sleep(10)
            finally:
                if self.tele_client:
                    try:
                        await self.tele_client.disconnect()
                    except Exception:
                        pass


def main():
    parser = argparse.ArgumentParser(description="Gold Pips Hunter Signal Copier")
    parser.add_argument("--account", default="training", choices=["training", "regular"], help="Account type")
    parser.add_argument("--lots", type=float, default=1.0, help="Order size in Gold units (default: 1.0 oz)")
    parser.add_argument("--leverage", type=int, default=100, help="Leverage to use (default: 100)")
    parser.add_argument("--tp-target", type=int, default=1, choices=[1, 2, 3], help="Take Profit target (1, 2, or 3)")
    parser.add_argument("--slippage", type=float, default=4.0, help="Max allowed slippage from entry zone in USD")

    args = parser.parse_args()

    copier = GoldPipsCopier(
        account_type=args.account,
        lots=args.lots,
        leverage=args.leverage,
        tp_target=args.tp_target,
        max_slippage=args.slippage
    )

    if copier.init_iq():
        asyncio.run(copier.start_telegram_listener())

if __name__ == "__main__":
    main()
