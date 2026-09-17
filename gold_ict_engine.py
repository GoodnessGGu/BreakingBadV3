"""
gold_ict_engine.py - Autonomous Institutional ICT / SMC Gold Trading Bot

Trades Gold (XAUUSD / mcfd.74) on IQ Option Marginal CFD engine.
Strategy:
  1. 15M Candlestick Orderflow & Liquidity Sweeps.
  2. Displacement + Fair Value Gap (FVG) creation.
  3. Dynamic Entry Zone on FVG retest.
  4. Auto-Breakeven at 1.0R profit.
  5. 1:2.0 Risk-to-Reward Ratio with tight SL at the sweep extreme.
  6. Live Google Sheets logging to "Forex_Margin_Trades".
"""

import os, sys, time, logging, asyncio, argparse, requests
from datetime import datetime, timezone
from typing import Dict, Any, Optional, List
import pandas as pd
import numpy as np
from dotenv import load_dotenv

sys.path.append(os.getcwd())
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8")
load_dotenv()

from forex_mcp_client import IQForexMCPClient
from gsheet_logger import gsheet_logger

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("GoldICTEngine")

# ── Telegram ───────────────────────────────────────────────────────────────
_TELEGRAM_TOKEN   = os.getenv("TELEGRAM_TOKEN", "")
_TELEGRAM_CHAT_ID = 6420777416  # your personal Telegram ID

def send_telegram_alert(text: str):
    """Fire-and-forget Telegram message. Never raises."""
    if not _TELEGRAM_TOKEN:
        logger.warning("TELEGRAM_TOKEN missing – skipping alert.")
        return
    try:
        r = requests.post(
            f"https://api.telegram.org/bot{_TELEGRAM_TOKEN}/sendMessage",
            json={"chat_id": _TELEGRAM_CHAT_ID, "text": text},
            timeout=10
        )
        if not r.ok:
            logger.warning(f"Telegram alert failed: {r.status_code} {r.text[:100]}")
    except Exception as e:
        logger.warning(f"Telegram alert error: {e}")
# ───────────────────────────────────────────────────────────────────────────

GOLD_ASSET_ID   = 74
GOLD_INSTRUMENT = "mcfd.74"
CANDLE_SIZE     = 900  # 15 minutes (900s)

class GoldICTEngine:
    def __init__(self, account_type="training", lots=1.0, leverage=100,
                 rr_ratio=2.0, sl_buffer=2.5, risk_usd=50.0):
        self.account_type = account_type.lower()
        self.lots         = lots
        self.leverage     = leverage
        self.rr_ratio     = rr_ratio
        self.sl_buffer    = sl_buffer
        self.risk_usd     = risk_usd

        self.mcp          = IQForexMCPClient(base_url="https://marginal-cfd.mcp.iqoption.com")
        self.balance_id   = None
        self.active_trade = None
        self.pending_fvg  = None

    def init_iq(self) -> bool:
        logger.info("Initializing IQ Option Marginal CFD session for ICT Engine...")
        if not self.mcp.initialize():
            logger.error("Failed to initialize IQ Option MCP.")
            return False
        bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
        if not bal:
            logger.error("Failed to get balance.")
            return False
        self.balance_id = bal["balance_id"]
        logger.info(f"Connected! Balance ID: {self.balance_id} | Equity: ${bal['equity']:.2f}")
        return True

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
            logger.warning(f"Price fetch error: {e}")
            return {"buy": 0.0, "sell": 0.0, "mid": 0.0}

    def fetch_recent_candles(self, count=50) -> Optional[pd.DataFrame]:
        try:
            candles = self.mcp.get_candles(asset_id=GOLD_ASSET_ID, size=CANDLE_SIZE, count=count)
            if not candles or len(candles) < 15:
                return None
            df = pd.DataFrame(candles)
            df.rename(columns={"open": "Open", "close": "Close", "min": "Low", "max": "High"}, inplace=True)
            for col in ["Open", "Close", "Low", "High"]:
                df[col] = df[col].astype(float)
            return df
        except Exception as e:
            logger.error(f"Error fetching candles: {e}")
            return None

    def scan_for_setups(self, df: pd.DataFrame):
        """Analyze last 15M candles for Liquidity Sweep + FVG."""
        if self.active_trade or self.pending_fvg:
            return

        highs = df['High'].values
        lows = df['Low'].values
        closes = df['Close'].values
        opens = df['Open'].values
        n = len(df)

        # Identify previous swing high/low (looking back 5 to 25 bars)
        recent_high = max(highs[-25:-5])
        recent_low = min(lows[-25:-5])

        # 1. Bearish Liquidity Sweep (High swept + Bearish FVG)
        # Bar -3 or -2 swept recent_high, but closed below
        swept_h = (highs[-3] > recent_high and closes[-3] < recent_high) or \
                  (highs[-2] > recent_high and closes[-2] < recent_high)
        has_bearish_fvg = lows[-3] > (highs[-1] + 0.3)
        disp_down = closes[-2] < opens[-2] and (highs[-2] - lows[-2]) > 2.0

        if swept_h and has_bearish_fvg and disp_down:
            sweep_peak = max(highs[-3], highs[-2])
            sl = round(sweep_peak + self.sl_buffer, 2)
            fvg_h = round(lows[-3], 2)
            fvg_l = round(highs[-1], 2)
            logger.info("=" * 60)
            logger.info(f"🔥 [ICT SETUP DETECTED] Bearish Liquidity Sweep at {sweep_peak:.2f}!")
            logger.info(f"   Bearish FVG Zone : {fvg_l:.2f} - {fvg_h:.2f}")
            logger.info(f"   Stop Loss        : {sl:.2f}")
            logger.info("=" * 60)
            send_telegram_alert(
                f"🔥 ICT SETUP DETECTED — SELL\n"
                f"Bearish Liquidity Sweep @ {sweep_peak:.2f}\n"
                f"FVG Zone : {fvg_l:.2f} – {fvg_h:.2f}\n"
                f"Stop Loss: {sl:.2f}\n"
                f"⏳ Waiting for price to retest FVG..."
            )
            self.pending_fvg = {
                "side": "SELL",
                "fvg_high": fvg_h,
                "fvg_low": fvg_l,
                "sl": sl,
                "detected_at": time.time()
            }
            return

        # 2. Bullish Liquidity Sweep (Low swept + Bullish FVG)
        swept_l = (lows[-3] < recent_low and closes[-3] > recent_low) or \
                  (lows[-2] < recent_low and closes[-2] > recent_low)
        has_bullish_fvg = highs[-3] < (lows[-1] - 0.3)
        disp_up = closes[-2] > opens[-2] and (highs[-2] - lows[-2]) > 2.0

        if swept_l and has_bullish_fvg and disp_up:
            sweep_trough = min(lows[-3], lows[-2])
            sl = round(sweep_trough - self.sl_buffer, 2)
            fvg_l = round(highs[-3], 2)
            fvg_h = round(lows[-1], 2)
            logger.info("=" * 60)
            logger.info(f"🔥 [ICT SETUP DETECTED] Bullish Liquidity Sweep at {sweep_trough:.2f}!")
            logger.info(f"   Bullish FVG Zone : {fvg_l:.2f} - {fvg_h:.2f}")
            logger.info(f"   Stop Loss        : {sl:.2f}")
            logger.info("=" * 60)
            send_telegram_alert(
                f"🔥 ICT SETUP DETECTED — BUY\n"
                f"Bullish Liquidity Sweep @ {sweep_trough:.2f}\n"
                f"FVG Zone : {fvg_l:.2f} – {fvg_h:.2f}\n"
                f"Stop Loss: {sl:.2f}\n"
                f"⏳ Waiting for price to retest FVG..."
            )
            self.pending_fvg = {
                "side": "BUY",
                "fvg_high": fvg_h,
                "fvg_low": fvg_l,
                "sl": sl,
                "detected_at": time.time()
            }
            return

    def check_fvg_retest_and_enter(self, cur_prices: Dict[str, float]):
        """If price taps the active FVG zone, execute trade immediately."""
        if not self.pending_fvg or self.active_trade:
            return

        mid = cur_prices["mid"]
        side = self.pending_fvg["side"]
        fvg_h = self.pending_fvg["fvg_high"]
        fvg_l = self.pending_fvg["fvg_low"]
        sl = self.pending_fvg["sl"]

        # Expire after 3 hours if not tapped
        if time.time() - self.pending_fvg["detected_at"] > (3 * 3600):
            logger.info("FVG setup expired without retest. Resetting.")
            self.pending_fvg = None
            return

        in_fvg = (fvg_l <= mid <= fvg_h)
        if in_fvg:
            exec_px = cur_prices["buy"] if side == "BUY" else cur_prices["sell"]
            risk_dist = abs(exec_px - sl)
            if risk_dist < 1.0:
                return

            tp = round(exec_px + (risk_dist * self.rr_ratio) if side == "BUY" else exec_px - (risk_dist * self.rr_ratio), 2)
            logger.info("=" * 60)
            logger.info(f"⚡ [ICT EXECUTION] Tapped FVG Zone! Firing {side} @ {exec_px:.2f}")
            logger.info(f"   SL: {sl:.2f} | TP (1:{self.rr_ratio:.1f} RR): {tp:.2f}")
            logger.info("=" * 60)
            send_telegram_alert(
                f"⚡ ICT ORDER FIRING — {side}\n"
                f"Entry : {exec_px:.2f} (FVG retest)\n"
                f"SL    : {sl:.2f}\n"
                f"TP    : {tp:.2f}  (1:{self.rr_ratio:.1f} RR)\n"
                f"Lots  : {self.lots}"
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
                logger.info(f"✅ ICT Order Filled! ID: #{order_id}")
                send_telegram_alert(
                    f"✅ ICT ORDER CONFIRMED — {side}\n"
                    f"Order ID : #{order_id}\n"
                    f"Entry    : {exec_px:.2f}\n"
                    f"SL       : {sl:.2f}\n"
                    f"TP       : {tp:.2f}\n"
                    f"🔄 Managing trade now..."
                )
                self.active_trade = {
                    "order_id": order_id,
                    "position_id": None,
                    "side": side,
                    "entry_price": exec_px,
                    "initial_sl": sl,
                    "current_sl": sl,
                    "tp": tp,
                    "moved_to_be": False,
                    "opened_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                }
                self.pending_fvg = None
            else:
                logger.error(f"Order placement failed: {res}")
                send_telegram_alert(f"❌ ICT ORDER FAILED — {side}\nResponse: {str(res)[:200]}")

    def manage_active_trade(self, cur_prices: Dict[str, float]):
        """Breakeven management: once trade reaches 1.0R, move SL to entry."""
        if not self.active_trade:
            return

        # Find position_id if not cached
        if not self.active_trade["position_id"]:
            positions = self.mcp.list_positions(balance_id=self.balance_id)
            for p in positions:
                if p.get("asset_id") == GOLD_ASSET_ID:
                    self.active_trade["position_id"] = p.get("position_id") or p.get("id")
                    break

        pos_id = self.active_trade["position_id"]
        if not pos_id:
            return

        # Check if position is still open
        open_positions = self.mcp.list_positions(balance_id=self.balance_id)
        is_still_open = any((p.get("position_id") or p.get("id")) == pos_id for p in open_positions)

        if not is_still_open:
            logger.info(f"ICT Trade #{pos_id} closed! Syncing Google Sheets...")
            self._log_trade_closure(pos_id)
            self.active_trade = None
            return

        # Breakeven check
        if not self.active_trade["moved_to_be"]:
            mid = cur_prices["mid"]
            entry = self.active_trade["entry_price"]
            risk_dist = abs(entry - self.active_trade["initial_sl"])
            side = self.active_trade["side"]

            hit_1r = (mid - entry >= risk_dist) if side == "BUY" else (entry - mid >= risk_dist)
            if hit_1r:
                be_level = round(entry + 0.5 if side == "BUY" else entry - 0.5, 2)
                logger.info(f"🛡️ [ICT BREAKEVEN] Reached 1.0R profit! Moving SL to {be_level:.2f}")
                res = self.mcp.change_position_stop_loss(position_id=pos_id, level=be_level)
                if not res.get("error"):
                    self.active_trade["current_sl"] = be_level
                    self.active_trade["moved_to_be"] = True

    def _log_trade_closure(self, pos_id):
        try:
            history = self.mcp.get_trade_history(balance_id=self.balance_id, limit=5)
            matched = next((h for h in history if h.get("position_id") == pos_id), None)
            pnl = float(matched.get("pnl", 0.0)) if matched else 0.0
            exit_px = float(matched.get("close_price", 0.0)) if matched else 0.0
            reason = matched.get("close_reason", "closed") if matched else "closed"
            entry_px = self.active_trade["entry_price"]

            bal = self.mcp.get_training_balance() if self.account_type == "training" else self.mcp.get_real_balance()
            eq = bal.get("equity", 0.0) if bal else 0.0

            gsheet_logger.log_forex_margin_trade({
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                "asset": "Gold ICT Autonomous",
                "side": self.active_trade["side"],
                "lots": self.lots,
                "entry_price": entry_px,
                "stop_loss": self.active_trade["current_sl"],
                "take_profit": self.active_trade["tp"],
                "exit_price": exit_px,
                "pnl": pnl,
                "pips": round(abs(exit_px - entry_px) * 10, 1),
                "risk_reward": f"1:{self.rr_ratio:.1f} (ICT Autonomous)",
                "exit_reason": reason,
                "position_id": pos_id,
                "balance_equity": eq
            })
            logger.info(f"Logged ICT trade #{pos_id} to Google Sheets! PnL: ${pnl:.2f}")
        except Exception as e:
            logger.error(f"Error logging trade: {e}")

    async def run(self):
        logger.info("🚀 Starting Autonomous ICT Gold Engine Loop...")
        last_candle_scan = 0

        while True:
            try:
                now = time.time()
                prices = self.get_market_price()

                if prices["mid"] > 0:
                    # Scan for new 15M setups every 60 seconds
                    if now - last_candle_scan > 60:
                        df = self.fetch_recent_candles(count=50)
                        if df is not None:
                            self.scan_for_setups(df)
                        last_candle_scan = now

                    # Check FVG retest entry
                    self.check_fvg_retest_and_enter(prices)

                    # Manage open trade (Breakeven)
                    self.manage_active_trade(prices)

                await asyncio.sleep(5) # 5-second tick loop
            except Exception as e:
                logger.error(f"Unexpected loop error: {e}")
                await asyncio.sleep(10)

def main():
    parser = argparse.ArgumentParser(description="Autonomous ICT Gold Trading Engine")
    parser.add_argument("--account", default="training", choices=["training", "regular"])
    parser.add_argument("--lots", type=float, default=1.0)
    parser.add_argument("--leverage", type=int, default=100)
    parser.add_argument("--rr", type=float, default=2.0)
    args = parser.parse_args()

    engine = GoldICTEngine(
        account_type=args.account,
        lots=args.lots,
        leverage=args.leverage,
        rr_ratio=args.rr
    )
    if engine.init_iq():
        asyncio.run(engine.run())

if __name__ == "__main__":
    main()