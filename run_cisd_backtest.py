"""
run_cisd_backtest.py - Advanced ICT + CISD Overhaul Backtester
Tests the upgraded ICT strategy with:
  1. Explicit CISD (Change In State of Delivery)
  2. 3-Candle FVG Imbalance
  3. Premium / Discount Equilibrium Filter
  4. Killzone / Session Window Analysis
  5. Dynamic 1.0R Breakeven Management
"""

import sys
import os
from typing import Optional, Dict, List
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

PROFILES = {
    "XAUUSD": {
        "ticker": "GC=F",
        "name": "Gold (XAU/USD)",
        "sl_buffer": 2.5,
        "disp_threshold": 1.8,
        "min_fvg_gap": 0.3,
        "digits": 2,
        "pip_mult": 10.0
    },
    "EURUSD": {
        "ticker": "EURUSD=X",
        "name": "EUR/USD",
        "sl_buffer": 0.0003,
        "disp_threshold": 0.0004,
        "min_fvg_gap": 0.0001,
        "digits": 5,
        "pip_mult": 10000.0
    },
    "GBPUSD": {
        "ticker": "GBPUSD=X",
        "name": "GBP/USD",
        "sl_buffer": 0.0004,
        "disp_threshold": 0.0005,
        "min_fvg_gap": 0.0001,
        "digits": 5,
        "pip_mult": 10000.0
    },
    "USDJPY": {
        "ticker": "USDJPY=X",
        "name": "USD/JPY",
        "sl_buffer": 0.04,
        "disp_threshold": 0.05,
        "min_fvg_gap": 0.01,
        "digits": 3,
        "pip_mult": 100.0
    },
    "AUDUSD": {
        "ticker": "AUDUSD=X",
        "name": "AUD/USD",
        "sl_buffer": 0.0003,
        "disp_threshold": 0.0004,
        "min_fvg_gap": 0.0001,
        "digits": 5,
        "pip_mult": 10000.0
    }
}

def fetch_asset_data(ticker: str, period: str = "60d", interval: str = "15m") -> Optional[pd.DataFrame]:
    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df is None or len(df) < 50:
            return None
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df.dropna(inplace=True)
        for col in ["Open", "High", "Low", "Close"]:
            df[col] = df[col].astype(float)
        return df
    except Exception as e:
        print(f"Error fetching {ticker}: {e}")
        return None

def backtest_cisd_ict(df: pd.DataFrame, profile: dict, rr_ratio: float = 2.0, 
                      use_cisd: bool = True, use_pd_filter: bool = True,
                      use_session_filter: bool = False, risk_usd: float = 50.0) -> pd.DataFrame:
    sl_buffer = profile["sl_buffer"]
    disp_threshold = profile["disp_threshold"]
    min_fvg_gap = profile["min_fvg_gap"]
    digits = profile["digits"]
    pip_mult = profile["pip_mult"]

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    times = df.index
    n = len(df)

    trades = []
    active_trade = None
    pending_fvg = None

    for i in range(25, n):
        cur_time = times[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        mid = (h + l) / 2.0

        # Session filter: London & NY (07:00 - 19:00 UTC)
        hour = cur_time.hour if hasattr(cur_time, 'hour') else 12
        in_session = (7 <= hour <= 19)

        # ── 1. Manage Active Trade ──────────────────────────────
        if active_trade is not None:
            active_trade['bars_held'] += 1
            side = active_trade['side']
            sl = active_trade['sl']
            tp = active_trade['tp']
            entry = active_trade['entry_price']
            risk_dist = abs(entry - active_trade['initial_sl'])

            # 1.0R Breakeven Trigger
            if not active_trade['moved_to_be']:
                hit_1r = (h - entry >= risk_dist) if side == "BUY" else (entry - l >= risk_dist)
                if hit_1r:
                    be_buf = min_fvg_gap * 0.5
                    active_trade['sl'] = round(entry + be_buf if side == "BUY" else entry - be_buf, digits)
                    active_trade['moved_to_be'] = True
                    sl = active_trade['sl']

            # Check SL / TP hits
            if side == "BUY":
                if l <= sl:
                    active_trade['exit_price'] = sl
                    active_trade['exit_time'] = cur_time
                    if active_trade['moved_to_be']:
                        active_trade['result'] = "BREAKEVEN"
                        active_trade['pnl_usd'] = 0.0
                    else:
                        active_trade['result'] = "LOSS"
                        active_trade['pnl_usd'] = -risk_usd
                    active_trade['pips'] = (sl - entry) * pip_mult
                    trades.append(active_trade)
                    active_trade = None
                elif h >= tp:
                    active_trade['exit_price'] = tp
                    active_trade['exit_time'] = cur_time
                    active_trade['result'] = "WIN"
                    active_trade['pnl_usd'] = risk_usd * rr_ratio
                    active_trade['pips'] = (tp - entry) * pip_mult
                    trades.append(active_trade)
                    active_trade = None
            else: # SELL
                if h >= sl:
                    active_trade['exit_price'] = sl
                    active_trade['exit_time'] = cur_time
                    if active_trade['moved_to_be']:
                        active_trade['result'] = "BREAKEVEN"
                        active_trade['pnl_usd'] = 0.0
                    else:
                        active_trade['result'] = "LOSS"
                        active_trade['pnl_usd'] = -risk_usd
                    active_trade['pips'] = (entry - sl) * pip_mult
                    trades.append(active_trade)
                    active_trade = None
                elif l <= tp:
                    active_trade['exit_price'] = tp
                    active_trade['exit_time'] = cur_time
                    active_trade['result'] = "WIN"
                    active_trade['pnl_usd'] = risk_usd * rr_ratio
                    active_trade['pips'] = (entry - tp) * pip_mult
                    trades.append(active_trade)
                    active_trade = None

            continue

        # ── 2. Manage Pending FVG Retest ────────────────────────
        if pending_fvg is not None:
            pending_fvg['bars_waited'] += 1
            if pending_fvg['bars_waited'] > 12: # 12 bars * 15m = 3 hours expiry
                pending_fvg = None
            else:
                side = pending_fvg['side']
                fvg_h = pending_fvg['fvg_high']
                fvg_l = pending_fvg['fvg_low']
                sl = pending_fvg['sl']
                eq_level = pending_fvg.get('eq_level', 0)

                in_fvg = (fvg_l <= mid <= fvg_h) or (side == "BUY" and l <= fvg_h and h >= fvg_l) or (side == "SELL" and h >= fvg_l and l <= fvg_h)
                
                # Check Discount (BUY) or Premium (SELL) if enabled
                pd_ok = True
                if use_pd_filter and eq_level > 0:
                    if side == "BUY" and mid > eq_level:
                        pd_ok = False # must retest in discount
                    elif side == "SELL" and mid < eq_level:
                        pd_ok = False # must retest in premium

                if in_fvg and pd_ok:
                    exec_px = fvg_h if side == "BUY" else fvg_l
                    risk_dist = abs(exec_px - sl)
                    if risk_dist >= (min_fvg_gap * 0.5):
                        tp = round(exec_px + (risk_dist * rr_ratio) if side == "BUY" else exec_px - (risk_dist * rr_ratio), digits)
                        active_trade = {
                            'entry_time': cur_time,
                            'side': side,
                            'entry_price': exec_px,
                            'initial_sl': sl,
                            'sl': sl,
                            'tp': tp,
                            'risk_usd': risk_usd,
                            'moved_to_be': False,
                            'bars_held': 0
                        }
                        pending_fvg = None
                        continue

        # ── 3. Scan for New Setup ───────────────────────────────
        if use_session_filter and not in_session:
            continue

        recent_high = max(highs[i-22:i-2])
        recent_low = min(lows[i-22:i-2])

        # ── Bearish Setup (High swept + CISD + Bearish FVG) ──────
        swept_h = (highs[i-2] > recent_high and closes[i-2] < recent_high) or \
                  (highs[i-1] > recent_high and closes[i-1] < recent_high)
        has_bearish_fvg = lows[i-2] > (highs[i] + min_fvg_gap)
        disp_down = closes[i-1] < opens[i-1] and (highs[i-1] - lows[i-1]) > disp_threshold

        # Pure ICT CISD: Displacement closes below the Open of the high-forming candle
        cisd_bearish = True
        if use_cisd:
            sweep_candle_open = opens[i-2] if highs[i-2] >= highs[i-1] else opens[i-1]
            cisd_bearish = (closes[i-1] < sweep_candle_open) or (closes[i] < sweep_candle_open)

        if swept_h and has_bearish_fvg and disp_down and cisd_bearish:
            sweep_peak = max(highs[i-2], highs[i-1])
            disp_low = min(lows[i-1], lows[i])
            eq_level = (sweep_peak + disp_low) / 2.0
            sl = round(sweep_peak + sl_buffer, digits)
            pending_fvg = {
                'side': "SELL",
                'fvg_high': round(lows[i-2], digits),
                'fvg_low': round(highs[i], digits),
                'eq_level': eq_level,
                'sl': sl,
                'bars_waited': 0
            }
            continue

        # ── Bullish Setup (Low swept + CISD + Bullish FVG) ──────
        swept_l = (lows[i-2] < recent_low and closes[i-2] > recent_low) or \
                  (lows[i-1] < recent_low and closes[i-1] > recent_low)
        has_bullish_fvg = highs[i-2] < (lows[i] - min_fvg_gap)
        disp_up = closes[i-1] > opens[i-1] and (highs[i-1] - lows[i-1]) > disp_threshold

        # Pure ICT CISD: Displacement closes above the Open of the low-forming candle
        cisd_bullish = True
        if use_cisd:
            sweep_candle_open = opens[i-2] if lows[i-2] <= lows[i-1] else opens[i-1]
            cisd_bullish = (closes[i-1] > sweep_candle_open) or (closes[i] > sweep_candle_open)

        if swept_l and has_bullish_fvg and disp_up and cisd_bullish:
            sweep_trough = min(lows[i-2], lows[i-1])
            disp_high = max(highs[i-1], highs[i])
            eq_level = (sweep_trough + disp_high) / 2.0
            sl = round(sweep_trough - sl_buffer, digits)
            pending_fvg = {
                'side': "BUY",
                'fvg_high': round(lows[i], digits),
                'fvg_low': round(highs[i-2], digits),
                'eq_level': eq_level,
                'sl': sl,
                'bars_waited': 0
            }
            continue

    return pd.DataFrame(trades)

def summarize(df_trades: pd.DataFrame, label: str) -> dict:
    if df_trades is None or len(df_trades) == 0:
        return {"Model": label, "Trades": 0, "Wins": 0, "Losses": 0, "BE": 0, "Effective WR": "0.0%", "PnL ($)": 0.0, "ROC (%)": "0.0%"}
    
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    losses = len(df_trades[df_trades['result'] == 'LOSS'])
    bes = len(df_trades[df_trades['result'] == 'BREAKEVEN'])
    total = len(df_trades)
    eff_wr = (wins / max(1, (wins + losses))) * 100.0
    pnl = df_trades['pnl_usd'].sum()
    roc = (pnl / 1000.0) * 100.0

    return {
        "Model": label,
        "Trades": total,
        "Wins": wins,
        "Losses": losses,
        "BE": bes,
        "Effective WR": f"{eff_wr:.1f}%",
        "PnL ($)": round(pnl, 2),
        "ROC (%)": f"{roc:+.1f}%"
    }

def main():
    print("=" * 85)
    print("🚀 ADVANCED ICT + CISD OVERHAUL BACKTEST (60-Day 15M Candles)")
    print("=" * 85)

    for symbol, profile in PROFILES.items():
        print(f"\n📊 Fetching data for {profile['name']} ({profile['ticker']})...")
        df = fetch_asset_data(profile["ticker"])
        if df is None:
            print(f"❌ Failed to fetch data for {symbol}")
            continue

        # 1. Baseline ICT
        base_trades = backtest_cisd_ict(df, profile, use_cisd=False, use_pd_filter=False)
        base_res = summarize(base_trades, "1. Baseline ICT")

        # 2. ICT + CISD
        cisd_trades = backtest_cisd_ict(df, profile, use_cisd=True, use_pd_filter=False)
        cisd_res = summarize(cisd_trades, "2. ICT + CISD")

        # 3. ICT + CISD + Premium/Discount Equilibrium
        full_trades = backtest_cisd_ict(df, profile, use_cisd=True, use_pd_filter=True)
        full_res = summarize(full_trades, "3. ICT + CISD + PD Equilibrium")

        # 4. ICT + CISD + PD + Session Filter
        sess_trades = backtest_cisd_ict(df, profile, use_cisd=True, use_pd_filter=True, use_session_filter=True)
        sess_res = summarize(sess_trades, "4. ICT + CISD + PD + Session")

        res_df = pd.DataFrame([base_res, cisd_res, full_res, sess_res])
        print(f"\n--- Results for {profile['name']} ---")
        print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
