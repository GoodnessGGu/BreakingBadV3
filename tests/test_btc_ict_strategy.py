"""
tests/test_btc_ict_strategy.py - Bitcoin (BTC/USD) ICT / SMC Strategy Engine & Backtester
Optimizes and validates:
  1. 15M Liquidity Sweeps + CISD (Change In State of Delivery)
  2. BTC Volatility-Calibrated Displacement ($100 - $500)
  3. Fair Value Gap (FVG) Imbalances & Retests
  4. 1.0R Breakeven Management
  5. Optimal Risk-to-Reward (1:2.0, 1:2.5, 1:3.0)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime

def fetch_btc_data() -> pd.DataFrame:
    print("📥 Downloading Bitcoin (BTC-USD) 15M candles (60 days)...")
    df = yf.download("BTC-USD", period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def backtest_btc_ict(df: pd.DataFrame,
                     lookback: int = 24,
                     disp_threshold: float = 250.0,
                     min_fvg_gap: float = 35.0,
                     sl_buffer: float = 60.0,
                     rr_ratio: float = 2.5,
                     body_ratio_req: float = 0.50,
                     use_cisd: bool = True,
                     risk_usd: float = 50.0) -> pd.DataFrame:
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    times = df.index
    n = len(df)

    trades = []
    active_trade = None
    pending_fvg = None

    for i in range(50, n):
        cur_time = times[i]
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        mid = (h + l) / 2.0

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
                    active_trade['sl'] = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)
                    active_trade['moved_to_be'] = True
                    sl = active_trade['sl']

            if side == "BUY":
                if l <= sl:
                    active_trade['result'] = "BREAKEVEN" if active_trade['moved_to_be'] else "LOSS"
                    active_trade['pnl_usd'] = 0.0 if active_trade['moved_to_be'] else -risk_usd
                    trades.append(active_trade)
                    active_trade = None
                elif h >= tp:
                    active_trade['result'] = "WIN"
                    active_trade['pnl_usd'] = risk_usd * rr_ratio
                    trades.append(active_trade)
                    active_trade = None
            else: # SELL
                if h >= sl:
                    active_trade['result'] = "BREAKEVEN" if active_trade['moved_to_be'] else "LOSS"
                    active_trade['pnl_usd'] = 0.0 if active_trade['moved_to_be'] else -risk_usd
                    trades.append(active_trade)
                    active_trade = None
                elif l <= tp:
                    active_trade['result'] = "WIN"
                    active_trade['pnl_usd'] = risk_usd * rr_ratio
                    trades.append(active_trade)
                    active_trade = None

            continue

        # ── 2. Manage Pending FVG Retest ────────────────────────
        if pending_fvg is not None:
            pending_fvg['bars_waited'] += 1
            if pending_fvg['bars_waited'] > 16: # 4 hours expiry
                pending_fvg = None
            else:
                side = pending_fvg['side']
                fvg_h = pending_fvg['fvg_high']
                fvg_l = pending_fvg['fvg_low']
                sl = pending_fvg['sl']

                in_fvg = (fvg_l <= mid <= fvg_h) or (side == "BUY" and l <= fvg_h and h >= fvg_l) or (side == "SELL" and h >= fvg_l and l <= fvg_h)
                if in_fvg:
                    exec_px = fvg_h if side == "BUY" else fvg_l
                    risk_dist = abs(exec_px - sl)
                    if risk_dist >= (min_fvg_gap * 0.5):
                        tp = round(exec_px + (risk_dist * rr_ratio) if side == "BUY" else exec_px - (risk_dist * rr_ratio), 2)
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
        recent_high = max(highs[i-lookback:i-2])
        recent_low = min(lows[i-lookback:i-2])

        # Bearish Setup (SELL)
        swept_h = (highs[i-2] > recent_high and closes[i-2] < recent_high) or \
                  (highs[i-1] > recent_high and closes[i-1] < recent_high)
        has_bearish_fvg = lows[i-2] > (highs[i] + min_fvg_gap)
        disp_range_down = highs[i-1] - lows[i-1]
        disp_body_down = abs(closes[i-1] - opens[i-1])
        disp_down = closes[i-1] < opens[i-1] and disp_range_down > disp_threshold
        body_ok_down = (disp_body_down / max(0.0001, disp_range_down)) >= body_ratio_req

        # Explicit CISD: Displacement closes below the Open of the high-forming candle
        sweep_open_h = opens[i-2] if highs[i-2] >= highs[i-1] else opens[i-1]
        cisd_down = (not use_cisd) or (closes[i-1] < sweep_open_h) or (closes[i] < sweep_open_h)

        if swept_h and has_bearish_fvg and disp_down and body_ok_down and cisd_down:
            sweep_peak = max(highs[i-2], highs[i-1])
            sl = round(sweep_peak + sl_buffer, 2)
            pending_fvg = {
                'side': "SELL",
                'fvg_high': round(lows[i-2], 2),
                'fvg_low': round(highs[i], 2),
                'sl': sl,
                'bars_waited': 0
            }
            continue

        # Bullish Setup (BUY)
        swept_l = (lows[i-2] < recent_low and closes[i-2] > recent_low) or \
                  (lows[i-1] < recent_low and closes[i-1] > recent_low)
        has_bullish_fvg = highs[i-2] < (lows[i] - min_fvg_gap)
        disp_range_up = highs[i-1] - lows[i-1]
        disp_body_up = abs(closes[i-1] - opens[i-1])
        disp_up = closes[i-1] > opens[i-1] and disp_range_up > disp_threshold
        body_ok_up = (disp_body_up / max(0.0001, disp_range_up)) >= body_ratio_req

        # Explicit CISD: Displacement closes above the Open of the low-forming candle
        sweep_open_l = opens[i-2] if lows[i-2] <= lows[i-1] else opens[i-1]
        cisd_up = (not use_cisd) or (closes[i-1] > sweep_open_l) or (closes[i] > sweep_open_l)

        if swept_l and has_bullish_fvg and disp_up and body_ok_up and cisd_up:
            sweep_trough = min(lows[i-2], lows[i-1])
            sl = round(sweep_trough - sl_buffer, 2)
            pending_fvg = {
                'side': "BUY",
                'fvg_high': round(lows[i], 2),
                'fvg_low': round(highs[i-2], 2),
                'sl': sl,
                'bars_waited': 0
            }
            continue

    return pd.DataFrame(trades)

def summarize(df_trades: pd.DataFrame, label: str) -> dict:
    if df_trades is None or len(df_trades) == 0:
        return {"Configuration": label, "Trades": 0, "Wins": 0, "Losses": 0, "BE": 0, "Win Rate": "0.0%", "PnL ($)": 0.0, "ROC (%)": "0.0%"}
    wins = len(df_trades[df_trades['result'] == 'WIN'])
    losses = len(df_trades[df_trades['result'] == 'LOSS'])
    bes = len(df_trades[df_trades['result'] == 'BREAKEVEN'])
    eff_wr = (wins / max(1, (wins + losses))) * 100.0
    pnl = df_trades['pnl_usd'].sum()
    roc = (pnl / 1000.0) * 100.0
    return {
        "Configuration": label,
        "Trades": len(df_trades),
        "Wins": wins,
        "Losses": losses,
        "BE (Risk-Free)": bes,
        "Win Rate": f"{eff_wr:.1f}%",
        "PnL ($)": round(pnl, 2),
        "ROC (%)": f"{roc:+.1f}%"
    }

def main():
    print("=" * 100)
    print("🚀 BITCOIN (BTC/USD) ICT / SMC STRATEGY OPTIMIZATION MATRIX")
    print("=" * 100)

    df = fetch_btc_data()
    print(f"Loaded {len(df)} 15M candles for Bitcoin.\n")

    grid = [
        {"name": "1. Standard Parameters (Disp $150, FVG $20, SL $50, 1:2.0 RR)", "disp": 150.0, "fvg": 20.0, "sl": 50.0, "rr": 2.0, "body": 0.50},
        {"name": "2. Medium Volatility (Disp $250, FVG $35, SL $75, 1:2.5 RR)", "disp": 250.0, "fvg": 35.0, "sl": 75.0, "rr": 2.5, "body": 0.50},
        {"name": "3. High Volatility (Disp $350, FVG $50, SL $100, 1:2.5 RR)", "disp": 350.0, "fvg": 50.0, "sl": 100.0, "rr": 2.5, "body": 0.50},
        {"name": "4. Institutional Impulse (Disp $450, FVG $60, SL $120, 1:2.5 RR)", "disp": 450.0, "fvg": 60.0, "sl": 120.0, "rr": 2.5, "body": 0.50},
        {"name": "5. Institutional Impulse + 1:3.0 RR Runner", "disp": 450.0, "fvg": 60.0, "sl": 120.0, "rr": 3.0, "body": 0.50},
        {"name": "6. Institutional Impulse + 1:3.5 RR Runner", "disp": 450.0, "fvg": 60.0, "sl": 120.0, "rr": 3.5, "body": 0.50},
        {"name": "7. Ultra-Strict Momentum (Disp $500, FVG $80, SL $150, 1:3.0 RR)", "disp": 500.0, "fvg": 80.0, "sl": 150.0, "rr": 3.0, "body": 0.60},
    ]

    results = []
    for g in grid:
        trades = backtest_btc_ict(
            df,
            disp_threshold=g["disp"],
            min_fvg_gap=g["fvg"],
            sl_buffer=g["sl"],
            rr_ratio=g["rr"],
            body_ratio_req=g["body"],
            use_cisd=True
        )
        res = summarize(trades, g["name"])
        results.append(res)

    res_df = pd.DataFrame(results)
    print(res_df.to_string(index=False))

    # Show sample trades from best configuration
    best_cfg = grid[4]
    best_trades = backtest_btc_ict(
        df,
        disp_threshold=best_cfg["disp"],
        min_fvg_gap=best_cfg["fvg"],
        sl_buffer=best_cfg["sl"],
        rr_ratio=best_cfg["rr"],
        body_ratio_req=best_cfg["body"],
        use_cisd=True
    )
    if len(best_trades) > 0:
        print("\n📋 Sample Recent Bitcoin Trades (Institutional 1:3.0 RR):")
        cols = ['entry_time', 'side', 'entry_price', 'sl', 'tp', 'result', 'pnl_usd']
        print(best_trades[cols].tail(10).to_string(index=False))

if __name__ == "__main__":
    main()
