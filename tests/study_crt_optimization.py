"""
tests/study_crt_optimization.py - Quantitative Study to Turn CRT from Losing to Winning
Evaluates:
1. Baseline: 24/7 unconstrained CRT
2. Filter 1: Killzone restriction (London 07:00-10:00 UTC & NY 12:00-15:00 UTC only)
3. Filter 2: Killzone + 1H 50 EMA Trend Filter
4. Filter 3: Killzone + 1H 50 EMA + 0.5 ATR Minimum Liquidity Sweep Buffer
"""

import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from typing import Dict, Any, List

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_crt_backtest(df: pd.DataFrame, use_kz: bool, use_trend: bool, sweep_atr_mult: float, target_rr: float = 2.0) -> Dict[str, Any]:
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    atrs = df['ATR'].values
    h1_50 = df['H1_EMA50'].values
    times = df.index
    n = len(df)

    trades = []
    active = None

    for i in range(16, n):
        h, l, c, o = highs[i], lows[i], closes[i], opens[i]
        cur_atr = atrs[i]
        t = times[i]
        hr = t.hour

        # Check active trade
        if active is not None:
            side = active["side"]
            entry = active["entry"]
            sl = active["sl"]
            risk = active["risk"]

            # Breakeven ratchet at 1.0R
            if not active["is_be"]:
                if (side == "BUY" and h >= entry + risk) or (side == "SELL" and l <= entry - risk):
                    active["sl"] = entry
                    active["is_be"] = True

            # Exit check
            closed = False
            pnl_r = 0.0
            if side == "BUY":
                if l <= active["sl"]:
                    pnl_r = 0.0 if active["is_be"] else -1.0
                    closed = True
                elif h >= entry + (risk * target_rr):
                    pnl_r = target_rr
                    closed = True
            elif side == "SELL":
                if h >= active["sl"]:
                    pnl_r = 0.0 if active["is_be"] else -1.0
                    closed = True
                elif l <= entry - (risk * target_rr):
                    pnl_r = target_rr
                    closed = True

            if closed:
                trades.append({"pnl_r": pnl_r, "pnl_usd": pnl_r * 100.0, "is_be": active["is_be"]})
                active = None

        if active is None:
            # 1. Killzone check
            if use_kz:
                # London KZ: 07:00-10:00 UTC | NY KZ: 12:00-15:00 UTC
                is_kz = (7 <= hr <= 10) or (12 <= hr <= 15)
                if not is_kz:
                    continue

            # 4-bar anchor range
            anchor_h = np.max(highs[i-8:i-4])
            anchor_l = np.min(lows[i-8:i-4])
            anchor_mid = (anchor_h + anchor_l) / 2.0
            min_sweep = sweep_atr_mult * cur_atr

            # Bullish CRT Reclaim
            prev_l = lows[i-2]
            prev_c = closes[i-1]
            if (prev_l < anchor_l - min_sweep) and (prev_c > anchor_l) and (prev_c < anchor_mid):
                if not use_trend or (c >= h1_50[i]):
                    risk = max(cur_atr * 0.8, c - prev_l)
                    active = {"side": "BUY", "entry": c, "sl": c - risk, "risk": risk, "is_be": False}

            # Bearish CRT Reclaim
            prev_h = highs[i-2]
            if (prev_h > anchor_h + min_sweep) and (prev_c < anchor_h) and (prev_c > anchor_mid):
                if not use_trend or (c <= h1_50[i]):
                    risk = max(cur_atr * 0.8, prev_h - c)
                    active = {"side": "SELL", "entry": c, "sl": c + risk, "risk": risk, "is_be": False}

    if not trades:
        return {"trades": 0, "win_rate": 0, "pf": 0, "net_pnl": 0}

    total = len(trades)
    wins = len([t for t in trades if t["pnl_r"] > 0])
    losses = len([t for t in trades if t["pnl_r"] < 0])
    pfs = (sum(t["pnl_usd"] for t in trades if t["pnl_usd"] > 0) /
           abs(sum(t["pnl_usd"] for t in trades if t["pnl_usd"] < 0) or 1))

    return {
        "trades": total,
        "wins": wins,
        "losses": losses,
        "bes": len([t for t in trades if t["pnl_r"] == 0]),
        "win_rate": round(wins / total * 100, 1),
        "pf": round(pfs, 2),
        "net_pnl": round(sum(t["pnl_usd"] for t in trades), 2)
    }

def main():
    assets = [
        {"name": "EUR/USD", "ticker": "EURUSD=X"},
        {"name": "Bitcoin (BTCUSD)", "ticker": "BTC-USD"}
    ]

    print("=" * 80)
    print("🔬 QUANTITATIVE STUDY: TURNING CRT STRATEGY FROM LOSING TO PROFITABLE")
    print("=" * 80)

    for a in assets:
        ticker = a["ticker"]
        name = a["name"]
        print(f"\n📥 Downloading 60d of 15m and 1h data for {name} ({ticker})...")
        df_15m = yf.download(ticker, period="60d", interval="15m", progress=False)
        if isinstance(df_15m.columns, pd.MultiIndex):
            df_15m.columns = df_15m.columns.get_level_values(0)
        df_15m.dropna(inplace=True)

        df_1h = yf.download(ticker, period="60d", interval="1h", progress=False)
        if isinstance(df_1h.columns, pd.MultiIndex):
            df_1h.columns = df_1h.columns.get_level_values(0)
        df_1h.dropna(inplace=True)

        for c in ["Open", "High", "Low", "Close"]:
            df_15m[c] = df_15m[c].astype(float)
            df_1h[c] = df_1h[c].astype(float)

        df_15m['ATR'] = calculate_atr(df_15m, 14)
        df_1h['EMA50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()

        # Map 1h EMA50
        h1_dict = {idx: float(row['EMA50']) for idx, row in df_1h.iterrows()}
        h1_keys = sorted(list(h1_dict.keys()))

        def get_h1(ts):
            p = [k for k in h1_keys if k <= ts]
            return h1_dict[p[-1]] if p else np.nan

        df_15m['H1_EMA50'] = [get_h1(ts) for ts in df_15m.index]
        df_15m.dropna(inplace=True)

        # Configurations:
        # Config 1: Baseline 24/7, no trend filter, 0 sweep buffer
        r1 = run_crt_backtest(df_15m, use_kz=False, use_trend=False, sweep_atr_mult=0.0)
        # Config 2: Session Killzone only
        r2 = run_crt_backtest(df_15m, use_kz=True, use_trend=False, sweep_atr_mult=0.0)
        # Config 3: Session Killzone + H1 50 EMA Trend Filter
        r3 = run_crt_backtest(df_15m, use_kz=True, use_trend=True, sweep_atr_mult=0.0)
        # Config 4: Session Killzone + H1 50 EMA + 0.3x ATR Sweep Buffer
        r4 = run_crt_backtest(df_15m, use_kz=True, use_trend=True, sweep_atr_mult=0.3)

        print(f"📊 Results for {name}:")
        print(f"  • Baseline (24/7 unconstrained):          Trades={r1['trades']:3} | WR={r1['win_rate']:4.1f}% | PF={r1['pf']:4.2f} | PnL=${r1['net_pnl']:+8.2f}")
        print(f"  • Optimization 1 (+Killzones 07-15 UTC):  Trades={r2['trades']:3} | WR={r2['win_rate']:4.1f}% | PF={r2['pf']:4.2f} | PnL=${r2['net_pnl']:+8.2f}")
        print(f"  • Optimization 2 (+H1 50 EMA Trend):      Trades={r3['trades']:3} | WR={r3['win_rate']:4.1f}% | PF={r3['pf']:4.2f} | PnL=${r3['net_pnl']:+8.2f}")
        print(f"  • Optimization 3 (+0.3x ATR Sweep Buffer):Trades={r4['trades']:3} | WR={r4['win_rate']:4.1f}% | PF={r4['pf']:4.2f} | PnL=${r4['net_pnl']:+8.2f}")

if __name__ == "__main__":
    main()
