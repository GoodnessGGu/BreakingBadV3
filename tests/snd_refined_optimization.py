"""
tests/snd_refined_optimization.py
Refined S&D + Imbalance Backtest exploring:
- Session Killzones (London 07-10 UTC & NY 12-16 UTC)
- Optimized Proximal Retest Buffer
- Trailing Stop vs Fixed R:R
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_asset(ticker: str, name: str, rr: float = 2.0, killzone_only: bool = True):
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df['ATR'] = calculate_atr(df)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df.dropna(inplace=True)

    highs, lows, opens, closes = df['High'].values, df['Low'].values, df['Open'].values, df['Close'].values
    atrs, ema100, times = df['ATR'].values, df['EMA100'].values, df.index
    n = len(df)

    active = None
    zones = []
    trades = []

    for i in range(25, n):
        h, l, o, c = highs[i], lows[i], opens[i], closes[i]
        cur_time = times[i]
        hour = cur_time.hour

        # Check Killzone (London: 07-11 UTC, NY: 12-17 UTC)
        in_kz = (7 <= hour <= 11) or (12 <= hour <= 17) if killzone_only else True

        # 1. Manage Active Trade
        if active is not None:
            side = active["side"]
            entry = active["entry"]
            sl = active["sl"]
            tp = active["tp"]
            risk = active["risk"]

            # Breakeven at +1.0R
            if not active["is_be"]:
                if side == "BUY" and h >= entry + risk:
                    active["sl"] = entry
                    active["is_be"] = True
                elif side == "SELL" and l <= entry - risk:
                    active["sl"] = entry
                    active["is_be"] = True

            hit_tp = (h >= tp) if side == "BUY" else (l <= tp)
            hit_sl = (l <= sl) if side == "BUY" else (h >= sl)

            if hit_tp and hit_sl:
                hit_tp = False

            if hit_tp:
                trades.append({"r": rr, "win": True})
                active = None
            elif hit_sl:
                r = 0.0 if active["is_be"] else -1.0
                trades.append({"r": r, "win": False})
                active = None

        # 2. Check Pending Zones for Entry
        if active is None and zones:
            rem = []
            for z in zones:
                if i - z["bar"] > 60:
                    continue
                if (z["side"] == "BUY" and l < z["distal"]) or (z["side"] == "SELL" and h > z["distal"]):
                    continue
                
                # Mitigation trigger
                if l <= z["entry"] <= h:
                    risk = abs(z["entry"] - z["sl"])
                    if risk > 0:
                        tp = z["entry"] + (risk * rr) if z["side"] == "BUY" else z["entry"] - (risk * rr)
                        active = {
                            "side": z["side"],
                            "entry": z["entry"],
                            "sl": z["sl"],
                            "tp": tp,
                            "risk": risk,
                            "is_be": False
                        }
                    continue
                rem.append(z)
            zones = rem

        # 3. Detect S&D Base + Imbalance
        c_base, c_disp, c_conf = i - 2, i - 1, i
        disp_body = abs(closes[c_disp] - opens[c_disp])
        if disp_body >= (1.2 * atrs[c_disp]) and (in_kz or not killzone_only):
            # Bullish
            if closes[c_disp] > opens[c_disp]:
                gap = lows[c_conf] - highs[c_base]
                if gap > (0.1 * atrs[c_conf]) and closes[i] > ema100[i]:
                    zones.append({
                        "side": "BUY",
                        "bar": i,
                        "entry": lows[c_conf], # Proximal
                        "sl": min(lows[c_base], lows[c_disp]) - (0.2 * atrs[i]),
                        "distal": min(lows[c_base], lows[c_disp])
                    })
            # Bearish
            elif closes[c_disp] < opens[c_disp]:
                gap = lows[c_base] - highs[c_conf]
                if gap > (0.1 * atrs[c_conf]) and closes[i] < ema100[i]:
                    zones.append({
                        "side": "SELL",
                        "bar": i,
                        "entry": highs[c_conf], # Proximal
                        "sl": max(highs[c_base], highs[c_disp]) + (0.2 * atrs[i]),
                        "distal": max(highs[c_base], highs[c_disp])
                    })

    # Stats
    tot = len(trades)
    if tot == 0:
        return
    wins = [t for t in trades if t["win"]]
    losses = [t for t in trades if not t["win"] and t["r"] < 0]
    bes = [t for t in trades if t["r"] == 0]
    net_r = sum(t["r"] for t in trades)
    wr = len(wins) / tot * 100
    gw = sum(t["r"] for t in wins)
    gl = abs(sum(t["r"] for t in losses))
    pf = (gw / gl) if gl > 0 else 99.0
    
    # DD
    eq, peak, dd = 0, 0, 0
    for t in trades:
        eq += t["r"]
        if eq > peak: peak = eq
        if peak - eq > dd: dd = peak - eq

    print(f"[{name}] Trades: {tot:3d} | Win%: {wr:4.1f}% | Net R: {net_r:+6.2f}R | PF: {pf:4.2f} | Max DD: {dd:4.2f}R | BEs: {len(bes)}")

if __name__ == "__main__":
    print("=== KILLZONE FILTERED BACKTEST (LONDON + NY KILLZONES) ===")
    run_asset("GC=F", "Gold (XAUUSD) - Killzone", rr=2.0, killzone_only=True)
    run_asset("EURUSD=X", "EURUSD - Killzone", rr=2.0, killzone_only=True)
    run_asset("BTC-USD", "Bitcoin - 24/7 No Killzone", rr=2.0, killzone_only=False)
    run_asset("BTC-USD", "Bitcoin - US Killzone Only", rr=2.0, killzone_only=True)
