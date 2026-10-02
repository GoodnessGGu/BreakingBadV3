"""
tests/test_nzdusd.py
Deep-dive backtest of NZD/USD under Strategy Config E:
- S&D Base + Imbalance (FVG) Proximal Entry
- 1:2.0 R:R Target
- +1.0R Breakeven Ratchet
- EMA100 Trend Invalidation
- 60 Trading Days, 15M candles
"""
import sys
import time
import pandas as pd
import numpy as np
import yfinance as yf

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def run_backtest(ticker="NZDUSD=X", name="NZD / USD (NZDUSD)"):
    print(f"Fetching data for {ticker}...")
    df = yf.download(ticker, period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df['ATR'] = calculate_atr(df, period=14)
    df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
    df.dropna(inplace=True)

    highs = df['High'].values
    lows = df['Low'].values
    opens = df['Open'].values
    closes = df['Close'].values
    atrs = df['ATR'].values
    ema100 = df['EMA100'].values
    times = df.index
    n = len(df)

    active_trade = None
    pending_zones = []
    trades = []

    rr_ratio = 2.0
    disp_mult = 1.2
    sl_buf_mult = 0.2
    max_zone_bars = 80

    for i in range(25, n):
        h, l, o, c = highs[i], lows[i], opens[i], closes[i]
        cur_atr = atrs[i]

        # 1. Manage Active Trade
        if active_trade is not None:
            side = active_trade["side"]
            entry = active_trade["entry"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]
            risk = active_trade["risk"]

            # Breakeven at +1.0R
            if not active_trade["is_be"]:
                if side == "BUY" and h >= entry + risk:
                    active_trade["sl"] = entry
                    active_trade["is_be"] = True
                elif side == "SELL" and l <= entry - risk:
                    active_trade["sl"] = entry
                    active_trade["is_be"] = True

            hit_tp = (h >= tp) if side == "BUY" else (l <= tp)
            hit_sl = (l <= sl) if side == "BUY" else (h >= sl)

            if hit_tp and hit_sl:
                hit_tp = False

            if hit_tp:
                trades.append({
                    "entry_time": active_trade["time"],
                    "exit_time": times[i],
                    "side": side,
                    "r": rr_ratio,
                    "outcome": "WIN"
                })
                active_trade = None
            elif hit_sl:
                r_val = 0.0 if active_trade["is_be"] else -1.0
                out_str = "BE" if active_trade["is_be"] else "LOSS"
                trades.append({
                    "entry_time": active_trade["time"],
                    "exit_time": times[i],
                    "side": side,
                    "r": r_val,
                    "outcome": out_str
                })
                active_trade = None

        # 2. Check Pending Zones
        if active_trade is None and pending_zones:
            rem = []
            for z in pending_zones:
                if i - z["bar"] > max_zone_bars:
                    continue
                if (z["side"] == "BUY" and l < z["distal"]) or (z["side"] == "SELL" and h > z["distal"]):
                    continue

                if l <= z["entry"] <= h:
                    risk = abs(z["entry"] - z["sl"])
                    if risk > 0:
                        tp_level = z["entry"] + (risk * rr_ratio) if z["side"] == "BUY" else z["entry"] - (risk * rr_ratio)
                        active_trade = {
                            "side": z["side"],
                            "entry": z["entry"],
                            "sl": z["sl"],
                            "tp": tp_level,
                            "risk": risk,
                            "is_be": False,
                            "time": times[i]
                        }
                    continue
                rem.append(z)
            pending_zones = rem

        # 3. Detect Base + Imbalance
        c_base = i - 2
        c_disp = i - 1
        c_conf = i

        disp_body = abs(closes[c_disp] - opens[c_disp])
        if disp_body >= (disp_mult * atrs[c_disp]):
            if closes[c_disp] > opens[c_disp]:
                gap = lows[c_conf] - highs[c_base]
                if gap > (0.1 * atrs[c_conf]) and closes[i] > ema100[i]:
                    base_low = min(lows[c_base], lows[c_disp])
                    pending_zones.append({
                        "side": "BUY",
                        "bar": i,
                        "entry": lows[c_conf],
                        "sl": base_low - (sl_buf_mult * cur_atr),
                        "distal": base_low
                    })
            elif closes[c_disp] < opens[c_disp]:
                gap = lows[c_base] - highs[c_conf]
                if gap > (0.1 * atrs[c_conf]) and closes[i] < ema100[i]:
                    base_high = max(highs[c_base], highs[c_disp])
                    pending_zones.append({
                        "side": "SELL",
                        "bar": i,
                        "entry": highs[c_conf],
                        "sl": base_high + (sl_buf_mult * cur_atr),
                        "distal": base_high
                    })

    tot = len(trades)
    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    bes = [t for t in trades if t["outcome"] == "BE"]

    win_rate = (len(wins) / tot) * 100.0 if tot > 0 else 0
    gw = sum(t["r"] for t in wins)
    gl = abs(sum(t["r"] for t in losses))
    net_r = gw - gl
    pf = (gw / gl) if gl > 0 else 99.0

    eq, peak, max_dd = 0.0, 0.0, 0.0
    for t in trades:
        eq += t["r"]
        if eq > peak:
            peak = eq
        dd = peak - eq
        if dd > max_dd:
            max_dd = dd

    print("\n" + "="*50)
    print(f"RESULTS FOR {name}")
    print("="*50)
    print(f"Candles Analyzed : {n}")
    print(f"Date Range       : {times[0]} to {times[-1]}")
    print(f"Total Trades     : {tot}")
    print(f"Wins             : {len(wins)}")
    print(f"Losses           : {len(losses)}")
    print(f"Breakevens (BE)  : {len(bes)}")
    print(f"Win Rate         : {win_rate:.2f}%")
    print(f"Breakeven Rate   : {(len(bes)/tot)*100:.2f}%")
    print(f"Gross Win (R)    : +{gw:.1f}R")
    print(f"Gross Loss (R)   : -{gl:.1f}R")
    print(f"Net Profit (R)   : {net_r:+.1f}R")
    print(f"Profit Factor    : {pf:.2f}")
    print(f"Max Drawdown (R) : {max_dd:.1f}R")
    print(f"Expectancy       : {net_r/tot:+.2f}R per trade")

    # Long vs Short
    longs = [t for t in trades if t["side"] == "BUY"]
    shorts = [t for t in trades if t["side"] == "SELL"]
    print(f"Long Trades      : {len(longs)} (Net: {sum(t['r'] for t in longs):+.1f}R)")
    print(f"Short Trades     : {len(shorts)} (Net: {sum(t['r'] for t in shorts):+.1f}R)")

if __name__ == "__main__":
    run_backtest()
