"""
tests/deep_dive_aud_cad.py
Extract deep granular metrics for AUDUSD and USDCAD under Config E:
- Monthly breakdown (July, Aug, Sept, Oct)
- Long vs Short stats
- Max consecutive wins & losses
- Trade duration (avg bars held)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pandas as pd
import numpy as np
import yfinance as yf

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_data_robust(ticker: str, retries: int = 3) -> pd.DataFrame:
    import time
    for attempt in range(retries):
        try:
            df = yf.download(ticker, period="60d", interval="15m", progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df.dropna(inplace=True)
            if len(df) > 500:
                for col in ["Open", "High", "Low", "Close"]:
                    df[col] = df[col].astype(float)
                df['ATR'] = calculate_atr(df, period=14)
                df['EMA100'] = df['Close'].ewm(span=100, adjust=False).mean()
                df.dropna(inplace=True)
                return df
        except Exception:
            pass
        time.sleep(2)
    return pd.DataFrame()

def run_deep_dive(ticker: str, name: str):
    df = fetch_data_robust(ticker)
    if df.empty:
        print(f"Failed to fetch data for {name}")
        return

    highs, lows, opens, closes = df['High'].values, df['Low'].values, df['Open'].values, df['Close'].values
    atrs, ema100, times = df['ATR'].values, df['EMA100'].values, df.index
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

        # Manage active
        if active_trade is not None:
            side = active_trade["side"]
            entry = active_trade["entry"]
            sl = active_trade["sl"]
            tp = active_trade["tp"]
            risk = active_trade["risk"]

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
                    "month": times[i].strftime("%B %Y"),
                    "side": side,
                    "r": rr_ratio,
                    "outcome": "WIN",
                    "bars": i - active_trade["bar"]
                })
                active_trade = None
            elif hit_sl:
                r_val = 0.0 if active_trade["is_be"] else -1.0
                out_str = "BE" if active_trade["is_be"] else "LOSS"
                trades.append({
                    "entry_time": active_trade["time"],
                    "exit_time": times[i],
                    "month": times[i].strftime("%B %Y"),
                    "side": side,
                    "r": r_val,
                    "outcome": out_str,
                    "bars": i - active_trade["bar"]
                })
                active_trade = None

        # Check pending
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
                            "time": times[i],
                            "bar": i
                        }
                    continue
                rem.append(z)
            pending_zones = rem

        # Detect S&D Base + FVG
        c_base, c_disp, c_conf = i - 2, i - 1, i
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

    df_t = pd.DataFrame(trades)
    print(f"\n{'='*70}\n📊 DEEP DIVE: {name} ({ticker})\n{'='*70}")
    tot = len(df_t)
    wins = len(df_t[df_t['outcome'] == 'WIN'])
    losses = len(df_t[df_t['outcome'] == 'LOSS'])
    bes = len(df_t[df_t['outcome'] == 'BE'])
    net_r = df_t['r'].sum()
    gw = df_t[df_t['outcome'] == 'WIN']['r'].sum()
    gl = abs(df_t[df_t['outcome'] == 'LOSS']['r'].sum())
    pf = gw / gl if gl > 0 else 99.0
    
    # Consecutive
    cur_streak, max_win_streak, max_loss_streak = 0, 0, 0
    prev_out = None
    for _, row in df_t.iterrows():
        out = row['outcome']
        if out == prev_out:
            cur_streak += 1
        else:
            cur_streak = 1
            prev_out = out
        if out == 'WIN' and cur_streak > max_win_streak:
            max_win_streak = cur_streak
        if out == 'LOSS' and cur_streak > max_loss_streak:
            max_loss_streak = cur_streak

    avg_bars = df_t['bars'].mean()
    avg_hours = (avg_bars * 15) / 60

    print(f"• Total Trades       : {tot}")
    print(f"• Record             : {wins} Wins | {losses} Losses | {bes} Breakevens")
    print(f"• Win Rate           : {wins/tot*100:.1f}%")
    print(f"• Net Profit         : {net_r:+.1f}R")
    print(f"• Profit Factor      : {pf:.2f}")
    print(f"• Max Win Streak     : {max_win_streak} trades")
    print(f"• Max Loss Streak    : {max_loss_streak} trades")
    print(f"• Avg Trade Duration : {avg_bars:.1f} bars (~{avg_hours:.1f} hours)")

    # Long vs Short
    longs = df_t[df_t['side'] == 'BUY']
    shorts = df_t[df_t['side'] == 'SELL']
    print("\n--- Long vs Short Breakdown ---")
    print(f"• BUY Trades  : {len(longs)} | Wins: {len(longs[longs['outcome']=='WIN'])} | Net: {longs['r'].sum():+.1f}R")
    print(f"• SELL Trades : {len(shorts)} | Wins: {len(shorts[shorts['outcome']=='WIN'])} | Net: {shorts['r'].sum():+.1f}R")

    # Monthly
    print("\n--- Monthly PnL Breakdown ---")
    for m, grp in df_t.groupby("month", sort=False):
        m_wins = len(grp[grp['outcome']=='WIN'])
        m_tot = len(grp)
        m_r = grp['r'].sum()
        print(f"• {m:<15}: {m_tot:2d} trades | Win Rate: {m_wins/m_tot*100:4.1f}% | Net: {m_r:+5.1f}R")

if __name__ == "__main__":
    run_deep_dive("AUDUSD=X", "AUD / USD")
    run_deep_dive("USDCAD=X", "USD / CAD")
