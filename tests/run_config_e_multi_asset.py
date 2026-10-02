"""
tests/run_config_e_multi_asset.py
Comprehensive Multi-Asset Backtest of Strategy Config E:
- Entry Style: Proximal Edge of Imbalance (FVG)
- Risk-Reward (R:R): 1:2.0
- Breakeven Trigger: +1.0R (Risk-Free)
- Invalidation: Distal edge of S&D Base candle + 0.2 ATR buffer
- Macro Trend: Filtered by EMA100
- Range: 60 Days of 15-minute institutional candles
"""

import sys
import time
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)

import pandas as pd
import numpy as np
import yfinance as yf
from typing import Dict, Any, List

def calculate_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df['High'], df['Low'], df['Close']
    tr = pd.concat([high - low, (high - close.shift(1)).abs(), (low - close.shift(1)).abs()], axis=1).max(axis=1)
    return tr.rolling(period).mean()

def fetch_data_robust(ticker: str, retries: int = 3) -> pd.DataFrame:
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
        except Exception as e:
            pass
        time.sleep(2)
    return pd.DataFrame()

def run_config_e(df: pd.DataFrame, asset_name: str) -> Dict[str, Any]:
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

        # 2. Check Pending Zones for Mitigation
        if active_trade is None and pending_zones:
            rem = []
            for z in pending_zones:
                if i - z["bar"] > max_zone_bars:
                    continue
                if (z["side"] == "BUY" and l < z["distal"]) or (z["side"] == "SELL" and h > z["distal"]):
                    continue

                # Proximal Touch
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

        # 3. Detect S&D Base + Imbalance
        c_base = i - 2
        c_disp = i - 1
        c_conf = i

        disp_body = abs(closes[c_disp] - opens[c_disp])
        if disp_body >= (disp_mult * atrs[c_disp]):
            # Bullish: Drop-Base-Rally + BISI
            if closes[c_disp] > opens[c_disp]:
                gap = lows[c_conf] - highs[c_base]
                if gap > (0.1 * atrs[c_conf]) and closes[i] > ema100[i]:
                    base_low = min(lows[c_base], lows[c_disp])
                    pending_zones.append({
                        "side": "BUY",
                        "bar": i,
                        "entry": lows[c_conf], # Proximal Edge
                        "sl": base_low - (sl_buf_mult * cur_atr),
                        "distal": base_low
                    })
            # Bearish: Rally-Base-Drop + SIBI
            elif closes[c_disp] < opens[c_disp]:
                gap = lows[c_base] - highs[c_conf]
                if gap > (0.1 * atrs[c_conf]) and closes[i] < ema100[i]:
                    base_high = max(highs[c_base], highs[c_disp])
                    pending_zones.append({
                        "side": "SELL",
                        "bar": i,
                        "entry": highs[c_conf], # Proximal Edge
                        "sl": base_high + (sl_buf_mult * cur_atr),
                        "distal": base_high
                    })

    # Stats calculation
    tot = len(trades)
    if tot == 0:
        return {"asset": asset_name, "trades": 0}

    wins = [t for t in trades if t["outcome"] == "WIN"]
    losses = [t for t in trades if t["outcome"] == "LOSS"]
    bes = [t for t in trades if t["outcome"] == "BE"]

    win_rate = (len(wins) / tot) * 100.0
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

    return {
        "asset": asset_name,
        "candles": n,
        "trades": tot,
        "wins": len(wins),
        "losses": len(losses),
        "bes": len(bes),
        "win_rate": round(win_rate, 1),
        "gross_win_r": round(gw, 1),
        "gross_loss_r": round(gl, 1),
        "net_r": round(net_r, 1),
        "pf": round(pf, 2),
        "max_dd": round(max_dd, 1),
        "expectancy": round(net_r / tot, 2)
    }

def main():
    assets = [
        ("Gold (XAUUSD)", "GC=F", "Commodity"),
        ("Euro / USD (EURUSD)", "EURUSD=X", "Forex Major"),
        ("USD / JPY (USDJPY)", "USDJPY=X", "Forex Major"),
        ("AUD / USD (AUDUSD)", "AUDUSD=X", "Forex Major"),
        ("USD / CAD (USDCAD)", "USDCAD=X", "Forex Major"),
        ("Nasdaq 100 Futures (NQ)", "NQ=F", "Equity Index"),
        ("S&P 500 Futures (ES)", "ES=F", "Equity Index"),
        ("Bitcoin (BTCUSD)", "BTC-USD", "Crypto"),
    ]

    print("=" * 105)
    print("🚀 CONFIG E MULTI-ASSET INSTITUTIONAL BACKTEST (60 DAYS, 15M INTRADAY)")
    print("Strategy: S&D + Imbalance (FVG) | Proximal Entry | 1:2.0 RR | Breakeven @ +1.0R | EMA100 Trend")
    print("=" * 105)

    results = []
    for name, ticker, category in assets:
        df = fetch_data_robust(ticker)
        if len(df) > 500:
            res = run_config_e(df, name)
            res["category"] = category
            results.append(res)
            status_icon = "🏆" if res["net_r"] > 0 else ("🛡️" if res["net_r"] == 0 else "❌")
            print(f"{status_icon} {name:<26} | Category: {category:<12} | Trades: {res['trades']:3d} | W/L/BE: {res['wins']:2d}/{res['losses']:2d}/{res['bes']:2d} | Win Rate: {res['win_rate']:4.1f}% | Net R: {res['net_r']:+5.1f}R | PF: {res['pf']:4.2f} | Max DD: {res['max_dd']:4.1f}R")
        else:
            print(f"⚠️ {name:<26} | Skipping (Insufficient data)")
        time.sleep(1)

    # Summary
    if results:
        df_all = pd.DataFrame(results)
        tot_trades = df_all['trades'].sum()
        tot_wins = df_all['wins'].sum()
        tot_losses = df_all['losses'].sum()
        tot_bes = df_all['bes'].sum()
        tot_net_r = df_all['net_r'].sum()
        avg_pf = df_all['pf'].mean()
        avg_wr = (tot_wins / tot_trades) * 100.0

        print("\n" + "=" * 105)
        print("🏆 PORTFOLIO AGGREGATE SUMMARY (ALL ASSETS COMBINED)")
        print("=" * 105)
        print(f"• Total Assets Analyzed : {len(results)}")
        print(f"• Total Trades Executed : {tot_trades}")
        print(f"• Total Record (W/L/BE) : {tot_wins} Wins / {tot_losses} Losses / {tot_bes} Breakevens")
        print(f"• Portfolio Win Rate    : {avg_wr:.1f}%")
        print(f"• Portfolio Combined PnL: {tot_net_r:+.1f}R Net Profit")
        print(f"• Average Profit Factor : {avg_pf:.2f}")

if __name__ == "__main__":
    main()
