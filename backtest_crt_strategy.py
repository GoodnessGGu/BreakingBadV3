"""
backtest_crt_strategy.py - Comprehensive Institutional CRT Backtester

Compares 2 Core CRT Models:
1. Model A: Classic Multi-Timeframe CRT (H1 Anchor Range + 5m Reclaim + Equilibrium Breakeven).
2. Model B: London Judas Asian Session CRT (00:00-06:00 UTC Asian Range Sweeps + Killzone + FVG).

Tested across Gold (XAUUSD), Silver (XAGUSD), EUR/USD, and Bitcoin (BTCUSD).
"""

import sys
import os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("CRT_Backtester")


def fetch_aligned_market_data(ticker="GC=F", period="30d"):
    """Fetches aligned 5m and 1h market data."""
    df_5m = yf.download(ticker, period=period, interval="5m", progress=False)
    if isinstance(df_5m.columns, pd.MultiIndex):
        df_5m.columns = df_5m.columns.get_level_values(0)
    df_5m.dropna(inplace=True)

    df_1h = yf.download(ticker, period=period, interval="1h", progress=False)
    if isinstance(df_1h.columns, pd.MultiIndex):
        df_1h.columns = df_1h.columns.get_level_values(0)
    df_1h.dropna(inplace=True)

    return df_5m, df_1h


def run_h1_candle_crt_backtest(
    df_5m: pd.DataFrame,
    df_1h: pd.DataFrame,
    ticker: str,
    asset_name: str,
    min_sweep: float,
    max_sweep: float,
    sl_buffer: float,
    target_rr: float = 2.0,
    be_r: float = 1.0,
    risk_usd: float = 100.0
):
    """Model A: H1 Candle Range CRT with Killzone & Trend Filter."""
    df_1h = df_1h.copy()
    df_1h['EMA_50'] = df_1h['Close'].ewm(span=50, adjust=False).mean()

    h1_ranges = {}
    for i in range(1, len(df_1h)):
        prev_h = float(df_1h['High'].iloc[i-1])
        prev_l = float(df_1h['Low'].iloc[i-1])
        ema_val = float(df_1h['EMA_50'].iloc[i-1])
        cur_time = df_1h.index[i]
        h1_ranges[cur_time] = {
            'high': prev_h,
            'low': prev_l,
            'mid': (prev_h + prev_l) / 2.0,
            'ema': ema_val
        }

    h1_keys = sorted(list(h1_ranges.keys()))

    def get_anchor(t):
        app = [k for k in h1_keys if k <= t]
        return h1_ranges[app[-1]] if app else None

    trades = []
    active = None

    for i in range(15, len(df_5m)):
        cur_t = df_5m.index[i]
        hr = cur_t.hour
        minute = cur_t.minute
        o = float(df_5m['Open'].iloc[i])
        h = float(df_5m['High'].iloc[i])
        l = float(df_5m['Low'].iloc[i])
        c = float(df_5m['Close'].iloc[i])

        if active:
            side = active['side']
            entry = active['entry_price']
            sl = active['sl_price']
            tp = active['tp_price']
            risk = active['risk']

            if side == 'BUY':
                if not active['is_be'] and h >= entry + (risk * be_r):
                    active['sl_price'] = entry
                    active['is_be'] = True
                if l <= active['sl_price']:
                    trades.append({'outcome': 'BREAKEVEN' if active['is_be'] else 'LOSS', 'pnl_r': 0.0 if active['is_be'] else -1.0, 'pnl_usd': 0.0 if active['is_be'] else -risk_usd})
                    active = None
                    continue
                elif h >= tp:
                    rr = (tp - entry) / risk
                    trades.append({'outcome': 'WIN', 'pnl_r': rr, 'pnl_usd': risk_usd * rr})
                    active = None
                    continue
            elif side == 'SELL':
                if not active['is_be'] and l <= entry - (risk * be_r):
                    active['sl_price'] = entry
                    active['is_be'] = True
                if h >= active['sl_price']:
                    trades.append({'outcome': 'BREAKEVEN' if active['is_be'] else 'LOSS', 'pnl_r': 0.0 if active['is_be'] else -1.0, 'pnl_usd': 0.0 if active['is_be'] else -risk_usd})
                    active = None
                    continue
                elif l <= tp:
                    rr = (entry - tp) / risk
                    trades.append({'outcome': 'WIN', 'pnl_r': rr, 'pnl_usd': risk_usd * rr})
                    active = None
                    continue

        if active is None:
            # Active Killzones (07:00-10:00 UTC, 12:30-16:00 UTC)
            if not ((7 <= hr <= 10) or (hr == 12 and minute >= 30) or (13 <= hr <= 16)):
                continue

            anc = get_anchor(cur_t)
            if not anc:
                continue

            a_high, a_low, a_mid, ema = anc['high'], anc['low'], anc['mid'], anc['ema']
            rec_low = df_5m['Low'].iloc[i-4:i+1].min()
            rec_high = df_5m['High'].iloc[i-4:i+1].max()

            # Bullish CRT: Swept below previous H1 low, closed back above anchor low with green displacement
            sweep_d = a_low - rec_low
            if min_sweep <= sweep_d <= max_sweep and c > ema:
                body = c - o
                tot_r = h - l
                if c > a_low and body > 0 and tot_r > 0 and (body / tot_r >= 0.50):
                    entry = c
                    sl = rec_low - sl_buffer
                    risk = entry - sl
                    if risk > 0:
                        tp = max(a_high, entry + (risk * target_rr))
                        if (tp - entry) / risk >= target_rr:
                            active = {'side': 'BUY', 'entry_price': entry, 'sl_price': sl, 'tp_price': tp, 'risk': risk, 'is_be': False}

            # Bearish CRT: Swept above previous H1 high, closed back below anchor high with red displacement
            sweep_u = rec_high - a_high
            if min_sweep <= sweep_u <= max_sweep and c < ema:
                body = o - c
                tot_r = h - l
                if c < a_high and body > 0 and tot_r > 0 and (body / tot_r >= 0.50):
                    entry = c
                    sl = rec_high + sl_buffer
                    risk = sl - entry
                    if risk > 0:
                        tp = min(a_low, entry - (risk * target_rr))
                        if (entry - tp) / risk >= target_rr:
                            active = {'side': 'SELL', 'entry_price': entry, 'sl_price': sl, 'tp_price': tp, 'risk': risk, 'is_be': False}

    return _compile_stats(trades, asset_name, "H1 Anchor Range CRT", ticker)


def run_asian_judas_crt_backtest(
    df_5m: pd.DataFrame,
    ticker: str,
    asset_name: str,
    min_sweep: float,
    max_sweep: float,
    sl_buffer: float,
    target_rr: float = 2.5,
    be_r: float = 1.0,
    risk_usd: float = 100.0
):
    """Model B: Asian Session Range (00:00-06:00 UTC) London Judas CRT."""
    df = df_5m.copy()
    df['Date'] = df.index.date
    df['Hour'] = df.index.hour
    df['Minute'] = df.index.minute

    asian_ranges = {}
    for date_val, group in df.groupby('Date'):
        asian_sub = group[(group['Hour'] >= 0) & (group['Hour'] < 6)]
        if len(asian_sub) >= 10:
            a_high = float(asian_sub['High'].max())
            a_low = float(asian_sub['Low'].min())
            asian_ranges[date_val] = {
                "high": a_high,
                "low": a_low,
                "mid": (a_high + a_low) / 2.0,
                "size": a_high - a_low
            }

    trades = []
    active = None

    for i in range(15, len(df)):
        cur_t = df.index[i]
        cur_d = cur_t.date()
        hr = cur_t.hour
        minute = cur_t.minute
        o = float(df['Open'].iloc[i])
        h = float(df['High'].iloc[i])
        l = float(df['Low'].iloc[i])
        c = float(df['Close'].iloc[i])

        if active:
            side = active['side']
            entry = active['entry_price']
            sl = active['sl_price']
            tp = active['tp_price']
            risk = active['risk']

            if side == 'BUY':
                if not active['is_be'] and h >= entry + (risk * be_r):
                    active['sl_price'] = entry
                    active['is_be'] = True
                if l <= active['sl_price']:
                    trades.append({'outcome': 'BREAKEVEN' if active['is_be'] else 'LOSS', 'pnl_r': 0.0 if active['is_be'] else -1.0, 'pnl_usd': 0.0 if active['is_be'] else -risk_usd})
                    active = None
                    continue
                elif h >= tp:
                    rr = (tp - entry) / risk
                    trades.append({'outcome': 'WIN', 'pnl_r': rr, 'pnl_usd': risk_usd * rr})
                    active = None
                    continue
            elif side == 'SELL':
                if not active['is_be'] and l <= entry - (risk * be_r):
                    active['sl_price'] = entry
                    active['is_be'] = True
                if h >= active['sl_price']:
                    trades.append({'outcome': 'BREAKEVEN' if active['is_be'] else 'LOSS', 'pnl_r': 0.0 if active['is_be'] else -1.0, 'pnl_usd': 0.0 if active['is_be'] else -risk_usd})
                    active = None
                    continue
                elif l <= tp:
                    rr = (entry - tp) / risk
                    trades.append({'outcome': 'WIN', 'pnl_r': rr, 'pnl_usd': risk_usd * rr})
                    active = None
                    continue

        if active is None:
            # London Judas window (07:00 - 10:00 UTC)
            if not (7 <= hr <= 10):
                continue

            asian = asian_ranges.get(cur_d)
            if not asian or asian['size'] <= 0:
                continue

            a_high, a_low, a_mid = asian['high'], asian['low'], asian['mid']
            rec_low = df['Low'].iloc[i-4:i+1].min()
            rec_high = df['High'].iloc[i-4:i+1].max()

            # Bullish Asian Judas Sweep
            sweep_d = a_low - rec_low
            if min_sweep <= sweep_d <= max_sweep:
                body = c - o
                tot_r = h - l
                if c > a_low and body > 0 and tot_r > 0 and (body / tot_r >= 0.50):
                    entry = c
                    sl = rec_low - sl_buffer
                    risk = entry - sl
                    if risk > 0:
                        tp = max(a_high, entry + (risk * target_rr))
                        if (tp - entry) / risk >= target_rr:
                            active = {'side': 'BUY', 'entry_price': entry, 'sl_price': sl, 'tp_price': tp, 'risk': risk, 'is_be': False}

            # Bearish Asian Judas Sweep
            sweep_u = rec_high - a_high
            if min_sweep <= sweep_u <= max_sweep:
                body = o - c
                tot_r = h - l
                if c < a_high and body > 0 and tot_r > 0 and (body / tot_r >= 0.50):
                    entry = c
                    sl = rec_high + sl_buffer
                    risk = sl - entry
                    if risk > 0:
                        tp = min(a_low, entry - (risk * target_rr))
                        if (entry - tp) / risk >= target_rr:
                            active = {'side': 'SELL', 'entry_price': entry, 'sl_price': sl, 'tp_price': tp, 'risk': risk, 'is_be': False}

    return _compile_stats(trades, asset_name, "Asian Judas CRT (London Killzone)", ticker)


def _compile_stats(trades, asset_name, model_name, ticker):
    if not trades:
        return None
    df_t = pd.DataFrame(trades)
    tot = len(df_t)
    w = len(df_t[df_t['outcome'] == 'WIN'])
    l = len(df_t[df_t['outcome'] == 'LOSS'])
    be = len(df_t[df_t['outcome'] == 'BREAKEVEN'])

    win_rate = (w / tot) * 100.0 if tot > 0 else 0.0
    eff_win_rate = (w / (w + l) * 100.0) if (w + l) > 0 else 0.0
    pnl_usd = df_t['pnl_usd'].sum()
    pnl_r = df_t['pnl_r'].sum()
    gp = df_t[df_t['pnl_usd'] > 0]['pnl_usd'].sum()
    gl = abs(df_t[df_t['pnl_usd'] < 0]['pnl_usd'].sum())
    pf = (gp / gl) if gl > 0 else 999.0
    avg_win_rr = df_t[df_t['outcome'] == 'WIN']['pnl_r'].mean() if w > 0 else 0.0

    df_t['cum_pnl'] = df_t['pnl_usd'].cumsum()
    cum_max = df_t['cum_pnl'].cummax()
    max_dd = (cum_max - df_t['cum_pnl']).max()

    return {
        "asset": asset_name,
        "model": model_name,
        "ticker": ticker,
        "total_trades": tot,
        "wins": w,
        "losses": l,
        "breakevens": be,
        "win_rate": round(win_rate, 2),
        "eff_win_rate": round(eff_win_rate, 2),
        "profit_factor": round(pf, 2),
        "pnl_usd": round(pnl_usd, 2),
        "pnl_r": round(pnl_r, 2),
        "avg_win_rr": round(avg_win_rr, 2),
        "max_drawdown": round(max_dd, 2)
    }


def print_summary_table(results):
    print("\n" + "=" * 88)
    print("      INSTITUTIONAL CANDLE RANGE THEORY (CRT) MULTI-ASSET BACKTEST REPORT")
    print("=" * 88)
    print(f"{'Asset':<18} | {'Model':<22} | {'Trades':<6} | {'Win %':<6} | {'Avg RR':<7} | {'PF':<5} | {'Net PnL ($)':<12}")
    print("-" * 88)
    for r in results:
        pnl_str = f"+${r['pnl_usd']:,.2f}" if r['pnl_usd'] >= 0 else f"-${abs(r['pnl_usd']):,.2f}"
        print(f"{r['asset']:<18} | {r['model'][:22]:<22} | {r['total_trades']:<6} | {r['win_rate']:<5.1f}% | 1:{r['avg_win_rr']:<4.2f} | {r['profit_factor']:<5.2f} | {pnl_str:<12}")
    print("=" * 88 + "\n")


if __name__ == "__main__":
    assets = [
        {"ticker": "GC=F", "name": "Gold (XAUUSD)", "min_sweep": 0.60, "max_sweep": 12.0, "sl_buf": 0.80},
        {"ticker": "SI=F", "name": "Silver (XAGUSD)", "min_sweep": 0.06, "max_sweep": 0.70, "sl_buf": 0.05},
        {"ticker": "EURUSD=X", "name": "EUR/USD", "min_sweep": 0.0004, "max_sweep": 0.0035, "sl_buf": 0.0003},
        {"ticker": "BTC-USD", "name": "Bitcoin (BTCUSD)", "min_sweep": 60.0, "max_sweep": 900.0, "sl_buf": 50.0}
    ]

    all_res = []
    for a in assets:
        try:
            logger.info(f"Processing {a['name']}...")
            df_5m, df_1h = fetch_aligned_market_data(a['ticker'], period="30d")
            
            # Model A: H1 Range CRT
            res_a = run_h1_candle_crt_backtest(df_5m, df_1h, a['ticker'], a['name'], a['min_sweep'], a['max_sweep'], a['sl_buf'])
            if res_a:
                all_res.append(res_a)

            # Model B: Asian Judas CRT
            res_b = run_asian_judas_crt_backtest(df_5m, a['ticker'], a['name'], a['min_sweep'], a['max_sweep'], a['sl_buf'])
            if res_b:
                all_res.append(res_b)

        except Exception as e:
            logger.error(f"Error processing {a['name']}: {e}", exc_info=True)

    print_summary_table(all_res)
