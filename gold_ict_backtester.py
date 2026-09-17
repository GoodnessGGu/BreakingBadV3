"""
gold_ict_backtester.py - Enhanced Institutional ICT / SMC Gold Backtester
Includes:
1. Session Timing Filter: London Session & New York Session only (no dead-hours chop).
2. Trend Filter: 50 EMA on 15m/1h to ensure trading with institutional orderflow.
3. Dynamic Breakeven: Once trade reaches 1.0R in profit, SL moves to entry price.
4. R:R Target: 1:2.0 with Breakeven protection.
"""

import sys, os
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timezone

def fetch_gold_data(period="60d", interval="5m"):
    print(f"Downloading {period} of {interval} Gold (GC=F) data...")
    df = yf.download("GC=F", period=period, interval=interval, progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    return df

def find_swing_points(df, window=5):
    highs = df['High'].values
    lows = df['Low'].values
    n = len(df)
    swing_highs = np.full(n, np.nan)
    swing_lows = np.full(n, np.nan)
    for i in range(window, n - window):
        if all(highs[i] > highs[i - w] for w in range(1, window + 1)) and \
           all(highs[i] >= highs[i + w] for w in range(1, window + 1)):
            swing_highs[i] = highs[i]
        if all(lows[i] < lows[i - w] for w in range(1, window + 1)) and \
           all(lows[i] <= lows[i + w] for w in range(1, window + 1)):
            swing_lows[i] = lows[i]
    df['Swing_High'] = swing_highs
    df['Swing_Low'] = swing_lows
    return df

def backtest_ict_enhanced(df, rr_ratio=2.0, sl_buffer=2.0, use_session_filter=True, use_trend_filter=True):
    # Calculate 50 EMA for trend filter
    df['EMA_50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df = find_swing_points(df, window=5)

    trades = []
    active_trade = None
    pending_setup = None

    opens = df['Open'].values
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    emas = df['EMA_50'].values
    times = df.index

    last_swing_high = None
    last_swing_low = None

    for i in range(10, len(df) - 1):
        cur_time = times[i]
        # Convert index time to UTC hour
        hour_utc = cur_time.tz_convert("UTC").hour if cur_time.tzinfo else cur_time.hour
        is_active_session = (7 <= hour_utc <= 18) # London + New York sessions (07:00 to 18:00 UTC)

        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        ema = emas[i]

        if not np.isnan(df['Swing_High'].iloc[i - 5]):
            last_swing_high = df['Swing_High'].iloc[i - 5]
        if not np.isnan(df['Swing_Low'].iloc[i - 5]):
            last_swing_low = df['Swing_Low'].iloc[i - 5]

        # ── Manage Active Trade ──────────────────────────────
        if active_trade is not None:
            active_trade['bars_held'] += 1
            side = active_trade['side']
            sl = active_trade['sl']
            tp = active_trade['tp']
            entry = active_trade['entry_price']
            risk_dist = abs(entry - active_trade['initial_sl'])

            # Breakeven check: If reached 1.0R, move SL to entry
            if not active_trade['moved_to_be']:
                if side == "BUY" and (h - entry) >= risk_dist:
                    active_trade['sl'] = entry + 0.5 # lock in small gain / cover fees
                    active_trade['moved_to_be'] = True
                    sl = active_trade['sl']
                elif side == "SELL" and (entry - l) >= risk_dist:
                    active_trade['sl'] = entry - 0.5
                    active_trade['moved_to_be'] = True
                    sl = active_trade['sl']

            if side == "BUY":
                if l <= sl:
                    active_trade['exit_price'] = sl
                    active_trade['exit_time'] = cur_time
                    if active_trade['moved_to_be']:
                        active_trade['result'] = "BREAKEVEN"
                        active_trade['pnl_usd'] = 0.0
                    else:
                        active_trade['result'] = "LOSS"
                        active_trade['pnl_usd'] = -active_trade['risk_usd']
                    active_trade['pips'] = (sl - entry) * 10
                    trades.append(active_trade)
                    active_trade = None
                elif h >= tp:
                    active_trade['exit_price'] = tp
                    active_trade['exit_time'] = cur_time
                    active_trade['result'] = "WIN"
                    active_trade['pnl_usd'] = active_trade['risk_usd'] * rr_ratio
                    active_trade['pips'] = (tp - entry) * 10
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
                        active_trade['pnl_usd'] = -active_trade['risk_usd']
                    active_trade['pips'] = (entry - sl) * 10
                    trades.append(active_trade)
                    active_trade = None
                elif l <= tp:
                    active_trade['exit_price'] = tp
                    active_trade['exit_time'] = cur_time
                    active_trade['result'] = "WIN"
                    active_trade['pnl_usd'] = active_trade['risk_usd'] * rr_ratio
                    active_trade['pips'] = (entry - tp) * 10
                    trades.append(active_trade)
                    active_trade = None

            continue

        # ── Manage Pending Setup ─────────────────────────────
        if pending_setup is not None:
            pending_setup['bars_waited'] += 1
            if pending_setup['bars_waited'] > 15:
                pending_setup = None
            else:
                side = pending_setup['side']
                fvg_h = pending_setup['fvg_high']
                fvg_l = pending_setup['fvg_low']
                sl = pending_setup['sl']

                if side == "BUY" and l <= fvg_h and h >= fvg_l:
                    entry_px = min(o, fvg_h)
                    risk_dist = entry_px - sl
                    if 1.0 < risk_dist < 15.0:
                        active_trade = {
                            'entry_time': cur_time,
                            'side': "BUY",
                            'entry_price': entry_px,
                            'sl': sl,
                            'initial_sl': sl,
                            'tp': entry_px + (risk_dist * rr_ratio),
                            'risk_usd': 50.0,
                            'bars_held': 0,
                            'moved_to_be': False,
                            'session': "London/NY" if is_active_session else "Asian/Off"
                        }
                        pending_setup = None
                        continue
                elif side == "SELL" and h >= fvg_l and l <= fvg_h:
                    entry_px = max(o, fvg_l)
                    risk_dist = sl - entry_px
                    if 1.0 < risk_dist < 15.0:
                        active_trade = {
                            'entry_time': cur_time,
                            'side': "SELL",
                            'entry_price': entry_px,
                            'sl': sl,
                            'initial_sl': sl,
                            'tp': entry_px - (risk_dist * rr_ratio),
                            'risk_usd': 50.0,
                            'bars_held': 0,
                            'moved_to_be': False,
                            'session': "London/NY" if is_active_session else "Asian/Off"
                        }
                        pending_setup = None
                        continue

        # ── Setup Filters ────────────────────────────────────
        if use_session_filter and not is_active_session:
            continue

        # 1. Bearish Liquidity Sweep + FVG (Sell)
        if last_swing_high is not None:
            swept_high = (highs[i-2] > last_swing_high and closes[i-2] < last_swing_high) or \
                         (highs[i-1] > last_swing_high and closes[i-1] < last_swing_high)
            has_bearish_fvg = lows[i-2] > (highs[i] + 0.3)
            is_displacement_down = closes[i-1] < opens[i-1] and (highs[i-1] - lows[i-1]) > 2.0
            trend_ok = (c < ema) if use_trend_filter else True

            if swept_high and has_bearish_fvg and is_displacement_down and trend_ok:
                sweep_peak = max(highs[i-2], highs[i-1])
                sl = sweep_peak + sl_buffer
                pending_setup = {
                    'side': "SELL",
                    'fvg_high': lows[i-2],
                    'fvg_low': highs[i],
                    'sl': sl,
                    'bars_waited': 0
                }

        # 2. Bullish Liquidity Sweep + FVG (Buy)
        if last_swing_low is not None and pending_setup is None:
            swept_low = (lows[i-2] < last_swing_low and closes[i-2] > last_swing_low) or \
                        (lows[i-1] < last_swing_low and closes[i-1] > last_swing_low)
            has_bullish_fvg = highs[i-2] < (lows[i] - 0.3)
            is_displacement_up = closes[i-1] > opens[i-1] and (highs[i-1] - lows[i-1]) > 2.0
            trend_ok = (c > ema) if use_trend_filter else True

            if swept_low and has_bullish_fvg and is_displacement_up and trend_ok:
                sweep_trough = min(lows[i-2], lows[i-1])
                sl = sweep_trough - sl_buffer
                pending_setup = {
                    'side': "BUY",
                    'fvg_high': lows[i],
                    'fvg_low': highs[i-2],
                    'sl': sl,
                    'bars_waited': 0
                }

    return pd.DataFrame(trades)

def print_enhanced_report(trades_df, initial_balance=1000.0):
    if len(trades_df) == 0:
        print("No trades triggered.")
        return

    total = len(trades_df)
    wins = len(trades_df[trades_df['result'] == 'WIN'])
    losses = len(trades_df[trades_df['result'] == 'LOSS'])
    bes = len(trades_df[trades_df['result'] == 'BREAKEVEN'])
    win_rate = (wins / total) * 100
    effective_win_rate = (wins / max(1, (wins + losses))) * 100

    total_pnl = trades_df['pnl_usd'].sum()
    total_pips = trades_df['pips'].sum()
    gross_win = trades_df[trades_df['pnl_usd'] > 0]['pnl_usd'].sum()
    gross_loss = abs(trades_df[trades_df['pnl_usd'] < 0]['pnl_usd'].sum())
    profit_factor = (gross_win / gross_loss) if gross_loss > 0 else float('inf')

    equity_curve = initial_balance + trades_df['pnl_usd'].cumsum()
    peak = equity_curve.cummax()
    drawdown = (peak - equity_curve)
    max_dd = drawdown.max()
    max_dd_pct = (drawdown / peak).max() * 100

    print("=" * 72)
    print("   ENHANCED ICT / SMC GOLD STRATEGY (LONDON + NY + BREAKEVEN + TREND)")
    print("=" * 72)
    print(f"Period Covered       : {trades_df['entry_time'].iloc[0]} -> {trades_df['entry_time'].iloc[-1]}")
    print(f"Total Trades Taken   : {total}")
    print(f"Outcome Breakdown    : {wins} Wins | {losses} Losses | {bes} Breakeven (0 risk)")
    print(f"Raw Win Rate         : {win_rate:.2f}% (including BE)")
    print(f"Effective Win Rate   : {effective_win_rate:.2f}% (excluding BE)")
    print(f"Profit Factor        : {profit_factor:.2f}")
    print(f"Net Profit (USD)     : ${total_pnl:.2f} (Risk $50/trade, 1000 base)")
    print(f"Return on Capital    : +{(total_pnl / initial_balance) * 100:.1f}%")
    print(f"Total Pips Bagged    : +{total_pips:.1f} pips")
    print(f"Max Drawdown         : ${max_dd:.2f} ({max_dd_pct:.2f}%)")
    print("=" * 72)
    print("\nRecent 10 Trades:")
    cols = ['entry_time', 'side', 'entry_price', 'sl', 'tp', 'exit_price', 'pnl_usd', 'pips', 'result']
    print(trades_df[cols].tail(10).to_string(index=False))

if __name__ == "__main__":
    df = fetch_gold_data(period="30d", interval="15m")
    trades = backtest_ict_enhanced(df, rr_ratio=4.0, sl_buffer=2.0, use_session_filter=False, use_trend_filter=False)
    print_enhanced_report(trades)