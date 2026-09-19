"""
optimize_gold_ict.py - Deep Optimization & Overhaul for Gold ICT Engine
Tests:
  1. Consequent Encroachment (50% FVG Midpoint Entry) vs Outer Boundary Entry
  2. Trend Bias Alignment (50/200 EMA Filter)
  3. Solid Body Displacement Ratio (>65% Body)
  4. Dynamic RR (1:1.5 vs 1:2.0 vs 1:2.5)
  5. Fractal Swing Liquidity Lookback (15 vs 25 vs 40 bars)
"""

import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_gold():
    df = yf.download("GC=F", period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for c in ["Open", "High", "Low", "Close"]:
        df[c] = df[c].astype(float)
    # 50 EMA for trend
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def test_gold_strategy(df: pd.DataFrame,
                       lookback: int = 25,
                       disp_threshold: float = 2.0,
                       min_fvg_gap: float = 0.3,
                       sl_buffer: float = 2.5,
                       rr_ratio: float = 2.0,
                       entry_mode: str = "outer", # "outer" or "ce" (50% midpoint)
                       body_ratio_req: float = 0.0, # e.g. 0.65 for solid displacement
                       use_trend: bool = False,
                       risk_usd: float = 50.0) -> dict:

    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    ema50 = df['EMA50'].values
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
                target_entry = pending_fvg['target_entry']
                sl = pending_fvg['sl']

                # Retest check
                retest_hit = False
                if entry_mode == "ce":
                    # Consequent encroachment (price touches CE level)
                    if side == "BUY" and l <= target_entry and h >= fvg_l:
                        retest_hit = True
                    elif side == "SELL" and h >= target_entry and l <= fvg_h:
                        retest_hit = True
                else:
                    # Outer FVG boundary
                    if (fvg_l <= mid <= fvg_h) or (side == "BUY" and l <= fvg_h and h >= fvg_l) or (side == "SELL" and h >= fvg_l and l <= fvg_h):
                        retest_hit = True

                if retest_hit:
                    exec_px = target_entry
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

        # Trend filter
        trend_up = closes[i] > ema50[i]
        trend_down = closes[i] < ema50[i]

        # Bearish Setup
        swept_h = (highs[i-2] > recent_high and closes[i-2] < recent_high) or \
                  (highs[i-1] > recent_high and closes[i-1] < recent_high)
        has_bearish_fvg = lows[i-2] > (highs[i] + min_fvg_gap)
        disp_range = highs[i-1] - lows[i-1]
        disp_body = abs(closes[i-1] - opens[i-1])
        disp_down = closes[i-1] < opens[i-1] and disp_range > disp_threshold
        body_ok = (disp_body / max(0.01, disp_range)) >= body_ratio_req

        # CISD Check: close below sweep candle open
        sweep_open = opens[i-2] if highs[i-2] >= highs[i-1] else opens[i-1]
        cisd_down = (closes[i-1] < sweep_open) or (closes[i] < sweep_open)

        trend_bear_ok = (not use_trend) or trend_down

        if swept_h and has_bearish_fvg and disp_down and cisd_down and body_ok and trend_bear_ok:
            sweep_peak = max(highs[i-2], highs[i-1])
            sl = round(sweep_peak + sl_buffer, 2)
            f_h = round(lows[i-2], 2)
            f_l = round(highs[i], 2)
            ce = round((f_h + f_l) / 2.0, 2)
            pending_fvg = {
                'side': "SELL",
                'fvg_high': f_h,
                'fvg_low': f_l,
                'target_entry': ce if entry_mode == "ce" else f_l,
                'sl': sl,
                'bars_waited': 0
            }
            continue

        # Bullish Setup
        swept_l = (lows[i-2] < recent_low and closes[i-2] > recent_low) or \
                  (lows[i-1] < recent_low and closes[i-1] > recent_low)
        has_bullish_fvg = highs[i-2] < (lows[i] - min_fvg_gap)
        disp_range = highs[i-1] - lows[i-1]
        disp_body = abs(closes[i-1] - opens[i-1])
        disp_up = closes[i-1] > opens[i-1] and disp_range > disp_threshold
        body_ok = (disp_body / max(0.01, disp_range)) >= body_ratio_req

        sweep_open = opens[i-2] if lows[i-2] <= lows[i-1] else opens[i-1]
        cisd_up = (closes[i-1] > sweep_open) or (closes[i] > sweep_open)

        trend_bull_ok = (not use_trend) or trend_up

        if swept_l and has_bullish_fvg and disp_up and cisd_up and body_ok and trend_bull_ok:
            sweep_trough = min(lows[i-2], lows[i-1])
            sl = round(sweep_trough - sl_buffer, 2)
            f_l = round(highs[i-2], 2)
            f_h = round(lows[i], 2)
            ce = round((f_h + f_l) / 2.0, 2)
            pending_fvg = {
                'side': "BUY",
                'fvg_high': f_h,
                'fvg_low': f_l,
                'target_entry': ce if entry_mode == "ce" else f_h,
                'sl': sl,
                'bars_waited': 0
            }
            continue

    if not trades:
        return {"Trades": 0, "Wins": 0, "Losses": 0, "BE": 0, "Eff WR": "0.0%", "PnL ($)": 0.0, "ROC (%)": "0.0%"}

    df_t = pd.DataFrame(trades)
    wins = len(df_t[df_t['result'] == 'WIN'])
    losses = len(df_t[df_t['result'] == 'LOSS'])
    bes = len(df_t[df_t['result'] == 'BREAKEVEN'])
    eff_wr = (wins / max(1, (wins + losses))) * 100.0
    pnl = df_t['pnl_usd'].sum()
    roc = (pnl / 1000.0) * 100.0

    return {
        "Trades": len(df_t),
        "Wins": wins,
        "Losses": losses,
        "BE": bes,
        "Eff WR": f"{eff_wr:.1f}%",
        "PnL ($)": round(pnl, 2),
        "ROC (%)": f"{roc:+.1f}%"
    }

def main():
    print("Fetching Gold 15M candles (60 days)...")
    df = fetch_gold()
    print(f"Loaded {len(df)} candles.")

    configs = [
        {"name": "1. Baseline (Current Engine)", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 2.5, "rr": 2.0, "entry": "outer", "body": 0.0, "trend": False},
        {"name": "2. Body Ratio > 50% (Outer Entry)", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 2.5, "rr": 2.0, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "3. Body Ratio > 60% (Outer Entry)", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 2.5, "rr": 2.0, "entry": "outer", "body": 0.60, "trend": False},
        {"name": "4. Body Ratio > 50% + 1:2.5 RR", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 2.5, "rr": 2.5, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "5. Body Ratio > 50% + 1:3.0 RR", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 2.5, "rr": 3.0, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "6. Body Ratio > 50% + SL Buffer $3.00", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 3.0, "rr": 2.0, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "7. Body Ratio > 50% + SL Buffer $3.00 + 1:2.5 RR", "lookback": 22, "disp": 2.0, "fvg": 0.3, "sl_buf": 3.0, "rr": 2.5, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "8. Disp $2.50 + Body > 50% + 1:2.0 RR", "lookback": 22, "disp": 2.5, "fvg": 0.3, "sl_buf": 2.5, "rr": 2.0, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "9. Disp $2.50 + Body > 50% + SL $3.00 + 1:2.5 RR", "lookback": 22, "disp": 2.5, "fvg": 0.3, "sl_buf": 3.0, "rr": 2.5, "entry": "outer", "body": 0.50, "trend": False},
        {"name": "10. Disp $2.50 + Body > 50% + SL $3.00 + 1:3.0 RR", "lookback": 22, "disp": 2.5, "fvg": 0.3, "sl_buf": 3.0, "rr": 3.0, "entry": "outer", "body": 0.50, "trend": False},
    ]

    results = []
    for c in configs:
        res = test_gold_strategy(
            df,
            lookback=c["lookback"],
            disp_threshold=c["disp"],
            min_fvg_gap=c["fvg"],
            sl_buffer=c["sl_buf"],
            rr_ratio=c["rr"],
            entry_mode=c["entry"],
            body_ratio_req=c["body"],
            use_trend=c["trend"]
        )
        res["Configuration"] = c["name"]
        results.append(res)

    res_df = pd.DataFrame(results)
    cols = ["Configuration", "Trades", "Wins", "Losses", "BE", "Eff WR", "PnL ($)", "ROC (%)"]
    print("\n" + "=" * 95)
    print("🏆 GOLD ICT STRATEGY OVERHAUL OPTIMIZATION MATRIX")
    print("=" * 95)
    print(res_df[cols].to_string(index=False))

if __name__ == "__main__":
    main()
