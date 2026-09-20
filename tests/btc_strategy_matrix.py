"""
tests/btc_strategy_matrix.py - Lean & Fast Bitcoin (BTC/USD) ICT/SMC Backtest Suite
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_btc():
    print("📥 Loading BTC-USD 15M Data (60 Days)...")
    df = yf.download("BTC-USD", period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def simulate_btc(df, name, lb=24, disp_th=150.0, min_fvg=25.0, sl_buf=50.0, rr=2.0, be_r=1.0, use_ema=False):
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    ema200 = df['EMA200'].values
    n = len(df)
    
    trades = []
    active = None
    pending = None
    
    for i in range(lb + 5, n):
        h, l = highs[i], lows[i]
        mid = (h + l) * 0.5
        
        # 1. Manage Active Trade
        if active is not None:
            side = active['side']
            sl = active['sl']
            tp = active['tp']
            entry = active['entry']
            risk_dist = abs(entry - active['init_sl'])
            
            # Breakeven check
            if be_r is not None and not active['be']:
                hit_be = (h - entry >= risk_dist * be_r) if side == "BUY" else (entry - l >= risk_dist * be_r)
                if hit_be:
                    active['sl'] = round(entry + min_fvg * 0.2 if side == "BUY" else entry - min_fvg * 0.2, 2)
                    active['be'] = True
                    sl = active['sl']
            
            if side == "BUY":
                if l <= sl:
                    trades.append({'result': 'BE' if active['be'] else 'LOSS', 'pnl': 0.0 if active['be'] else -50.0})
                    active = None
                elif h >= tp:
                    trades.append({'result': 'WIN', 'pnl': 50.0 * rr})
                    active = None
            else: # SELL
                if h >= sl:
                    trades.append({'result': 'BE' if active['be'] else 'LOSS', 'pnl': 0.0 if active['be'] else -50.0})
                    active = None
                elif l <= tp:
                    trades.append({'result': 'WIN', 'pnl': 50.0 * rr})
                    active = None
            continue
            
        # 2. Manage Pending FVG
        if pending is not None:
            pending['bars'] += 1
            if pending['bars'] > 12: # 3 hour expiry
                pending = None
            else:
                side = pending['side']
                fh, fl, sl = pending['fh'], pending['fl'], pending['sl']
                in_fvg = (fl <= mid <= fh) or (side == "BUY" and l <= fh and h >= fl) or (side == "SELL" and h >= fl and l <= fh)
                if in_fvg:
                    entry = fh if side == "BUY" else fl
                    rd = abs(entry - sl)
                    if rd >= 20.0: # Filter out noise
                        tp = round(entry + rd * rr if side == "BUY" else entry - rd * rr, 2)
                        active = {'side': side, 'entry': entry, 'init_sl': sl, 'sl': sl, 'tp': tp, 'be': False}
                        pending = None
                        continue
        
        # 3. Detect Setup
        rh = max(highs[i-lb:i-2])
        rl = min(lows[i-lb:i-2])
        
        # Bearish CISD + FVG
        swept_h = (highs[i-2] > rh and closes[i-2] < rh) or (highs[i-1] > rh and closes[i-1] < rh)
        fvg_d = lows[i-2] > (highs[i] + min_fvg)
        dr_d = highs[i-1] - lows[i-1]
        db_d = abs(closes[i-1] - opens[i-1])
        disp_d = closes[i-1] < opens[i-1] and dr_d > disp_th and (db_d / max(0.001, dr_d)) >= 0.45
        sweep_open_h = opens[i-2] if highs[i-2] >= highs[i-1] else opens[i-1]
        cisd_d = (closes[i-1] < sweep_open_h) or (closes[i] < sweep_open_h)
        trend_d = (not use_ema) or (closes[i] < ema200[i])
        
        if swept_h and fvg_d and disp_d and cisd_d and trend_d:
            sl = max(highs[i-2], highs[i-1]) + sl_buf
            pending = {'side': "SELL", 'fh': lows[i-2], 'fl': highs[i], 'sl': sl, 'bars': 0}
            continue
            
        # Bullish CISD + FVG
        swept_l = (lows[i-2] < rl and closes[i-2] > rl) or (lows[i-1] < rl and closes[i-1] > rl)
        fvg_u = highs[i-2] < (lows[i] - min_fvg)
        dr_u = highs[i-1] - lows[i-1]
        db_u = abs(closes[i-1] - opens[i-1])
        disp_u = closes[i-1] > opens[i-1] and dr_u > disp_th and (db_u / max(0.001, dr_u)) >= 0.45
        sweep_open_l = opens[i-2] if lows[i-2] <= lows[i-1] else opens[i-1]
        cisd_u = (closes[i-1] > sweep_open_l) or (closes[i] > sweep_open_l)
        trend_u = (not use_ema) or (closes[i] > ema200[i])
        
        if swept_l and fvg_u and disp_u and cisd_u and trend_u:
            sl = min(lows[i-2], lows[i-1]) - sl_buf
            pending = {'side': "BUY", 'fh': lows[i], 'fl': highs[i-2], 'sl': sl, 'bars': 0}
            continue
            
    if not trades:
        return {'Configuration': name, 'Trades': 0, 'Wins': 0, 'Losses': 0, 'BE': 0, 'Win Rate': '0.0%', 'PnL ($)': 0.0, 'ROC (%)': '0.0%'}
        
    tdf = pd.DataFrame(trades)
    wins = len(tdf[tdf['result'] == 'WIN'])
    losses = len(tdf[tdf['result'] == 'LOSS'])
    bes = len(tdf[tdf['result'] == 'BE'])
    wr = (wins / max(1, wins + losses)) * 100
    pnl = tdf['pnl'].sum()
    roc = (pnl / 1000.0) * 100.0
    
    return {
        'Configuration': name,
        'Trades': len(tdf),
        'Wins': wins,
        'Losses': losses,
        'BE': bes,
        'Win Rate': f"{wr:.1f}%",
        'PnL ($)': round(pnl, 2),
        'ROC (%)': f"{roc:+.1f}%"
    }

def main():
    df = fetch_btc()
    print(f"Loaded {len(df)} candles for Bitcoin (60 days 15M).\n")
    
    configs = [
        # Scalp / Fast Reversal
        {"name": "BTC-1: Aggressive Scalp (Disp $120, FVG $15, SL $30, 1:1.8 RR, BE 0.8R)", "lb": 16, "disp_th": 120.0, "min_fvg": 15.0, "sl_buf": 30.0, "rr": 1.8, "be_r": 0.8, "use_ema": False},
        {"name": "BTC-2: Fast FVG Scalp (Disp $150, FVG $20, SL $40, 1:2.0 RR, BE 1.0R)", "lb": 16, "disp_th": 150.0, "min_fvg": 20.0, "sl_buf": 40.0, "rr": 2.0, "be_r": 1.0, "use_ema": False},
        {"name": "BTC-3: Trend-Aligned Scalp (Disp $150, FVG $20, SL $40, 1:2.0 RR + EMA200)", "lb": 16, "disp_th": 150.0, "min_fvg": 20.0, "sl_buf": 40.0, "rr": 2.0, "be_r": 1.0, "use_ema": True},
        
        # Medium Swing / SMC Standard
        {"name": "BTC-4: Standard SMC (Disp $180, FVG $25, SL $50, 1:2.0 RR, BE 1.0R)", "lb": 24, "disp_th": 180.0, "min_fvg": 25.0, "sl_buf": 50.0, "rr": 2.0, "be_r": 1.0, "use_ema": False},
        {"name": "BTC-5: Standard SMC + 1:2.5 RR (Disp $180, FVG $25, SL $50, 1:2.5 RR)", "lb": 24, "disp_th": 180.0, "min_fvg": 25.0, "sl_buf": 50.0, "rr": 2.5, "be_r": 1.0, "use_ema": False},
        {"name": "BTC-6: Standard SMC + EMA Filter (Disp $180, FVG $25, SL $50, 1:2.5 RR)", "lb": 24, "disp_th": 180.0, "min_fvg": 25.0, "sl_buf": 50.0, "rr": 2.5, "be_r": 1.0, "use_ema": True},
        
        # High Volatility / Institutional
        {"name": "BTC-7: Institutional Impulse (Disp $250, FVG $35, SL $75, 1:2.0 RR)", "lb": 24, "disp_th": 250.0, "min_fvg": 35.0, "sl_buf": 75.0, "rr": 2.0, "be_r": 1.0, "use_ema": False},
        {"name": "BTC-8: Institutional Impulse + 1:2.5 RR (Disp $250, FVG $35, SL $75)", "lb": 24, "disp_th": 250.0, "min_fvg": 35.0, "sl_buf": 75.0, "rr": 2.5, "be_r": 1.0, "use_ema": False},
        {"name": "BTC-9: Wide Swing Liquidity (Disp $200, FVG $30, SL $60, 1:2.5 RR, lb=36)", "lb": 36, "disp_th": 200.0, "min_fvg": 30.0, "sl_buf": 60.0, "rr": 2.5, "be_r": 1.0, "use_ema": False},
        {"name": "BTC-10: Wide Swing + 1:3.0 RR (Disp $200, FVG $30, SL $60, 1:3.0 RR, lb=36)", "lb": 36, "disp_th": 200.0, "min_fvg": 30.0, "sl_buf": 60.0, "rr": 3.0, "be_r": 1.0, "use_ema": False},
        
        # No Breakeven (Let Runner Hit Full TP)
        {"name": "BTC-11: Raw Target (No BE) (Disp $150, FVG $20, SL $50, 1:2.0 RR)", "lb": 24, "disp_th": 150.0, "min_fvg": 20.0, "sl_buf": 50.0, "rr": 2.0, "be_r": None, "use_ema": False},
        {"name": "BTC-12: Raw Target (No BE) (Disp $180, FVG $25, SL $50, 1:2.5 RR)", "lb": 24, "disp_th": 180.0, "min_fvg": 25.0, "sl_buf": 50.0, "rr": 2.5, "be_r": None, "use_ema": False},
    ]
    
    rows = [simulate_btc(df, **c) for c in configs]
    res_df = pd.DataFrame(rows)
    print(res_df.to_string(index=False))

if __name__ == "__main__":
    main()
