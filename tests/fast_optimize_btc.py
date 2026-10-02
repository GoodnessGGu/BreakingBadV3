"""
tests/fast_optimize_btc.py - Fast Parallel Grid Optimizer for Bitcoin ICT
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import pandas as pd
import numpy as np
import yfinance as yf
from concurrent.futures import ProcessPoolExecutor

def fetch_btc():
    df = yf.download("BTC-USD", period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

# Global cache for data
_DF = None

def init_worker(df):
    global _DF
    _DF = df

def eval_config(cfg):
    global _DF
    lb = cfg['lb']
    disp_th = cfg['disp']
    min_fvg = cfg['fvg']
    sl_buf = cfg['sl']
    rr = cfg['rr']
    use_ema = cfg['ema']
    be_r = cfg['be']
    
    highs = _DF['High'].values
    lows = _DF['Low'].values
    closes = _DF['Close'].values
    opens = _DF['Open'].values
    ema200 = _DF['EMA200'].values
    n = len(_DF)
    
    trades = []
    active = None
    pending = None
    
    for i in range(lb + 5, n):
        h, l = highs[i], lows[i]
        mid = (h + l) * 0.5
        
        if active is not None:
            side = active['side']
            sl = active['sl']
            tp = active['tp']
            entry = active['entry']
            risk_dist = abs(entry - active['init_sl'])
            
            if be_r is not None and not active['be']:
                hit_be = (h - entry >= risk_dist * be_r) if side == "BUY" else (entry - l >= risk_dist * be_r)
                if hit_be:
                    active['sl'] = round(entry + min_fvg * 0.2 if side == "BUY" else entry - min_fvg * 0.2, 2)
                    active['be'] = True
                    sl = active['sl']
            
            if side == "BUY":
                if l <= sl:
                    trades.append((0.0 if active['be'] else -50.0, "BE" if active['be'] else "LOSS"))
                    active = None
                elif h >= tp:
                    trades.append((50.0 * rr, "WIN"))
                    active = None
            else:
                if h >= sl:
                    trades.append((0.0 if active['be'] else -50.0, "BE" if active['be'] else "LOSS"))
                    active = None
                elif l <= tp:
                    trades.append((50.0 * rr, "WIN"))
                    active = None
            continue
            
        if pending is not None:
            pending['bars'] += 1
            if pending['bars'] > 12:
                pending = None
            else:
                side = pending['side']
                fh, fl, sl = pending['fh'], pending['fl'], pending['sl']
                in_fvg = (fl <= mid <= fh) or (side == "BUY" and l <= fh and h >= fl) or (side == "SELL" and h >= fl and l <= fh)
                if in_fvg:
                    entry = fh if side == "BUY" else fl
                    rd = abs(entry - sl)
                    if rd >= 20.0:
                        tp = round(entry + rd * rr if side == "BUY" else entry - rd * rr, 2)
                        active = {'side': side, 'entry': entry, 'init_sl': sl, 'sl': sl, 'tp': tp, 'be': False}
                        pending = None
                        continue
                        
        rh = max(highs[i-lb:i-2])
        rl = min(lows[i-lb:i-2])
        
        # Bearish
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
            
        # Bullish
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
        return None
    
    wins = sum(1 for p, r in trades if r == "WIN")
    losses = sum(1 for p, r in trades if r == "LOSS")
    bes = sum(1 for p, r in trades if r == "BE")
    pnl = sum(p for p, r in trades)
    wr = (wins / max(1, wins + losses)) * 100
    
    cfg_copy = dict(cfg)
    cfg_copy.update({
        'trades': len(trades),
        'wins': wins,
        'losses': losses,
        'bes': bes,
        'win_rate': round(wr, 1),
        'pnl': round(pnl, 2),
        'roc': round((pnl / 1000.0) * 100.0, 1)
    })
    return cfg_copy

def main():
    df = fetch_btc()
    print(f"Loaded {len(df)} candles for Bitcoin (60 days 15M).")
    
    grid = []
    for lb in [16, 24, 36]:
        for disp in [100.0, 150.0, 200.0, 300.0]:
            for fvg in [15.0, 25.0, 50.0]:
                for sl in [30.0, 50.0, 80.0]:
                    for rr in [1.5, 2.0, 2.5, 3.0]:
                        for ema in [False, True]:
                            for be in [0.8, 1.0, None]:
                                grid.append({
                                    'lb': lb, 'disp': disp, 'fvg': fvg, 'sl': sl,
                                    'rr': rr, 'ema': ema, 'be': be
                                })
                                
    print(f"Testing {len(grid)} parameter permutations with ProcessPoolExecutor...")
    
    with ProcessPoolExecutor(initializer=init_worker, initargs=(df,)) as executor:
        results = list(executor.map(eval_config, grid))
        
    valid = [r for r in results if r is not None and r['trades'] >= 10 and r['pnl'] > 0]
    
    if valid:
        rdf = pd.DataFrame(valid).sort_values(by="pnl", ascending=False)
        print(f"\n✅ Found {len(rdf)} profitable parameter sets (Min 10 trades).")
        print("\n🏆 TOP 15 BITCOIN ICT CONFIGURATIONS:")
        print(rdf[['lb', 'disp', 'fvg', 'sl', 'rr', 'ema', 'be', 'trades', 'wins', 'losses', 'bes', 'win_rate', 'pnl', 'roc']].head(15).to_string(index=False))
    else:
        print("No profitable sets found.")

if __name__ == "__main__":
    main()
