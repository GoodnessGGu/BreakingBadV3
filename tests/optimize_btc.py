"""
tests/optimize_btc.py - Deep Parameter Optimization for Bitcoin ICT SMC
"""
import sys
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", line_buffering=True)
import pandas as pd
import numpy as np
import yfinance as yf

def fetch_btc():
    df = yf.download("BTC-USD", period="60d", interval="15m", progress=False)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    df.dropna(inplace=True)
    for col in ["Open", "High", "Low", "Close"]:
        df[col] = df[col].astype(float)
    df['EMA50'] = df['Close'].ewm(span=50, adjust=False).mean()
    df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()
    return df

def run_sim(df, lookback, disp_th, min_fvg, sl_buf, rr, use_ema_filter, be_r, cisd):
    highs = df['High'].values
    lows = df['Low'].values
    closes = df['Close'].values
    opens = df['Open'].values
    ema200 = df['EMA200'].values
    times = df.index
    n = len(df)
    
    trades = []
    active_trade = None
    pending_fvg = None
    
    for i in range(lookback + 5, n):
        o, h, l, c = opens[i], highs[i], lows[i], closes[i]
        mid = (h + l) / 2.0
        
        # 1. Manage Active Trade
        if active_trade is not None:
            side = active_trade['side']
            sl = active_trade['sl']
            tp = active_trade['tp']
            entry = active_trade['entry']
            risk_dist = abs(entry - active_trade['initial_sl'])
            
            # BE Management
            if be_r is not None and not active_trade['moved_to_be']:
                hit_be = (h - entry >= risk_dist * be_r) if side == "BUY" else (entry - l >= risk_dist * be_r)
                if hit_be:
                    be_buf = min_fvg * 0.3
                    active_trade['sl'] = round(entry + be_buf if side == "BUY" else entry - be_buf, 2)
                    active_trade['moved_to_be'] = True
                    sl = active_trade['sl']
            
            if side == "BUY":
                if l <= sl:
                    is_be = active_trade['moved_to_be']
                    trades.append({"result": "BE" if is_be else "LOSS", "pnl": 0.0 if is_be else -50.0})
                    active_trade = None
                elif h >= tp:
                    trades.append({"result": "WIN", "pnl": 50.0 * rr})
                    active_trade = None
            else:
                if h >= sl:
                    is_be = active_trade['moved_to_be']
                    trades.append({"result": "BE" if is_be else "LOSS", "pnl": 0.0 if is_be else -50.0})
                    active_trade = None
                elif l <= tp:
                    trades.append({"result": "WIN", "pnl": 50.0 * rr})
                    active_trade = None
            continue
            
        # 2. Manage Pending FVG
        if pending_fvg is not None:
            pending_fvg['bars'] += 1
            if pending_fvg['bars'] > 12: # 3 hours
                pending_fvg = None
            else:
                side = pending_fvg['side']
                fvg_h = pending_fvg['fvg_h']
                fvg_l = pending_fvg['fvg_l']
                sl = pending_fvg['sl']
                
                in_fvg = (fvg_l <= mid <= fvg_h) or (side == "BUY" and l <= fvg_h and h >= fvg_l) or (side == "SELL" and h >= fvg_l and l <= fvg_h)
                if in_fvg:
                    entry = fvg_h if side == "BUY" else fvg_l
                    risk_dist = abs(entry - sl)
                    if risk_dist >= 20.0:
                        tp = round(entry + (risk_dist * rr) if side == "BUY" else entry - (risk_dist * rr), 2)
                        active_trade = {
                            'side': side,
                            'entry': entry,
                            'initial_sl': sl,
                            'sl': sl,
                            'tp': tp,
                            'moved_to_be': False
                        }
                        pending_fvg = None
                        continue
        
        # 3. New Setup
        rh = max(highs[i-lookback:i-2])
        rl = min(lows[i-lookback:i-2])
        
        # Bearish
        swept_h = (highs[i-2] > rh and closes[i-2] < rh) or (highs[i-1] > rh and closes[i-1] < rh)
        fvg_down = lows[i-2] > (highs[i] + min_fvg)
        dr_down = highs[i-1] - lows[i-1]
        db_down = abs(closes[i-1] - opens[i-1])
        disp_d = closes[i-1] < opens[i-1] and dr_down > disp_th and (db_down / max(0.001, dr_down)) >= 0.45
        
        sweep_open_h = opens[i-2] if highs[i-2] >= highs[i-1] else opens[i-1]
        cisd_d = (not cisd) or (closes[i-1] < sweep_open_h) or (closes[i] < sweep_open_h)
        trend_d = (not use_ema_filter) or (closes[i] < ema200[i])
        
        if swept_h and fvg_down and disp_d and cisd_d and trend_d:
            sl = max(highs[i-2], highs[i-1]) + sl_buf
            pending_fvg = {'side': "SELL", 'fvg_h': lows[i-2], 'fvg_l': highs[i], 'sl': sl, 'bars': 0}
            continue
            
        # Bullish
        swept_l = (lows[i-2] < rl and closes[i-2] > rl) or (lows[i-1] < rl and closes[i-1] > rl)
        fvg_up = highs[i-2] < (lows[i] - min_fvg)
        dr_up = highs[i-1] - lows[i-1]
        db_up = abs(closes[i-1] - opens[i-1])
        disp_u = closes[i-1] > opens[i-1] and dr_up > disp_th and (db_up / max(0.001, dr_up)) >= 0.45
        
        sweep_open_l = opens[i-2] if lows[i-2] <= lows[i-1] else opens[i-1]
        cisd_u = (not cisd) or (closes[i-1] > sweep_open_l) or (closes[i] > sweep_open_l)
        trend_u = (not use_ema_filter) or (closes[i] > ema200[i])
        
        if swept_l and fvg_up and disp_u and cisd_u and trend_u:
            sl = min(lows[i-2], lows[i-1]) - sl_buf
            pending_fvg = {'side': "BUY", 'fvg_h': lows[i], 'fvg_l': highs[i-2], 'sl': sl, 'bars': 0}
            continue
            
    if not trades:
        return {"trades": 0, "win_rate": 0, "pnl": 0, "wins": 0, "losses": 0, "bes": 0}
    
    tdf = pd.DataFrame(trades)
    wins = len(tdf[tdf['result'] == 'WIN'])
    losses = len(tdf[tdf['result'] == 'LOSS'])
    bes = len(tdf[tdf['result'] == 'BE'])
    wr = (wins / max(1, wins + losses)) * 100
    pnl = tdf['pnl'].sum()
    return {"trades": len(tdf), "wins": wins, "losses": losses, "bes": bes, "win_rate": round(wr, 1), "pnl": round(pnl, 2)}

def main():
    df = fetch_btc()
    print(f"Data loaded: {len(df)} bars")
    
    results = []
    
    lookbacks = [16, 24, 32]
    disp_thresholds = [120.0, 180.0, 250.0, 350.0]
    min_fvgs = [15.0, 25.0, 40.0]
    sl_buffers = [30.0, 50.0, 80.0]
    rrs = [1.8, 2.0, 2.5, 3.0]
    ema_filters = [False, True]
    be_rs = [0.8, 1.0, 1.2, None]
    
    for lb in lookbacks:
        for disp in disp_thresholds:
            for fvg in min_fvgs:
                for sl in sl_buffers:
                    for rr in rrs:
                        for ema in ema_filters:
                            for be in be_rs:
                                res = run_sim(df, lb, disp, fvg, sl, rr, ema, be, True)
                                if res['trades'] >= 10 and res['pnl'] > 0:
                                    res.update({
                                        'lb': lb, 'disp': disp, 'fvg': fvg, 'sl': sl,
                                        'rr': rr, 'ema': ema, 'be': be
                                    })
                                    results.append(res)
    
    rdf = pd.DataFrame(results)
    if len(rdf) > 0:
        rdf = rdf.sort_values(by="pnl", ascending=False)
        print(f"\nTop 15 Parameter Configurations for Bitcoin (out of {len(rdf)} profitable sets):")
        print(rdf[['lb', 'disp', 'fvg', 'sl', 'rr', 'ema', 'be', 'trades', 'wins', 'losses', 'bes', 'win_rate', 'pnl']].head(15).to_string(index=False))
    else:
        print("No profitable sets found with >=10 trades.")

if __name__ == "__main__":
    main()
