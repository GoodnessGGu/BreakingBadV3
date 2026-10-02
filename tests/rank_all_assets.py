"""
tests/rank_all_assets.py
Consolidated ranking of all assets under Strategy Config E:
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
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from run_config_e_multi_asset import run_config_e, fetch_data_robust

def main():
    assets = [
        ('NZD / USD (NZDUSD)', 'NZDUSD=X', 'Forex Major'),
        ('AUD / USD (AUDUSD)', 'AUDUSD=X', 'Forex Major'),
        ('USD / CAD (USDCAD)', 'USDCAD=X', 'Forex Major'),
        ('EUR / USD (EURUSD)', 'EURUSD=X', 'Forex Major'),
        ('GBP / USD (GBPUSD)', 'GBPUSD=X', 'Forex Major'),
        ('USD / JPY (USDJPY)', 'USDJPY=X', 'Forex Major'),
        ('Gold (XAUUSD)', 'GC=F', 'Commodity'),
        ('Nasdaq 100 (NQ)', 'NQ=F', 'Index'),
        ('S&P 500 (ES)', 'ES=F', 'Index'),
        ('Bitcoin (BTCUSD)', 'BTC-USD', 'Crypto')
    ]

    res_list = []
    print("Fetching and backtesting all assets...")
    for name, ticker, cat in assets:
        df = fetch_data_robust(ticker)
        if len(df) > 500:
            r = run_config_e(df, name)
            r['ticker'] = ticker
            r['category'] = cat
            res_list.append(r)
            print(f"Done: {name}")
        else:
            print(f"Skipped: {name}")
        time.sleep(1)

    df_res = pd.DataFrame(res_list)
    df_sorted_net = df_res.sort_values(by='net_r', ascending=False)

    print("\n" + "="*95)
    print("🏆 ALL ASSETS RANKED BY NET PROFIT (R)")
    print("="*95)
    for idx, row in df_sorted_net.iterrows():
        print(f"{row['asset']:<22} | Net: {row['net_r']:+5.1f}R | PF: {row['pf']:4.2f} | WR: {row['win_rate']:4.1f}% | DD: {row['max_dd']:4.1f}R | Trades: {row['trades']:3d} | W/L/BE: {row['wins']}/{row['losses']}/{row['bes']}")

    df_sorted_pf = df_res.sort_values(by='pf', ascending=False)
    print("\n" + "="*95)
    print("🎯 ALL ASSETS RANKED BY PROFIT FACTOR (PF)")
    print("="*95)
    for idx, row in df_sorted_pf.iterrows():
        print(f"{row['asset']:<22} | PF: {row['pf']:4.2f} | Net: {row['net_r']:+5.1f}R | WR: {row['win_rate']:4.1f}% | DD: {row['max_dd']:4.1f}R | Trades: {row['trades']:3d}")

if __name__ == "__main__":
    main()
