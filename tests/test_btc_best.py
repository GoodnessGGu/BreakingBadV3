"""
tests/test_btc_best.py - Targeted Fine-Tuning for Bitcoin Trend SMC
"""
import pandas as pd
import yfinance as yf
from btc_strategy_matrix import simulate_btc

df = yf.download("BTC-USD", period="60d", interval="15m", progress=False)
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)
df.dropna(inplace=True)
for col in ["Open", "High", "Low", "Close"]:
    df[col] = df[col].astype(float)
df['EMA200'] = df['Close'].ewm(span=200, adjust=False).mean()

configs = [
    {"name": "Combo A: Trend + Disp $150, FVG $20, SL $50, RR 2.0 (No BE)", "lb": 24, "disp_th": 150.0, "min_fvg": 20.0, "sl_buf": 50.0, "rr": 2.0, "be_r": None, "use_ema": True},
    {"name": "Combo B: Trend + Disp $150, FVG $20, SL $50, RR 2.5 (No BE)", "lb": 24, "disp_th": 150.0, "min_fvg": 20.0, "sl_buf": 50.0, "rr": 2.5, "be_r": None, "use_ema": True},
    {"name": "Combo C: Trend + Disp $140, FVG $18, SL $45, RR 2.0 (BE 1.0R)", "lb": 20, "disp_th": 140.0, "min_fvg": 18.0, "sl_buf": 45.0, "rr": 2.0, "be_r": 1.0, "use_ema": True},
    {"name": "Combo D: Trend + Disp $140, FVG $18, SL $45, RR 2.0 (No BE)", "lb": 20, "disp_th": 140.0, "min_fvg": 18.0, "sl_buf": 45.0, "rr": 2.0, "be_r": None, "use_ema": True},
    {"name": "Combo E: Trend + Disp $160, FVG $20, SL $50, RR 2.2 (BE 1.0R)", "lb": 24, "disp_th": 160.0, "min_fvg": 20.0, "sl_buf": 50.0, "rr": 2.2, "be_r": 1.0, "use_ema": True},
]

rows = [simulate_btc(df, **c) for c in configs]
print(pd.DataFrame(rows).to_string(index=False))
