import pandas as pd
import numpy as np

def wma(series, period):
    """Calculates Weighted Moving Average."""
    return series.rolling(period).apply(
        lambda x: ((x * np.arange(1, period + 1)).sum()) / np.arange(1, period + 1).sum(), 
        raw=True
    )

def analyze_strategy(candles_data):
    """
    Analyzes candle data and returns a signal ('CALL', 'PUT', or None).
    Implements the 'Sniper' and 'MA Crossover' logic from newscript.txt.
    """
    if not candles_data or len(candles_data) < 35:
        return None

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(candles_data)
    
    # Ensure numeric columns
    cols = ['open', 'close', 'min', 'max']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    # --- Strategy 1: MA Crossover (Section 1 of Lua) ---
    # Lua: smaFast(1), smaSlow(34), buffer1 = Fast - Slow, buffer2 = WMA(buffer1, 4)
    df['sma_fast'] = df['close'].rolling(window=1).mean() # Effectively just close price
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = wma(df['buffer1'], 4)
    
    # --- Strategy 2: Sniper Pattern (Section 3 of Lua) ---
    # We need the last 4 candles: 
    # [3] (Oldest), [2], [1] (Previous), [0] (Current/Forming)
    
    curr = df.iloc[-1] # Current forming candle
    c1 = df.iloc[-2]   # Previous completed
    c2 = df.iloc[-3]
    c3 = df.iloc[-4]
    
    # Sniper CALL Logic
    sniper_call = (
        c3['open'] < c3['close'] and       # Candle 3: Green
        c2['open'] < c2['close'] and       # Candle 2: Green
        c1['open'] > c1['close'] and       # Candle 1: Red (Pullback)
        c1['close'] > c2['open'] and       # Pullback didn't break trend start
        c1['open'] > c2['open'] and
        curr['open'] < curr['close']       # Current: Green (Resumption)
    )
    
    # Sniper PUT Logic
    sniper_put = (
        c3['open'] > c3['close'] and       # Candle 3: Red
        c2['open'] > c2['close'] and       # Candle 2: Red
        c1['open'] < c1['close'] and       # Candle 1: Green (Pullback)
        c1['close'] < c2['open'] and       # Pullback didn't break trend start
        c1['open'] < c2['open'] and
        curr['open'] > curr['close']       # Current: Red (Resumption)
    )
    
    if sniper_call:
        return "CALL"
    elif sniper_put:
        return "PUT"
        
    # --- Strategy 3: MA Crossover (Fallback) ---
    # Check if Buffer1 (Fast-Slow) crosses Buffer2 (Signal)
    # Current candle ([-1]) vs Previous candle ([-2])
    
    ma_call = df['buffer1'].iloc[-1] > df['buffer2'].iloc[-1] and \
              df['buffer1'].iloc[-2] < df['buffer2'].iloc[-2]
              
    ma_put = df['buffer1'].iloc[-1] < df['buffer2'].iloc[-1] and \
             df['buffer1'].iloc[-2] > df['buffer2'].iloc[-2]
             
    if ma_call: return "CALL"
    if ma_put: return "PUT"
    
    return None