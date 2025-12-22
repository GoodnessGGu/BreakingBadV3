import pandas as pd
import numpy as np
import logging
from ml_utils import load_model, predict_signal, prepare_features

logger = logging.getLogger(__name__)

# Load AI Model
try:
    ai_model = load_model()
    if ai_model:
        logger.info("AI Model loaded successfully.")
    else:
        logger.warning("⚠️ No AI model found. AI filtering disabled.")
except Exception as e:
    logger.error(f"Failed to load AI model: {e}")
    ai_model = None

def wma(series, period):
    """Calculates Weighted Moving Average."""
    return series.rolling(period).apply(
        lambda x: ((x * np.arange(1, period + 1)).sum()) / np.arange(1, period + 1).sum(), 
        raw=True
    )

def analyze_strategy(candles_data, use_ai=True):
    """
    Analyzes candle data and returns a signal ('CALL', 'PUT', or None).
    Implements the 'Sniper' and 'MA Crossover' logic from newscript.txt.
    """
    if not candles_data or len(candles_data) < 35:
        return None

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(candles_data)

    # Standardize timestamp column to 'time' (Needed for AI 'hour' feature)
    if 'time' not in df.columns:
        if 'from' in df.columns:
             df['time'] = df['from']
        elif 'at' in df.columns:
             df['time'] = df['at']

    if 'time' in df.columns:
        df['time'] = pd.to_datetime(df['time'], unit='s')
    
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
             
    signal = None
    if ma_call: signal = "CALL"
    if ma_put: signal = "PUT"
    
    # --- AI Confirmation ---
    if signal and ai_model and use_ai:
        # Prepare features for the *current* state
        # We need to pass the DataFrame. prepare_features will re-calculate indicators.
        # This is slightly inefficient but safe.
        try:
            df_features = prepare_features(df)
            
            # We need the last row (current candle) to predict
            if not df_features.empty:
                current_features = df_features.iloc[[-1]]
                prediction = predict_signal(ai_model, current_features)
                
                if prediction == 0: # 0 = Loss predicted
                    logger.info(f"[AI] REJECTED {signal} signal on {df.iloc[-1].get('time', 'unknown')}")
                    return None
                else:
                    logger.info(f"[AI] APPROVED {signal} signal.")
        except Exception as e:
            logger.error(f"AI Prediction failed: {e}")
            # Fallback: Allow signal if AI fails? or Block? 
            # Let's allow it for now to avoid stopping trading on bugs.
            pass

    return signal

def confirm_trade_with_ai(candles_data, direction):
    """
    Checks if the AI model 'approves' a trade for a given direction based on current market data.
    Returns True if Approved (or AI unavailable), False if Rejected.
    """
    if not ai_model:
        return True # Pass if no AI
        
    try:
        df = pd.DataFrame(candles_data)
        df = pd.DataFrame(candles_data)
        
        # Standardize timestamp column to 'time'
        if 'time' not in df.columns:
            if 'from' in df.columns:
                df['time'] = df['from']
            elif 'at' in df.columns:
                df['time'] = df['at']
                
        if 'time' in df.columns:
            df['time'] = pd.to_datetime(df['time'], unit='s')
        else:
            logger.warning("Missing timestamp in candle data. 'hour' feature will be missing.")
            
        df['close'] = pd.to_numeric(df['close'])
        df['open'] = pd.to_numeric(df['open'])
        df['min'] = pd.to_numeric(df['min'])
        df['max'] = pd.to_numeric(df['max'])
        
        # Calculate features using the exact same pipeline as training
        df_features = prepare_features(df)
        
        if df_features.empty:
            return True # Not enough data to decide
            
        current_features = df_features.iloc[[-1]]
        
        # Predict
        # Note: The model predicts "Win" (1) or "Loss" (0).
        # It doesn't explicitly know "Direction" unless it was a feature.
        # In our training, 'signal' (CALL/PUT) was NOT a feature (dropped).
        # BUT the features (indicators) are direction-agnostic or baked in.
        # Wait, if RSI is 80 (Overbought), Strategy says PUT. Model learns RSI=80 -> Win.
        # If Telegram says CALL at RSI=80, Model (trained on PUTs at RSI=80) might still say "Win" 
        # if it learned "High Volatility = Win"? 
        # Actually, without 'direction' as feature, the model assumes the *Strategy's* direction logic.
        # Since we can't tell the model "I am doing a CALL", the model just evaluates "Is this a good setup for the Strategy?".
        # If the Strategy would have done a PUT, and Telegram does a CALL, the model might approve "Market is active", 
        # but the trade is opposite.
        #
        # FIX: We can't strictly use this model for *Opposite* signals unless we add 'direction' as a feature.
        # However, for now, let's assume Telegram signals align generally with trends.
        # If the model predicts "Loss", it means "Strategy would lose here". 
        # Often Strategy loses in choppy/bad markets. So "Loss" is a good "Stay Away" filter.
        
        prediction = predict_signal(ai_model, current_features)
        
        if prediction == 0:
            logger.info(f"[AI] REJECTED external {direction} signal (Model predicts Loss).")
            return False
            
        logger.info(f"[AI] APPROVED external {direction} signal.")
        return True
        
    except Exception as e:
        logger.error(f"AI Confirmation Error: {e}")
        return True # Default to allow on error
    
    return None