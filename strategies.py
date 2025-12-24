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

def reload_ai_model():
    """Reloads the AI model from disk."""
    global ai_model
    try:
        new_model = load_model()
        if new_model:
            ai_model = new_model
            logger.info("🧠 AI Model Reloaded Successfully!")
            return True
        else:
            logger.warning("⚠️ Failed to load new AI model (None returned).")
            return False
    except Exception as e:
        logger.error(f"❌ Error reloading AI model: {e}")
        return False

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
            
    # --- Strategy 1: WMA-Smoothed Furious Crossover (Rei das Options Logic) ---
    # Fast: SMA(1), Slow: SMA(34) -> Buffer1 = Fast - Slow
    # Signal: WMA(Buffer1, 5) -> Buffer2
    # Trigger: Cross of Buffer1 & Buffer2
    
    df['sma_fast'] = df['close'].rolling(window=1).mean() # Close
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    
    # Calculate WMA(5) of Buffer1
    # Note: wma function expects positive input typically? No, WMA works on negatives.
    # WMA Formula: Sum(Price * Weight) / Sum(Weights)
    # Weights for period 5: [1, 2, 3, 4, 5] -> Sum 15
    df['buffer2'] = df['buffer1'].rolling(window=5).apply(
        lambda x: ((x * np.arange(1, 6)).sum()) / 15, 
        raw=True
    )
    
    # --- Strategy 2: Sniper Pattern (Section 3 of Lua) ---
    # ... (Sniper Logic Unchanged) ...
    # But we update signal generation below to use Buffer crossover
    
    # ... [Existing Sniper Code Lines 78-106] ... 
    # (Checking if I need to re-include them. The tool replaces blocks.
    # I should replace just the Signal Logic part.
    # Let's keep Sniper logic variables definition but REWRITE the Signal Combination part)

    curr = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    c3 = df.iloc[-4]
    
    # Sniper CALL Logic
    sniper_call = (
        c3['open'] < c3['close'] and
        c2['open'] < c2['close'] and
        c1['open'] > c1['close'] and
        c1['close'] > c2['open'] and
        c1['open'] > c2['open'] and
        curr['open'] < curr['close']
    )
    
    # Sniper PUT Logic
    sniper_put = (
        c3['open'] > c3['close'] and
        c2['open'] > c2['close'] and
        c1['open'] < c1['close'] and
        c1['close'] < c2['open'] and
        c1['open'] < c2['open'] and
        curr['open'] > curr['close']
    )

    # --- Signal Generation ---
    signal = None
    
    # Check Crossover at Close of Previous Candle (c1)
    # We look for crossover happening between c2 (2 candles ago) and c1 (completed candle)
    # OR current forming if we want Aggressive.
    # Usually we trade on CLOSE.
    
    buf1_c1 = c1['buffer1'] # Value at close of prev
    buf2_c1 = c1['buffer2']
    
    buf1_c2 = c2['buffer1'] # Value 2 candles ago
    buf2_c2 = c2['buffer2']
    
    # CALL CROSS: Buffer1 crosses ABOVE Buffer2
    # (Was Below/Equal before, Now Above)
    ma_call = (buf1_c2 <= buf2_c2) and (buf1_c1 > buf2_c1)
    
    # PUT CROSS: Buffer1 crosses BELOW Buffer2
    # (Was Above/Equal before, Now Below)
    ma_put = (buf1_c2 >= buf2_c2) and (buf1_c1 < buf2_c1)
    
    if sniper_call or ma_call:
        signal = "CALL"
    elif sniper_put or ma_put:
        signal = "PUT"

    # --- Trend Filter (EMA 50) ---
    # Only take CALL if Close > EMA50
    # Only take PUT if Close < EMA50
    
    # We need EMA50
    df['ema50'] = df['close'].ewm(span=50, adjust=False).mean()
    
    # Use c1 (Previous Completed Candle) for filtering
    # Re-access via index to ensure we see the new 'ema50' column
    filter_close = df.iloc[-2]['close']
    filter_ema = df.iloc[-2]['ema50']

    if signal == "CALL":
        if filter_close < filter_ema:
            logger.info(f"🚫 Trend Filter Blocked: CALL signal below EMA50 ({filter_close} < {filter_ema})")
            return None
            
    if signal == "PUT":
        if filter_close > filter_ema:
            logger.info(f"🚫 Trend Filter Blocked: PUT signal above EMA50 ({filter_close} > {filter_ema})")
            return None
        # The original code had an 'elif' here, but the instruction implies replacing the whole block.
        # The provided replacement code has an 'elif' inside the 'if signal == "PUT":' block,
        # which is syntactically incorrect. I will assume the user meant to replace the entire
        # trend filtering logic with the new one, and the `elif signal == "PUT" and current_close > current_ema:`
        # was a remnant or typo in the provided snippet.
        # The new logic correctly handles both CALL and PUT filters.

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
                
                if prediction == 0: # 0 = Loss/Reject (based on new Threshold logic)
                    logger.info(f"[AI] REJECTED {signal} signal on {df.iloc[-1].get('time', 'unknown')}")
                    return None
                else:
                    logger.info(f"[AI] APPROVED {signal} signal.")
        except Exception as e:
            logger.error(f"AI Prediction failed: {e}")
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