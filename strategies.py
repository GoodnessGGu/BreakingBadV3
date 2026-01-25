import pandas as pd
import numpy as np
import logging
from ml_utils import load_model, predict_signal, prepare_features, calculate_rsi, calculate_atr

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
    """
    if not candles_data or len(candles_data) < 205: # Need 200 for EMA200
        return None

    # Convert list of dicts to DataFrame
    df = pd.DataFrame(candles_data)
    
    # Ensure numeric columns
    cols = ['open', 'close', 'min', 'max']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    # --- Indicators for Filters ---
    if 'rsi' not in df.columns:
        df['rsi'] = calculate_rsi(df['close'], 14)
    if 'ema_200' not in df.columns:
        df['ema_200'] = df['close'].ewm(span=200, adjust=False).mean()
        
    rsi_curr = df['rsi'].iloc[-1]
    ema200_curr = df['ema_200'].iloc[-1]
    close_curr = df['close'].iloc[-1]

    # --- Strategy 1: MA Crossover (AM_IQ) ---
    df['sma_fast'] = df['close'].rolling(window=1).mean() 
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = wma(df['buffer1'], 4)
    
    amiq_call = df['buffer1'].iloc[-1] > df['buffer2'].iloc[-1] and \
                df['buffer1'].iloc[-2] < df['buffer2'].iloc[-2]
              
    amiq_put = df['buffer1'].iloc[-1] < df['buffer2'].iloc[-1] and \
               df['buffer1'].iloc[-2] > df['buffer2'].iloc[-2]

    # --- Strategy 2: Sniper Pattern ---
    curr = df.iloc[-1]
    c1 = df.iloc[-2]
    c2 = df.iloc[-3]
    c3 = df.iloc[-4]
    
    sniper_call = (
        c3['open'] < c3['close'] and c2['open'] < c2['close'] and
        c1['open'] > c1['close'] and c1['close'] > c2['open'] and
        c1['open'] > c2['open'] and curr['open'] < curr['close']
    )
    
    sniper_put = (
        c3['open'] > c3['close'] and c2['open'] > c2['close'] and
        c1['open'] < c1['close'] and c1['close'] < c2['open'] and
        c1['open'] < c2['open'] and curr['open'] > curr['close']
    )
    
    # --- Signal Aggregation ---
    signal = None
    source = ""
    
    # 3. Multi-Strategy / Confluence Logic
    # We track how many signals we have
    signals_found = []
    if amiq_call: signals_found.append(("CALL", "AM_IQ"))
    if sniper_call: signals_found.append(("CALL", "Sniper"))
    if amiq_put: signals_found.append(("PUT", "AM_IQ"))
    if sniper_put: signals_found.append(("PUT", "Sniper"))
    
    if not signals_found:
        return None
        
    # Check for contradictions
    directions = set([s[0] for s in signals_found])
    if len(directions) > 1:
        logger.info(f"❌ CONTRADICTION: Both CALL and PUT found. Skipping.")
        return None
        
    # Standard choice: Use the first one found, but track count
    signal, source = signals_found[0]
    is_confluence = len(signals_found) > 1
    
    if is_confluence:
        source = f"{source}+Confluence"
        
    # --- PHASE 1 & 3: ADVANCED SAFETY FILTERS ---
    
    # 4. Time-Based Session Filter (Phase 3)
    from datetime import datetime
    current_hour = datetime.utcnow().hour
    # Avoid: Extremes only. Hour 0 (OTC spread) and 23 (market close)
    if current_hour in [0, 23]:
         logger.info(f"❌ BLOCKED {source} {signal}: Risky Session (Hour {current_hour})")
         return None

    # 5. ATR Volatility Filter (Phase 3)
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df, 14)
    
    atr_curr = df['atr'].iloc[-1]
    atr_avg = df['atr'].rolling(100).mean().iloc[-1]
    
    if not pd.isna(atr_avg):
        # Allow more volatility: spike threshold 3.5x, dead market 0.2x
        if atr_curr > (atr_avg * 3.5):
            logger.info(f"❌ BLOCKED {source} {signal}: Volatility Spike (ATR {atr_curr:.5f} > Avg*3.5)")
            return None
        if atr_curr < (atr_avg * 0.2):
            logger.info(f"❌ BLOCKED {source} {signal}: Dead Market (ATR {atr_curr:.5f} < Avg*0.2)")
            return None

    # 1. RSI Filter (Phase 1)
    if signal == "CALL" and rsi_curr > 70:
        logger.info(f"❌ BLOCKED {source} CALL: RSI {rsi_curr:.1f} > 70 (Overbought)")
        return None
    elif signal == "PUT" and rsi_curr < 30:
        logger.info(f"❌ BLOCKED {source} PUT: RSI {rsi_curr:.1f} < 30 (Oversold)")
        return None
        
    # 2. EMA200 Trend Filter (Phase 1)
    if signal == "CALL" and close_curr < ema200_curr:
        logger.info(f"❌ BLOCKED {source} CALL: Price below EMA200 (Downtrend)")
        return None
    elif signal == "PUT" and close_curr > ema200_curr:
        logger.info(f"❌ BLOCKED {source} PUT: Price above EMA200 (Uptrend)")
        return None
            
    # --- AI Confirmation ---
    if signals_found and ai_model and use_ai:
        try:
            df_features = prepare_features(df)
            if not df_features.empty:
                current_features = df_features.iloc[[-1]]
                prediction = predict_signal(ai_model, current_features)
                
                if prediction == 0:
                    logger.info(f"[AI] REJECTED {signal} ({source})")
                    return None
                else:
                    logger.info(f"[AI] APPROVED {signal} ({source})")
        except Exception as e:
            logger.error(f"AI Prediction failed: {e}")
            pass

    return signal
