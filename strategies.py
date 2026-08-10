import pandas as pd
import numpy as np
import logging
from ml_utils import load_model, predict_signal, prepare_features, calculate_rsi, calculate_atr, calculate_orderflow_features, load_loss_detector, predict_loss, calculate_sr_levels, identify_candle_sequence
from nn_utils import predict_nn_trade
from settings import config

logger = logging.getLogger(__name__)

# Load AI Models (win predictors)
ai_models = {"otc": None, "real": None}
# Load Loss Detectors (loss blockers)
loss_detectors = {"otc": None, "real": None}

def load_all_models():
    global ai_models, loss_detectors
    for mtype in ai_models.keys():
        try:
            model = load_model(mtype)
            if model:
                logger.info(f"AI Model ({mtype.upper()}) loaded successfully.")
                ai_models[mtype] = model
            else:
                logger.warning(f"⚠️ No {mtype.upper()} AI model found.")
        except Exception as e:
            logger.error(f"Failed to load {mtype.upper()} model: {e}")
        try:
            ld = load_loss_detector(mtype)
            if ld:
                logger.info(f"Loss Detector ({mtype.upper()}) loaded successfully.")
                loss_detectors[mtype] = ld
            else:
                logger.warning(f"⚠️ No {mtype.upper()} loss detector found. Will train after first run.")
        except Exception as e:
            logger.error(f"Failed to load {mtype.upper()} loss detector: {e}")

load_all_models()

def wma(series, period):
    """Calculates Weighted Moving Average."""
    return series.rolling(period).apply(
        lambda x: ((x * np.arange(1, period + 1)).sum()) / np.arange(1, period + 1).sum(), 
        raw=True
    )

def analyze_strategy(candles_data, use_ai=True, expiry=1, return_features=False, orderbook=None, recent_win_rate=1.0, api=None, ignore_session_filter=False):
    """
    Analyzes candle data and returns a signal ('CALL', 'PUT', or None).
    expiry: Signal duration in minutes (used for NN confirmation)
    return_features: If True, returns (signal, features_dict)
    orderbook: Current Market Depth data for orderflow validation
    recent_win_rate: Current session win rate (0.0 - 1.0) to adjust AI threshold
    api: Optional IQOptionAPI client for counterfactual virtual trade logging
    ignore_session_filter: If True, skips session hour 0/23 risk filtering for testing
    """
    if not candles_data or len(candles_data) < 205: # Need 200 for EMA200
        return (None, None) if return_features else None

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
    
    # --- Strategy 3: 3-Candle S/R ---
    sup, res = calculate_sr_levels(df, lookback=200)
    sequence = identify_candle_sequence(df, count=3)
    
    # Proximity threshold: 0.5 * ATR (current volatility)
    if 'atr' not in df.columns:
        df['atr'] = calculate_atr(df, 14)
    atr_val = df['atr'].iloc[-1]
    prox = atr_val * 0.5
    
    is_near_sup = any(abs(close_curr - s) < prox for s in sup)
    is_near_res = any(abs(close_curr - r) < prox for r in res)
    
    three_candle_call = is_near_sup and sequence == 'BULLISH' and close_curr > ema200_curr
    three_candle_put = is_near_res and sequence == 'BEARISH' and close_curr < ema200_curr

    # --- Signal Aggregation ---
    signal = None
    source = ""
    
    # 3. Multi-Strategy / Confluence Logic
    signals_found = []
    if amiq_call: signals_found.append(("CALL", "AM_IQ"))
    if sniper_call: signals_found.append(("CALL", "Sniper"))
    if three_candle_call: signals_found.append(("CALL", "3CandleSR"))
    if amiq_put: signals_found.append(("PUT", "AM_IQ"))
    if sniper_put: signals_found.append(("PUT", "Sniper"))
    if three_candle_put: signals_found.append(("PUT", "3CandleSR"))
    
    if not signals_found:
        return (None, None) if return_features else None
        
    # Check for contradictions
    directions = set([s[0] for s in signals_found])
    if len(directions) > 1:
        logger.info(f"❌ CONTRADICTION: Both CALL and PUT found. Skipping.")
        return (None, None) if return_features else None
        
    # Standard choice: Use the first one found
    signal, source = signals_found[0]
    initial_signal = signal
    is_confluence = len(signals_found) > 1
    
    if is_confluence:
        source = f"{source}+Confluence"

    rejection_reason = None
        
    # --- PHASE 1 & 3: ADVANCED SAFETY FILTERS ---
    
    # 4. Time-Based Session Filter (Phase 3)
    if not ignore_session_filter:
        from datetime import datetime
        current_hour = datetime.utcnow().hour
        if current_hour in [0, 23]:
             logger.info(f"❌ BLOCKED {source} {signal}: Risky Session (Hour {current_hour})")
             rejection_reason = f"RISKY_SESSION (Hour {current_hour})"
             signal = None

    # 5. ATR Volatility Filter (Phase 3)
    if signal and 'atr' not in df.columns:
        df['atr'] = calculate_atr(df, 14)
    
    if signal:
        atr_curr = df['atr'].iloc[-1]
        atr_avg = df['atr'].rolling(100).mean().iloc[-1]
        
        if not pd.isna(atr_avg):
            if atr_curr > (atr_avg * 3.5):
                logger.info(f"❌ BLOCKED {source} {signal}: Volatility Spike (ATR {atr_curr:.5f} > Avg*3.5)")
                rejection_reason = f"VOLATILITY_SPIKE (ATR {atr_curr:.5f})"
                signal = None
            elif atr_curr < (atr_avg * 0.2):
                logger.info(f"❌ BLOCKED {source} {signal}: Dead Market (ATR {atr_curr:.5f} < Avg*0.2)")
                rejection_reason = f"DEAD_MARKET (ATR {atr_curr:.5f})"
                signal = None

    # --- Orderflow Confirmation (Marginal Data) ---
    if signal and config.use_orderflow_confirmation:
        if not check_orderflow_confirmation(signal, orderbook):
             rejection_reason = "ORDERFLOW_REJECTED"
             signal = None

    # 1. RSI Filter (Phase 1)
    if signal == "CALL" and rsi_curr > 75:
        logger.info(f"❌ BLOCKED {source} CALL: RSI {rsi_curr:.1f} > 75 (Overbought)")
        rejection_reason = f"RSI_OVERBOUGHT ({rsi_curr:.1f})"
        signal = None
    elif signal == "PUT" and rsi_curr < 25:
        logger.info(f"❌ BLOCKED {source} PUT: RSI {rsi_curr:.1f} < 25 (Oversold)")
        rejection_reason = f"RSI_OVERSOLD ({rsi_curr:.1f})"
        signal = None
        
    # 2. EMA200 Trend Filter (Phase 1)
    if signal and config.use_strict_trend:
        if signal == "CALL" and close_curr < ema200_curr:
            logger.info(f"❌ BLOCKED {source} CALL: Price below EMA200 (Downtrend)")
            rejection_reason = "EMA200_DOWNTREND"
            signal = None
        elif signal == "PUT" and close_curr > ema200_curr:
            logger.info(f"❌ BLOCKED {source} PUT: Price above EMA200 (Uptrend)")
            rejection_reason = "EMA200_UPTREND"
            signal = None
            
    # --- AI Confirmation (Double-Gate) ---
    if signal and use_ai:
        try:
            df_features = prepare_features(df)
            if df_features.empty:
                logger.warning("Feature preparation returned empty DataFrame. AI skipped.")
            else:
                current_features_row = df_features.iloc[[-1]]

                asset = candles_data[-1].get('asset', 'unknown').upper() if isinstance(candles_data[-1], dict) else "unknown"
                is_otc = "-OTC" in asset or "OTC" in asset
                mtype = "otc" if is_otc else "real"

                # === GATE 1: Loss Detector ===
                loss_model = loss_detectors.get(mtype)
                if loss_model and signal:
                    loss_prob = predict_loss(loss_model, current_features_row.copy())
                    if loss_prob > 0.65:
                        logger.info(f"[LOSS-GATE] BLOCKED {signal} ({source}): Loss probability {loss_prob:.2f} > 0.65")
                        rejection_reason = f"LOSS_GATE (prob {loss_prob:.2f})"
                        signal = None
                    else:
                        logger.info(f"[LOSS-GATE] PASSED {signal} ({source}): Loss probability {loss_prob:.2f}")

                # === GATE 2: Win Predictor ===
                selected_model = ai_models.get(mtype)
                if selected_model and signal:
                    prediction = predict_signal(selected_model, current_features_row.copy())
                    if prediction == 0:
                        logger.info(f"[WIN-GATE] REJECTED {signal} ({source})")
                        rejection_reason = "WIN_GATE_REJECTED"
                        signal = None
                    else:
                        logger.info(f"[WIN-GATE] APPROVED {signal} ({source})")

                # === GATE 3: Neural Network probability threshold ===
                if signal and config.use_nn_filter:
                    nn_prob = predict_nn_trade(df_features, expiry=expiry)

                    base_threshold = config.nn_threshold
                    adaptive_penalty = max(0, 0.5 - recent_win_rate) * 0.4
                    threshold = base_threshold + adaptive_penalty

                    atr_curr = df_features['atr'].iloc[-1]
                    atr_avg = df_features['atr'].rolling(20).mean().iloc[-1]
                    market_erratic = atr_curr > (atr_avg * 1.8)

                    if market_erratic:
                        logger.info(f"⚠️ SENSOR SUPPRESSION: Erratic Volatility (ATR {atr_curr:.5f} > 1.8x avg). Signal {signal} blocked.")
                        rejection_reason = "ERRATIC_VOLATILITY"
                        signal = None
                    elif nn_prob < threshold:
                        logger.info(f"[NN-GATE] REJECTED {signal} ({source}): Prob {nn_prob:.2f} < {threshold:.2f}")
                        rejection_reason = f"NN_GATE (prob {nn_prob:.2f} < {threshold:.2f})"
                        signal = None
                    else:
                        logger.info(f"[NN-GATE] APPROVED {signal} ({source}): Prob {nn_prob:.2f} >= {threshold:.2f}")

        except Exception as e:
            logger.error(f"AI Confirmation Error: {e}")

    # Generate features dict if requested or for virtual trade counterfactual logging
    final_features = None
    try:
        full_df = prepare_features(df.copy())
        if not full_df.empty:
            last_row = full_df.iloc[-1].to_dict()
            final_features = {k: v for k, v in last_row.items() if isinstance(v, (int, float, np.number))}
            if 'nn_prob' not in final_features:
                final_features['nn_prob'] = predict_nn_trade(full_df, expiry=expiry)
    except Exception as e:
        logger.error(f"Error extracting features for strategy: {e}")

    # --- COUNTERFACTUAL VIRTUAL TRADE LOGGING ---
    if rejection_reason and initial_signal and config.enable_counterfactual_learning and api:
        try:
            import asyncio
            from trade_database import db
            from trade import evaluate_virtual_trade_counterfactual
            asset = candles_data[-1].get('asset', 'UNKNOWN').upper() if isinstance(candles_data[-1], dict) else "UNKNOWN"
            entry_px = float(close_curr)
            vid = db.save_virtual_trade({
                'asset': asset,
                'direction': initial_signal,
                'expiry': expiry,
                'rejection_reason': rejection_reason,
                'entry_price': entry_px,
                'features': final_features or {}
            })
            if vid:
                asyncio.create_task(evaluate_virtual_trade_counterfactual(api, vid, asset, initial_signal, expiry, entry_px, final_features or {}))
        except Exception as e:
            logger.error(f"Failed to log counterfactual virtual trade in strategy: {e}")

    if return_features:
        return signal, final_features

    return signal

def check_orderflow_confirmation(signal, orderbook):
    """
    Validates a signal using real-time orderflow imbalance from Marginal-CFD data.
    """
    if not orderbook:
        # Fail open: if no data, don't block.
        return True
        
    of = calculate_orderflow_features(orderbook)
    
    # CALL Confirmation: Positive delta and Imbalance > 0.8
    if signal == "CALL":
        if of['of_imbalance'] < 0.8 or of['of_delta'] < 0:
            logger.info(f"📊 [Orderflow] REJECTED CALL | Imbal: {of['of_imbalance']:.2f} | Delta: {of['of_delta']:.0f}")
            return False
            
    # PUT Confirmation: Negative delta and Imbalance < 1.2
    if signal == "PUT":
        if of['of_imbalance'] > 1.2 or of['of_delta'] > 0:
            logger.info(f"📊 [Orderflow] REJECTED PUT | Imbal: {of['of_imbalance']:.2f} | Delta: {of['of_delta']:.0f}")
            return False
            
    logger.info(f"📊 [Orderflow] APPROVED {signal} | Imbal: {of['of_imbalance']:.2f}")
    return True
