import pandas as pd
import numpy as np
import joblib
import os
import logging
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS_DIR = "models"
MODEL_PATH = os.path.join(MODELS_DIR, "trade_model.pkl")

# --- Indicators ---
def calculate_rsi(series, period=14):
    """Calculates Relative Strength Index (RSI)."""
    delta = series.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

def calculate_bollinger_bands(series, period=20, std_dev=2):
    """Calculates Bollinger Bands."""
    sma = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = sma + (std * std_dev)
    lower = sma - (std * std_dev)
    return upper, lower

def calculate_adx(df, period=14):
    """Calculates ADX."""
    alpha = 1/period
    # True Range
    h_l = df['max'] - df['min']
    h_yc = abs(df['max'] - df['close'].shift(1))
    l_yc = abs(df['min'] - df['close'].shift(1))
    tr = pd.concat([h_l, h_yc, l_yc], axis=1).max(axis=1)
    
    # Directional Movement
    up = df['max'] - df['max'].shift(1)
    down = df['min'].shift(1) - df['min']
    plus_dm = np.where((up > down) & (up > 0), up, 0)
    minus_dm = np.where((down > up) & (down > 0), down, 0)
    
    # Smoothing
    tr_s = tr.ewm(alpha=alpha, adjust=False).mean()
    plus_dm_s = pd.Series(plus_dm).ewm(alpha=alpha, adjust=False).mean()
    minus_dm_s = pd.Series(minus_dm).ewm(alpha=alpha, adjust=False).mean()
    
    dx = 100 * abs(plus_dm_s - minus_dm_s) / (plus_dm_s + minus_dm_s)
    return dx.ewm(alpha=alpha, adjust=False).mean()

def calculate_atr(df, period=14):
    """Calculates Average True Range (Volatility)."""
    high = df['max']
    low = df['min']
    close = df['close']
    
    tr1 = high - low
    tr2 = abs(high - close.shift(1))
    tr3 = abs(low - close.shift(1))
    
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    atr = tr.rolling(window=period).mean()
    return atr

def calculate_macd(series, fast=12, slow=26, signal=9):
    """Calculates MACD (Moving Average Convergence Divergence)."""
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    histogram = macd - signal_line
    return macd, signal_line, histogram

def calculate_stochastic(df, period=14, k_period=3, d_period=3):
    """Calculates Stochastic Oscillator."""
    low_min = df['min'].rolling(window=period).min()
    high_max = df['max'].rolling(window=period).max()
    
    k = 100 * ((df['close'] - low_min) / (high_max - low_min))
    # Fast Stochastic %K
    percent_k = k
    # Smooth %D
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d

def calculate_cci(df, period=20):
    """Calculates Commodity Channel Index (CCI)."""
    tp = (df['max'] + df['min'] + df['close']) / 3
    sma = tp.rolling(window=period).mean()
    mad = (tp - sma).abs().rolling(window=period).mean()
    
    # Avoid division by zero
    mad = mad.replace(0, 0.001)
    
    cci = (tp - sma) / (0.015 * mad)
    return cci

def prepare_features(df):
    """
    Generates technical indicators as features for the ML model.
    """
    df = df.copy()
    
    # Ensure numeric
    cols = ['open', 'close', 'min', 'max', 'volume']
    for c in cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c])
            
    if 'time' in df.columns:
        df['hour'] = df['time'].dt.hour
        # Encoding cyclical time features can be better, but raw hour is a good start
    
    # 1. Momentum & Trend
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['atr'] = calculate_atr(df, 14)
    
    # 2. Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    
    # 3. Bollinger Bands
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # 4. Price Action
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['max'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['min']

    # 5. Advanced Indicators [NEW]
    df['macd'], df['macd_signal'], df['macd_hist'] = calculate_macd(df['close'])
    df['stoch_k'], df['stoch_d'] = calculate_stochastic(df)
    df['cci'] = calculate_cci(df)
    
    # Pattern: Engulfing (1 = Bullish, -1 = Bearish, 0 = None)
    # Bullish Engulfing: Prev Red, Curr Green, Curr Open < Prev Close, Curr Close > Prev Open
    # Bearish Engulfing: Prev Green, Curr Red, Curr Open > Prev Close, Curr Close < Prev Open
    
    prev_open = df['open'].shift(1)
    prev_close = df['close'].shift(1)
    curr_open = df['open']
    curr_close = df['close']
    
    # Vectorized Engulfing Logic
    is_bullish_engulfing = (prev_close < prev_open) & (curr_close > curr_open) & \
                           (curr_open < prev_close) & (curr_close > prev_open)
                           
    is_bearish_engulfing = (prev_close > prev_open) & (curr_close < curr_open) & \
                           (curr_open > prev_close) & (curr_close < prev_open)
                           
    df['pattern_engulfing'] = 0
    df.loc[is_bullish_engulfing, 'pattern_engulfing'] = 1
    df.loc[is_bearish_engulfing, 'pattern_engulfing'] = -1

    # 5. Lagged Features (Previous candles)
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
    
    # Drop rows with NaN (due to rolling windows)
    df = df.dropna()
    return df

def train_model(data_path="training_data.csv"):
    """
    Trains a Gradient Boosting model using labeled data.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    
    # Separate features (X) and target (y)
    if 'outcome' not in df.columns:
        logger.error("Data missing 'outcome' column.")
        return

    # Drop non-feature columns
    drop_cols = ['time', 'outcome', 'signal', 'asset', 'from', 'to']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['outcome']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    logger.info("Training Random Forest Classifier...")
    clf = RandomForestClassifier(
        n_estimators=200, 
        max_depth=10,            # Deeper trees to capture patterns
        min_samples_split=10, 
        min_samples_leaf=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)
    
    # Evaluate
    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Model Accuracy: {acc:.2f}")
    logger.info("\n" + classification_report(y_test, y_pred))
    
    # Save
    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)
        
    joblib.dump(clf, MODEL_PATH)
    logger.info(f"Model saved to {MODEL_PATH}")
    return clf

def load_model():
    """Loads the trained model."""
    if os.path.exists(MODEL_PATH):
        try:
            return joblib.load(MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
    return None

def predict_signal(model, features_df):
    """
    Predicts outcome for a single row of features.
    Returns: 1 (Win) or 0 (Loss)
    """
    if model is None:
        return 1 # Fallback: Assume win if no model
        
    try:
        # Align features with model
        if hasattr(model, "feature_names_in_"):
            # Only keep columns that the model knows
            # Add missing columns as 0 (if any, though unlikely if prepare_features is consistent)
            valid_cols = [c for c in model.feature_names_in_ if c in features_df.columns]
            
            if len(valid_cols) < len(model.feature_names_in_):
                missing = set(model.feature_names_in_) - set(features_df.columns)
                logger.warning(f"Missing features for prediction: {missing}")
                # Optional: Add missing as 0 or fail. For now, let's try to proceed with what we have if possible, 
                # but sklearn usually strictly requires all features in order.
                for c in missing:
                    features_df[c] = 0
            
            # Reorder columns to match model
            features_df = features_df[model.feature_names_in_]
            
            features_df = features_df[model.feature_names_in_]
            
        # prediction = model.predict(features_df)
        # return prediction[0]
        
        # Use Probability instead of hard prediction
        proba = model.predict_proba(features_df)
        # proba returns [[prob_0, prob_1]]
        
        loss_prob = proba[0][0]
        win_prob = proba[0][1]
        
        # Confidence Threshold: Only trade if model is > 65% sure it's a WIN
        threshold = 0.65 
        
        if win_prob >= threshold:
            return 1
        else:
            return 0 # Treat as Loss (Reject)
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 1 # Fallback
