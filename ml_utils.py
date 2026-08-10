import warnings
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn.utils.parallel")
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
MODEL_PATH_TEMPLATE = os.path.join(MODELS_DIR, "trade_model_{}.pkl")
# Legacy path for migration/fallback
LEGACY_MODEL_PATH = os.path.join(MODELS_DIR, "trade_model.pkl")

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
    return atr

def calculate_sr_levels(df, lookback=200):
    """
    Identifies Support and Resistance levels using Fractal Highs/Lows.
    Returns (support_levels, resistance_levels)
    """
    if len(df) < lookback:
        return [], []
    
    # Use only last 'lookback' candles
    data = df.iloc[-lookback:].copy()
    
    # Identify Fractals (neighboring context)
    # A local high is higher than 2 candles before and 2 after
    data['is_resistance'] = (data['max'] > data['max'].shift(1)) & \
                             (data['max'] > data['max'].shift(2)) & \
                             (data['max'] > data['max'].shift(-1)) & \
                             (data['max'] > data['max'].shift(-2))
    
    data['is_support'] = (data['min'] < data['min'].shift(1)) & \
                          (data['min'] < data['min'].shift(2)) & \
                          (data['min'] < data['min'].shift(-1)) & \
                          (data['min'] < data['min'].shift(-2))
    
    res_levels = data[data['is_resistance']]['max'].tolist()
    sup_levels = data[data['is_support']]['min'].tolist()
    
    return sorted(sup_levels), sorted(res_levels)

def identify_candle_sequence(df, count=3):
    """
    Analyzes the last N candles to determine if they form a consistent group.
    Returns: 'BULLISH' (all green), 'BEARISH' (all red), or 'MIXED'
    """
    if len(df) < count:
        return 'MIXED'
    
    last_n = df.iloc[-count:]
    is_green = (last_n['close'] > last_n['open']).all()
    is_red = (last_n['close'] < last_n['open']).all()
    
    if is_green: return 'BULLISH'
    if is_red: return 'BEARISH'
    return 'MIXED'

# --- Orderflow & Volume Analysis ---
def calculate_orderflow_features(orderbook_data, quotes_data=None):
    """
    Extracts high-probability orderflow features from market depth and tick data.
    """
    metrics = {
        'of_delta': 0.0,
        'of_imbalance': 0.0,
        'of_spread': 0.0,
        'of_pressure': 0.0 # Ratio of top levels volume
    }
    
    if not orderbook_data:
        return metrics

    bids = orderbook_data.get('bid', [])
    asks = orderbook_data.get('ask', [])
    
    if not bids or not asks:
        return metrics

    # 1. Total Liquidity Depth
    total_bid_vol = sum(b.get('volume', 0) for b in bids)
    total_ask_vol = sum(a.get('volume', 0) for a in asks)
    
    metrics['of_delta'] = total_bid_vol - total_ask_vol
    metrics['of_imbalance'] = (total_bid_vol + 1e-9) / (total_ask_vol + 1e-9)
    
    # 2. Top-of-Book Spread
    metrics['of_spread'] = asks[0]['price'] - bids[0]['price']
    
    # 3. Aggressive Pressure (Top 3 levels)
    top_bid_vol = sum(b.get('volume', 0) for b in bids[:3])
    top_ask_vol = sum(a.get('volume', 0) for a in asks[:3])
    metrics['of_pressure'] = (top_bid_vol + 1e-9) / (top_ask_vol + 1e-9)
    
    return metrics

def calculate_vwap(df):
    """Calculates Volume Weighted Average Price. Falls back to typical price when volume is 0 (e.g., OTC pairs)."""
    v = df['volume']
    p = (df['max'] + df['min'] + df['close']) / 3
    cum_vol = v.cumsum()
    # Avoid NaN on zero-volume pairs (OTC): fall back to typical price
    vwap = (p * v).cumsum() / cum_vol.where(cum_vol > 0, other=pd.NA)
    return vwap.fillna(p)

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
            
    if 'time' not in df.columns and 'from' in df.columns:
        df['time'] = pd.to_datetime(df['from'], unit='s')
        
    if 'time' in df.columns:
        if not pd.api.types.is_datetime64_any_dtype(df['time']):
             df['time'] = pd.to_datetime(df['time'])
        df['hour'] = df['time'].dt.hour
        df['minute'] = df['time'].dt.minute
        
        # --- Cyclical Time Features ---
        df['hour_sin'] = np.sin(2 * np.pi * df['hour'] / 24.0)
        df['hour_cos'] = np.cos(2 * np.pi * df['hour'] / 24.0)
        df['min_sin'] = np.sin(2 * np.pi * df['minute'] / 60.0)
        df['min_cos'] = np.cos(2 * np.pi * df['minute'] / 60.0)
    
    # 1. Momentum & Trend
    df['rsi'] = calculate_rsi(df['close'], 14)
    df['adx'] = calculate_adx(df, 14)
    df['atr'] = calculate_atr(df, 14)
    
    # Stochastic Oscillator (%K and %D)
    period = 14
    smooth_k = 3
    smooth_d = 3
    
    # Calculate %K: (Current Close - Lowest Low) / (Highest High - Lowest Low) * 100
    low_min = df['min'].rolling(window=period).min()
    high_max = df['max'].rolling(window=period).max()
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min + 1e-9))
    
    # Smooth %K to get %K line
    df['stoch_k'] = df['stoch_k'].rolling(window=smooth_k).mean()
    
    # Calculate %D: Moving average of %K
    df['stoch_d'] = df['stoch_k'].rolling(window=smooth_d).mean()
    
    # Stochastic signals: Overbought (>80), Oversold (<20)
    df['stoch_oversold'] = (df['stoch_k'] < 20).astype(int)
    df['stoch_overbought'] = (df['stoch_k'] > 80).astype(int)
    
    # Stochastic crossover: %K crosses above %D (bullish), %K crosses below %D (bearish)
    df['stoch_cross_up'] = ((df['stoch_k'] > df['stoch_d']) & (df['stoch_k'].shift(1) <= df['stoch_d'].shift(1))).astype(int)
    df['stoch_cross_down'] = ((df['stoch_k'] < df['stoch_d']) & (df['stoch_k'].shift(1) >= df['stoch_d'].shift(1))).astype(int)
    
    # NEW: Rate of Change (Momentum)
    df['roc_5'] = df['close'].pct_change(5)
    df['roc_10'] = df['close'].pct_change(10)
    
    # 2. Moving Averages
    df['sma_20'] = df['close'].rolling(window=20).mean()
    df['sma_50'] = df['close'].rolling(window=50).mean()
    df['ema_100'] = df['close'].ewm(span=100, adjust=False).mean()
    df['vwap'] = calculate_vwap(df)
    
    # NEW: Price Distance from MAs/VWAP
    df['dist_sma20'] = (df['close'] - df['sma_20']) / df['close']
    df['dist_sma50'] = (df['close'] - df['sma_50']) / df['close'] 
    df['dist_vwap'] = (df['close'] - df['vwap']) / df['close']
    
    # 3. Bollinger Bands
    df['bb_upper'], df['bb_lower'] = calculate_bollinger_bands(df['close'], 20, 2)
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['close']
    df['bb_pos'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'])
    
    # NEW: Volume Indicators
    if 'volume' in df.columns:
        df['vol_sma20'] = df['volume'].rolling(20).mean()
        df['vol_ratio'] = df['volume'] / (df['vol_sma20'] + 1e-9)
        
    # NEW: Pivot Points (Classic)
    # Using rolling window to estimate daily high/low roughly or just local swing
    # For 1m candles, we can use local pivots (e.g. 60 min)
    # Better to use recent 50-candle High/Low/Close for support/resistance context
    df['recent_high'] = df['max'].rolling(50).max()
    df['recent_low'] = df['min'].rolling(50).min()
    df['pivot'] = (df['recent_high'] + df['recent_low'] + df['close']) / 3
    df['res1'] = (2 * df['pivot']) - df['recent_low']
    df['sup1'] = (2 * df['pivot']) - df['recent_high']
    
    # 4. Price Action
    df['body_size'] = abs(df['close'] - df['open'])
    df['upper_shadow'] = df['max'] - df[['open', 'close']].max(axis=1)
    df['lower_shadow'] = df[['open', 'close']].min(axis=1) - df['min']
    
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

    # USER OBSERVATION: Oversized Engulfing Reversal Pattern
    # When engulfing body is 2-3x+ larger than previous, expect reversal
    prev_body_size = abs(prev_close - prev_open)
    curr_body_size = abs(curr_close - curr_open)
    body_ratio = curr_body_size / (prev_body_size + 1e-9)
    
    # Oversized bullish engulfing (likely to reverse down)
    df['pattern_exhaustion_bull'] = ((is_bullish_engulfing) & (body_ratio >= 2.0)).astype(int)
    # Oversized bearish engulfing (likely to reverse up)
    df['pattern_exhaustion_bear'] = ((is_bearish_engulfing) & (body_ratio >= 2.0)).astype(int)
    
    # Doji Pattern (open ≈ close, indecision)
    # Body is very small relative to total range
    total_range = df['max'] - df['min']
    df['pattern_doji'] = (df['body_size'] / (total_range + 1e-9) < 0.1).astype(int)
    
    # Hammer Pattern (bullish reversal)
    # Small body at top, long lower shadow (2x+ body), little/no upper shadow
    is_hammer = (
        (df['lower_shadow'] > 2 * df['body_size']) &
        (df['upper_shadow'] < 0.3 * df['body_size']) &
        (df['body_size'] > 0)  # Not a doji
    )
    df['pattern_hammer'] = is_hammer.astype(int)
    
    # Shooting Star (bearish reversal)
    # Small body at bottom, long upper shadow (2x+ body), little/no lower shadow
    is_shooting_star = (
        (df['upper_shadow'] > 2 * df['body_size']) &
        (df['lower_shadow'] < 0.3 * df['body_size']) &
        (df['body_size'] > 0)
    )
    df['pattern_shooting_star'] = is_shooting_star.astype(int)
    
    # Pin Bar (strong reversal signal)
    # Very long wick on one side (3x+ body), opposite wick is small
    is_bullish_pin = (
        (df['lower_shadow'] > 3 * df['body_size']) &
        (df['upper_shadow'] < df['body_size'])
    )
    is_bearish_pin = (
        (df['upper_shadow'] > 3 * df['body_size']) &
        (df['lower_shadow'] < df['body_size'])
    )
    df['pattern_pin_bar'] = 0
    df.loc[is_bullish_pin, 'pattern_pin_bar'] = 1
    df.loc[is_bearish_pin, 'pattern_pin_bar'] = -1
    
    # Inside Bar (consolidation, breakout setup)
    # Current high < prev high AND current low > prev low
    prev_high = df['max'].shift(1)
    prev_low = df['min'].shift(1)
    df['pattern_inside_bar'] = ((df['max'] < prev_high) & (df['min'] > prev_low)).astype(int)
    
    # Strong Momentum Candle (large body relative to recent average)
    avg_body_20 = df['body_size'].rolling(20).mean()
    df['pattern_strong_momentum'] = (df['body_size'] > 2 * avg_body_20).astype(int)

    # NEW: Signal Indicators as Features
    # AM_IQ logic
    df['sma_fast'] = df['close'].rolling(window=1).mean()
    df['sma_slow'] = df['close'].rolling(window=34).mean()
    df['buffer1'] = df['sma_fast'] - df['sma_slow']
    df['buffer2'] = df['buffer1'].rolling(4).apply(
        lambda x: ((x * np.arange(1, 5)).sum()) / np.arange(1, 5).sum(), 
        raw=True
    )
    df['amiq_call'] = ((df['buffer1'] > df['buffer2']) & (df['buffer1'].shift(1) < df['buffer2'].shift(1))).astype(int)
    df['amiq_put'] = ((df['buffer1'] < df['buffer2']) & (df['buffer1'].shift(1) > df['buffer2'].shift(1))).astype(int)
    
    # Sniper logic
    prev1_o = df['open'].shift(1)
    prev1_c = df['close'].shift(1)
    prev2_o = df['open'].shift(2)
    prev2_c = df['close'].shift(2)
    prev3_o = df['open'].shift(3)
    prev3_c = df['close'].shift(3)
    curr_o = df['open']
    curr_c = df['close']
    
    df['sniper_call'] = (
        (prev3_c > prev3_o) & (prev2_c > prev2_o) &
        (prev1_c < prev1_o) & (prev1_c > prev2_o) &
        (prev1_o > prev2_o) & (curr_c > curr_o)
    ).astype(int)
    
    df['sniper_put'] = (
        (prev3_c < prev3_o) & (prev2_c < prev2_o) &
        (prev1_c > prev1_o) & (prev1_c < prev2_o) &
        (prev1_o < prev2_o) & (curr_c < curr_o)
    ).astype(int)

    # 5. Lagged Features (Previous candles)
    for lag in [1, 2, 3]:
        df[f'close_lag_{lag}'] = df['close'].shift(lag)
        df[f'rsi_lag_{lag}'] = df['rsi'].shift(lag)
        
    # Drop rows with NaN (due to rolling windows)
    df = df.dropna()
    return df

def train_model(data_path="training_data.csv", market_type="otc"):
    """
    Trains a Gradient Boosting model using labeled data.
    market_type: "otc" or "real"
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return
    
    logger.info("Loading data...")
    df = pd.read_csv(data_path)
    df.dropna(inplace=True)
    
    # Separate features (X) and target (y)
    if 'outcome' not in df.columns:
        logger.error("Data missing 'outcome' column.")
        return

    # Drop non-feature columns (including string/meta columns that can't be used as features)
    drop_cols = ['time', 'outcome', 'signal', 'asset', 'from', 'to', 'market_type', 'pair',
                 'outcome_1m', 'outcome_5m', 'id']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df['outcome']
    
    # Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train - Random Forest
    logger.info("Training Random Forest Classifier...")
    # Parameters tuned for stability and better generalization than GB
    clf = RandomForestClassifier(
        n_estimators=100, 
        max_depth=10, 
        min_samples_split=5,
        random_state=42,
        n_jobs=-1 # Use all cores
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
        
    save_path = MODEL_PATH_TEMPLATE.format(market_type)
    joblib.dump(clf, save_path)
    logger.info(f"Model saved to {save_path}")
    return clf

def load_model(market_type="otc"):
    """Loads the trained model for a specific market type."""
    path = MODEL_PATH_TEMPLATE.format(market_type)
    
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load {market_type} model: {e}")
    elif market_type == "otc" and os.path.exists(LEGACY_MODEL_PATH):
        # Migration logic: if otc model doesn't exist but legacy one does, load legacy
        try:
            logger.info(f"Loading legacy model for OTC...")
            return joblib.load(LEGACY_MODEL_PATH)
        except Exception as e:
            logger.error(f"Failed to load legacy model: {e}")
    
    return None

LOSS_DETECTOR_PATH_TEMPLATE = os.path.join(MODELS_DIR, "loss_detector_{}.pkl")

def train_loss_detector(data_path="training_data.csv", market_type="otc"):
    """
    Trains a loss detector model with FLIPPED labels.
    y=1 means LOSS, y=0 means WIN.
    The model learns consistent losing patterns rather than noisy winning ones.
    """
    if not os.path.exists(data_path):
        logger.error(f"Data file not found: {data_path}")
        return

    logger.info("Loading data for loss detector training...")
    df = pd.read_csv(data_path, on_bad_lines='skip')
    df.dropna(inplace=True)

    if 'outcome' not in df.columns:
        logger.error("Data missing 'outcome' column.")
        return

    drop_cols = ['time', 'outcome', 'signal', 'asset', 'from', 'to', 'market_type', 'pair',
                 'outcome_1m', 'outcome_5m', 'id']
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])

    # FLIP labels: 1=LOSS, 0=WIN — we want to DETECT losing patterns
    y = 1 - df['outcome']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    logger.info("Training Loss Detector (Random Forest)...")
    clf = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    logger.info(f"Loss Detector Accuracy: {acc:.2f}")
    logger.info("\n" + classification_report(y_test, y_pred))

    if not os.path.exists(MODELS_DIR):
        os.makedirs(MODELS_DIR)

    save_path = LOSS_DETECTOR_PATH_TEMPLATE.format(market_type)
    joblib.dump(clf, save_path)
    logger.info(f"Loss detector saved to {save_path}")
    return clf

def load_loss_detector(market_type="otc"):
    """Loads the loss detector model for a specific market type."""
    path = LOSS_DETECTOR_PATH_TEMPLATE.format(market_type)
    if os.path.exists(path):
        try:
            return joblib.load(path)
        except Exception as e:
            logger.error(f"Failed to load loss detector ({market_type}): {e}")
    return None

def predict_loss(loss_model, features_df) -> float:
    """
    Predicts the probability that a trade will LOSE.
    Returns: float between 0.0 and 1.0 (higher = more likely a loss).
    """
    if loss_model is None:
        return 0.0  # No detector loaded: don't block

    try:
        if hasattr(loss_model, "feature_names_in_"):
            for c in set(loss_model.feature_names_in_) - set(features_df.columns):
                features_df[c] = 0
            features_df = features_df[loss_model.feature_names_in_]

        proba = loss_model.predict_proba(features_df)
        # Class 1 = LOSS probability
        return float(proba[0][1])
    except Exception as e:
        logger.error(f"Loss prediction error: {e}")
        return 0.0


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
            
        prediction = model.predict(features_df)
        return prediction[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return 1 # Fallback

def retrain_from_live_data(feedback_path="live_trade_feedback.csv"):
    """
    Triggers retraining using accumulated live feedback data.
    """
    if not os.path.exists(feedback_path):
        logger.warning(f"No feedback data found at {feedback_path}")
        return None
    
    try:
        df = pd.read_csv(feedback_path, on_bad_lines='skip')
        if len(df) < 50: # Minimum data threshold
            logger.info(f"Not enough feedback data ({len(df)}/50) to retrain.")
            return None
        
        # Normalize column name: feedback logs 'result', train_model expects 'outcome'
        if 'result' in df.columns and 'outcome' not in df.columns:
            df = df.rename(columns={'result': 'outcome'})
        
        logger.info(f"🔄 Retraining model with {len(df)} live trade results...")
        
        # Save normalized version to temp file for train_model
        temp_path = feedback_path.replace(".csv", "_normalized.csv")
        df.to_csv(temp_path, index=False)
        
        clf = train_model(data_path=temp_path, market_type="otc")
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except Exception:
            pass
        
        if clf:
            logger.info("✅ Model retrained and saved successfully.")
            return clf
    except Exception as e:
        logger.error(f"❌ Continuous learning error: {e}")
    
    return None

if __name__ == "__main__":
    retrain_from_live_data()
