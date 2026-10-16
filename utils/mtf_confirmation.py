"""
utils/mtf_confirmation.py - Multi-Timeframe (MTF) Higher-Timeframe Trend Confirmation

Verifies 15-Minute strategy execution setups (SND Imbalance & Trendline) against
higher-timeframe macro market structure (1-Hour & 4-Hour EMA trend filters).
"""

import logging
from typing import Dict, Any, Tuple, Optional, List
import pandas as pd
import numpy as np

logger = logging.getLogger("MTFConfirmation")


class MTFConfirmation:
    """
    Higher-timeframe macro confirmation filter.
    Gates lower-timeframe strategy entries to ensure alignment with institutional orderflow.
    """

    def __init__(self, h1_size: int = 3600, h4_size: int = 14400, strict: bool = False):
        self.h1_size = h1_size
        self.h4_size = h4_size
        self.strict = strict

    @staticmethod
    def _calculate_ema(series: pd.Series, span: int) -> pd.Series:
        return series.ewm(span=span, adjust=False).mean()

    def check_higher_timeframe_bias(
        self,
        mcp_client: Any,
        asset_id: int,
        setup_side: str,
        symbol: str = "",
        h1_count: int = 60,
        h4_count: int = 30
    ) -> Tuple[bool, str, Dict[str, Any]]:
        """
        Evaluate H1 and H4 candle structure for directional alignment with a proposed setup.
        Returns: (is_aligned: bool, reason: str, metrics: dict)
        """
        setup_side = setup_side.upper()
        metrics: Dict[str, Any] = {
            "h1_trend": "UNKNOWN",
            "h4_trend": "UNKNOWN",
            "h1_ema50": None,
            "h1_ema200": None,
            "h4_ema50": None,
            "h4_close": None
        }

        if not mcp_client:
            return True, "No MCP client available - bypassed", metrics

        try:
            # 1. Fetch H1 Candles
            h1_candles = mcp_client.get_candles(asset_id=asset_id, size=self.h1_size, count=h1_count)
            if not h1_candles or len(h1_candles) < 20:
                logger.warning(f"⚠️ [MTF] Insufficient H1 candles for {symbol or asset_id} ({len(h1_candles) if h1_candles else 0} fetched). Failing open.")
                return True, "Insufficient H1 candles (fail-open)", metrics

            df_h1 = pd.DataFrame(h1_candles)
            close_col = "close" if "close" in df_h1.columns else "c"
            df_h1["Close"] = pd.to_numeric(df_h1[close_col])
            
            # EMA calculations
            ema_fast_span = 20 if len(df_h1) < 50 else 50
            ema_slow_span = 50 if len(df_h1) < 100 else 200
            
            df_h1["ema_fast"] = self._calculate_ema(df_h1["Close"], ema_fast_span)
            df_h1["ema_slow"] = self._calculate_ema(df_h1["Close"], ema_slow_span)

            latest_h1_close = float(df_h1["Close"].iloc[-1])
            latest_h1_fast = float(df_h1["ema_fast"].iloc[-1])
            latest_h1_slow = float(df_h1["ema_slow"].iloc[-1])

            metrics["h1_close"] = latest_h1_close
            metrics["h1_ema50"] = latest_h1_fast
            metrics["h1_ema200"] = latest_h1_slow

            h1_bullish = latest_h1_fast >= latest_h1_slow and latest_h1_close >= latest_h1_fast * 0.998
            h1_bearish = latest_h1_fast <= latest_h1_slow and latest_h1_close <= latest_h1_fast * 1.002

            if h1_bullish:
                metrics["h1_trend"] = "BULLISH"
            elif h1_bearish:
                metrics["h1_trend"] = "BEARISH"
            else:
                metrics["h1_trend"] = "NEUTRAL"

            # 2. Fetch H4 Candles (if available)
            try:
                h4_candles = mcp_client.get_candles(asset_id=asset_id, size=self.h4_size, count=h4_count)
                if h4_candles and len(h4_candles) >= 15:
                    df_h4 = pd.DataFrame(h4_candles)
                    c_col = "close" if "close" in df_h4.columns else "c"
                    df_h4["Close"] = pd.to_numeric(df_h4[c_col])
                    df_h4["ema50"] = self._calculate_ema(df_h4["Close"], min(30, len(df_h4) - 1))
                    
                    latest_h4_close = float(df_h4["Close"].iloc[-1])
                    latest_h4_ema = float(df_h4["ema50"].iloc[-1])
                    metrics["h4_close"] = latest_h4_close
                    metrics["h4_ema50"] = latest_h4_ema

                    if latest_h4_close >= latest_h4_ema:
                        metrics["h4_trend"] = "BULLISH"
                    else:
                        metrics["h4_trend"] = "BEARISH"
            except Exception as h4_err:
                logger.debug(f"[MTF] H4 check skipped: {h4_err}")

            # 3. Decision Gate
            if setup_side == "BUY":
                if metrics["h1_trend"] == "BEARISH" and (metrics["h4_trend"] == "BEARISH" or self.strict):
                    msg = f"BUY rejected: Strong macro downtrend on H1/H4 (H1 Close: {latest_h1_close:.5f} < EMA: {latest_h1_fast:.5f})"
                    logger.warning(f"🛡️ [MTF Gate] {symbol} {msg}")
                    return False, msg, metrics
                return True, "Macro trend confirmed BUY alignment", metrics

            elif setup_side == "SELL":
                if metrics["h1_trend"] == "BULLISH" and (metrics["h4_trend"] == "BULLISH" or self.strict):
                    msg = f"SELL rejected: Strong macro uptrend on H1/H4 (H1 Close: {latest_h1_close:.5f} > EMA: {latest_h1_fast:.5f})"
                    logger.warning(f"🛡️ [MTF Gate] {symbol} {msg}")
                    return False, msg, metrics
                return True, "Macro trend confirmed SELL alignment", metrics

            return True, "Setup aligned", metrics

        except Exception as e:
            logger.error(f"[MTF] Error evaluating MTF bias for {symbol or asset_id}: {e}")
            # Fail-open for continuous execution
            return True, f"MTF evaluation error ({e}) - fail-open", metrics


mtf_gate = MTFConfirmation()
