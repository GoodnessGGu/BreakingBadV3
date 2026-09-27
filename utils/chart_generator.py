"""
utils/chart_generator.py - Advanced Candlestick Chart Engine

Generates sleek, dark-themed TradingView-style candlestick charts highlighting:
1. ICT Setup Detection (Sweep + CISD + FVG zone + Target levels)
2. Live Trade Execution (Long / Short entry markers, Entry line, SL & TP)
3. Breakeven / Trailing Stop Lock (Risk-free transition, SL ratchet, floating profit)
4. News Straddle Breakout Range (Pre-news consolidation, upper/lower breakout triggers)
"""

import io
import logging
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np

# Ensure headless Agg backend for background plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

# Dark Pro TradingView Theme Palette
BG_OUTER = '#131722'
BG_INNER = '#1e222d'
GRID_COLOR = '#2a2e39'
TEXT_COLOR = '#d1d4dc'
GREEN_CANDLE = '#089981'
RED_CANDLE = '#f23645'
CYAN_LINE = '#00e5ff'
GOLD_LINE = '#ffd700'
PURPLE_LINE = '#e040fb'


def _candles_to_df(raw_candles: Any) -> Optional[pd.DataFrame]:
    """Helper to ensure raw list of candles or DataFrame is converted to standardized DataFrame."""
    if raw_candles is None:
        return None
    if isinstance(raw_candles, pd.DataFrame):
        return raw_candles
    if isinstance(raw_candles, list):
        if not raw_candles:
            return None
        # Support dict format with min/max or Low/High
        rows = []
        for c in raw_candles:
            o = c.get("open") or c.get("Open") or c.get("from", 0.0)
            h = c.get("max") or c.get("High") or c.get("high", 0.0)
            l = c.get("min") or c.get("Low") or c.get("low", 0.0)
            cl = c.get("close") or c.get("Close") or c.get("to", 0.0)
            rows.append({"Open": float(o), "High": float(h), "Low": float(l), "Close": float(cl)})
        return pd.DataFrame(rows)
    return None


def generate_ict_setup_chart(
    df: Any,
    symbol: str,
    side: str,
    sweep_level: float,
    cisd_level: float,
    fvg_low: float,
    fvg_high: float,
    sl: float,
    tp: float,
    timeframe: str = "M1",
    num_candles: int = 40
) -> Optional[bytes]:
    """Renders an in-memory PNG candlestick chart of an ICT Setup."""
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 10:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        fig.patch.set_facecolor(BG_OUTER)
        ax.set_facecolor(BG_INNER)
        ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)

        width = 0.60
        wick_width = 1.1

        for i in range(n):
            c_open = float(sub_df.loc[i, 'Open'])
            c_close = float(sub_df.loc[i, 'Close'])
            c_high = float(sub_df.loc[i, 'High'])
            c_low = float(sub_df.loc[i, 'Low'])

            is_bull = c_close >= c_open
            color = GREEN_CANDLE if is_bull else RED_CANDLE

            ax.plot([i, i], [c_low, c_high], color=color, linewidth=wick_width, zorder=2)
            lower = min(c_open, c_close)
            height = max(abs(c_close - c_open), (c_high - c_low) * 0.01)
            rect = patches.Rectangle(
                (i - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3
            )
            ax.add_patch(rect)

        # Draw FVG Retest Zone
        fvg_color = '#00e676' if side == 'BUY' else '#ff5252'
        fvg_start_idx = max(0, n - 4)
        fvg_span_width = (n - fvg_start_idx) + 6

        rect_fvg = patches.Rectangle(
            (fvg_start_idx - 0.5, fvg_low),
            fvg_span_width,
            abs(fvg_high - fvg_low),
            facecolor=fvg_color,
            alpha=0.22,
            edgecolor=fvg_color,
            linestyle='--',
            linewidth=1.2,
            zorder=1
        )
        ax.add_patch(rect_fvg)

        fvg_mid = (fvg_low + fvg_high) / 2.0
        ax.text(
            fvg_start_idx + fvg_span_width - 1, fvg_mid,
            f"  FVG Retest [{fvg_low:.2f} - {fvg_high:.2f}]",
            color='#69f0ae' if side == 'BUY' else '#ff8a80',
            fontsize=8.5,
            verticalalignment='center',
            fontweight='bold'
        )

        # Draw CISD Level
        ax.axhline(cisd_level, color=CYAN_LINE, linestyle='-.', linewidth=1.2, alpha=0.9, zorder=4)
        ax.text(0.5, cisd_level, f" CISD Shift ({cisd_level:.2f})", color=CYAN_LINE, fontsize=8.5, fontweight='bold', verticalalignment='bottom' if side == 'BUY' else 'top')

        # Draw Liquidity Sweep Level
        ax.axhline(sweep_level, color=PURPLE_LINE, linestyle=':', linewidth=1.3, alpha=0.9, zorder=4)
        sweep_tag = f" Low Sweep ({sweep_level:.2f})" if side == 'BUY' else f" High Sweep ({sweep_level:.2f})"
        ax.text(0.5, sweep_level, sweep_tag, color=PURPLE_LINE, fontsize=8.5, fontweight='bold', verticalalignment='top' if side == 'BUY' else 'bottom')

        # Draw Planned Stop Loss & Take Profit
        ax.axhline(sl, color='#ff1744', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, sl, f" SL: {sl:.2f}", color='#ff1744', fontsize=8.5, fontweight='bold', verticalalignment='top' if side == 'BUY' else 'bottom')

        ax.axhline(tp, color='#00e676', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, tp, f" Projected TP: {tp:.2f}", color='#00e676', fontsize=8.5, fontweight='bold', verticalalignment='bottom' if side == 'BUY' else 'top')

        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), sweep_level, cisd_level, fvg_low, fvg_high, sl, tp]
        min_y, max_y = min(all_vals), max(all_vals)
        pad_y = (max_y - min_y) * 0.08
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        ax.set_title(
            f"[ICT {side} SETUP]  {symbol}  ({timeframe})  |  Waiting for FVG Retest",
            color='#ffffff', fontsize=11.5, fontweight='bold', pad=14, loc='left'
        )

        ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate ICT setup chart: {e}", exc_info=True)
        return None


def generate_trade_execution_chart(
    df: Any,
    symbol: str,
    side: str,
    entry_px: float,
    sl: float,
    tp: float,
    engine_name: str = "ICT",
    event_title: str = "",
    timeframe: str = "M1",
    num_candles: int = 35
) -> Optional[bytes]:
    """
    Renders an in-memory PNG candlestick chart highlighting instant trade entry.
    Features:
      - Prominent LONG (▲) / SHORT (▼) entry badge and direction icon
      - Distinct Entry Level line
      - Hard SL & TP boundaries
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        fig.patch.set_facecolor(BG_OUTER)
        ax.set_facecolor(BG_INNER)
        ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)

        width = 0.60
        wick_width = 1.1

        for i in range(n):
            c_open = float(sub_df.loc[i, 'Open'])
            c_close = float(sub_df.loc[i, 'Close'])
            c_high = float(sub_df.loc[i, 'High'])
            c_low = float(sub_df.loc[i, 'Low'])

            is_bull = c_close >= c_open
            color = GREEN_CANDLE if is_bull else RED_CANDLE

            ax.plot([i, i], [c_low, c_high], color=color, linewidth=wick_width, zorder=2)
            lower = min(c_open, c_close)
            height = max(abs(c_close - c_open), (c_high - c_low) * 0.01)
            rect = patches.Rectangle(
                (i - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3
            )
            ax.add_patch(rect)

        # Plot Entry Level & Marker
        is_long = side.upper() == "BUY" or side.upper() == "LONG"
        side_tag = "BUY / LONG" if is_long else "SELL / SHORT"
        entry_color = '#00e676' if is_long else '#ff5252'
        marker_symbol = '^' if is_long else 'v'

        # Entry Horizontal Line
        ax.axhline(entry_px, color=entry_color, linestyle='-', linewidth=1.6, alpha=0.95, zorder=4)
        ax.text(0.5, entry_px, f"  ENTRY @ {entry_px:.2f}", color=entry_color, fontsize=9.0, fontweight='bold', verticalalignment='bottom' if is_long else 'top')

        # Entry Marker on the latest execution candle
        last_idx = n - 1
        marker_y = sub_df.loc[last_idx, 'Low'] if is_long else sub_df.loc[last_idx, 'High']
        ax.scatter([last_idx], [marker_y], color=entry_color, s=180, marker=marker_symbol, edgecolor='#ffffff', linewidth=1.5, zorder=6)
        ax.text(
            last_idx, marker_y,
            f"  {side_tag}\n  @{entry_px:.2f}",
            color='#ffffff',
            fontsize=9.0,
            fontweight='bold',
            verticalalignment='top' if is_long else 'bottom',
            bbox=dict(boxstyle='round,pad=0.25', facecolor=entry_color, alpha=0.85, edgecolor='none'),
            zorder=7
        )

        # Stop Loss
        ax.axhline(sl, color='#ff1744', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, sl, f"  SL: {sl:.2f}", color='#ff1744', fontsize=8.5, fontweight='bold', verticalalignment='top' if is_long else 'bottom')

        # Take Profit
        ax.axhline(tp, color='#00e676', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, tp, f"  TP: {tp:.2f}", color='#00e676', fontsize=8.5, fontweight='bold', verticalalignment='bottom' if is_long else 'top')

        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), entry_px, sl, tp]
        min_y, max_y = min(all_vals), max(all_vals)
        pad_y = (max_y - min_y) * 0.08
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        header_event = f" | {event_title}" if event_title else ""
        ax.set_title(
            f"[{engine_name.upper()} {side_tag} EXECUTED]  {symbol}  ({timeframe}){header_event}",
            color='#ffffff', fontsize=11.5, fontweight='bold', pad=14, loc='left'
        )

        ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate execution chart: {e}", exc_info=True)
        return None


def generate_breakeven_chart(
    df: Any,
    symbol: str,
    side: str,
    entry_px: float,
    be_sl: float,
    initial_sl: float,
    tp: float,
    cur_px: float,
    engine_name: str = "Straddle",
    timeframe: str = "M1",
    num_candles: int = 35
) -> Optional[bytes]:
    """
    Renders an in-memory PNG candlestick chart when Stop Loss is moved to Breakeven.
    Highlights:
      - Active Trade Direction
      - New Breakeven SL level (Risk-free gold line)
      - Faint initial SL to show risk elimination
      - Current floating profit zone
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        fig.patch.set_facecolor(BG_OUTER)
        ax.set_facecolor(BG_INNER)
        ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)

        width = 0.60
        wick_width = 1.1

        for i in range(n):
            c_open = float(sub_df.loc[i, 'Open'])
            c_close = float(sub_df.loc[i, 'Close'])
            c_high = float(sub_df.loc[i, 'High'])
            c_low = float(sub_df.loc[i, 'Low'])

            is_bull = c_close >= c_open
            color = GREEN_CANDLE if is_bull else RED_CANDLE

            ax.plot([i, i], [c_low, c_high], color=color, linewidth=wick_width, zorder=2)
            lower = min(c_open, c_close)
            height = max(abs(c_close - c_open), (c_high - c_low) * 0.01)
            rect = patches.Rectangle(
                (i - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3
            )
            ax.add_patch(rect)

        is_long = side.upper() in ["BUY", "LONG"]

        # Original Entry Line
        ax.axhline(entry_px, color=CYAN_LINE, linestyle='-', linewidth=1.2, alpha=0.8, zorder=4)
        ax.text(0.5, entry_px, f"  Entry: {entry_px:.2f}", color=CYAN_LINE, fontsize=8.5, verticalalignment='bottom' if is_long else 'top')

        # Stop Loss (Gold Shield Line)
        ax.axhline(be_sl, color=GOLD_LINE, linestyle='-', linewidth=1.8, alpha=0.95, zorder=5)
        ax.text(0.5, be_sl, f"  * BREAKEVEN SL: {be_sl:.2f} (100% Risk-Free)", color=GOLD_LINE, fontsize=9.0, fontweight='bold', verticalalignment='top' if is_long else 'bottom')

        # Initial SL (Faint Dotted Red)
        if initial_sl and abs(initial_sl - be_sl) > 0.05:
            ax.axhline(initial_sl, color='#ff5252', linestyle=':', linewidth=1.0, alpha=0.4, zorder=3)
            ax.text(0.5, initial_sl, f"  [Initial SL: {initial_sl:.2f} Eliminated]", color='#ff8a80', fontsize=7.5, alpha=0.6, verticalalignment='top' if is_long else 'bottom')

        # Take Profit
        ax.axhline(tp, color='#00e676', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, tp, f"  TP: {tp:.2f}", color='#00e676', fontsize=8.5, fontweight='bold', verticalalignment='bottom' if is_long else 'top')

        # Risk-Free Badge on Current Bar
        last_idx = n - 1
        live_y = cur_px if cur_px > 0 else (sub_df.loc[last_idx, 'Close'])
        ax.scatter([last_idx], [live_y], color=GOLD_LINE, s=150, marker='*', edgecolor='#ffffff', linewidth=1.2, zorder=6)
        ax.text(
            last_idx, live_y,
            f"  * BREAKEVEN LOCKED\n  Live: {live_y:.2f}",
            color='#131722',
            fontsize=8.5,
            fontweight='bold',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=GOLD_LINE, alpha=0.9, edgecolor='none'),
            zorder=7
        )

        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), entry_px, be_sl, initial_sl, tp, cur_px]
        min_y, max_y = min(all_vals), max(all_vals)
        pad_y = (max_y - min_y) * 0.08
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        ax.set_title(
            f"[{engine_name.upper()} BREAKEVEN LOCKED]  {symbol}  ({timeframe})  |  Trade is 100% Risk-Free",
            color='#ffd700', fontsize=11.5, fontweight='bold', pad=14, loc='left'
        )

        ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate breakeven chart: {e}", exc_info=True)
        return None


def generate_straddle_setup_chart(
    df: Any,
    symbol: str,
    event_title: str,
    pre_high: float,
    pre_low: float,
    buy_trigger: float,
    sell_trigger: float,
    timeframe: str = "M1",
    num_candles: int = 25
) -> Optional[bytes]:
    """
    Renders the Pre-News Consolidation Range and the upper/lower breakout triggers.
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        fig.patch.set_facecolor(BG_OUTER)
        ax.set_facecolor(BG_INNER)
        ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)

        width = 0.60
        wick_width = 1.1

        for i in range(n):
            c_open = float(sub_df.loc[i, 'Open'])
            c_close = float(sub_df.loc[i, 'Close'])
            c_high = float(sub_df.loc[i, 'High'])
            c_low = float(sub_df.loc[i, 'Low'])

            is_bull = c_close >= c_open
            color = GREEN_CANDLE if is_bull else RED_CANDLE

            ax.plot([i, i], [c_low, c_high], color=color, linewidth=wick_width, zorder=2)
            lower = min(c_open, c_close)
            height = max(abs(c_close - c_open), (c_high - c_low) * 0.01)
            rect = patches.Rectangle(
                (i - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3
            )
            ax.add_patch(rect)

        # Pre-News Consolidation Shaded Box
        range_height = abs(pre_high - pre_low)
        rect_range = patches.Rectangle(
            (-0.5, pre_low),
            n + 6,
            range_height,
            facecolor='#3d5afe',
            alpha=0.15,
            edgecolor='#536dfe',
            linestyle=':',
            linewidth=1.3,
            zorder=1
        )
        ax.add_patch(rect_range)

        range_mid = (pre_high + pre_low) / 2.0
        ax.text(
            0.5, range_mid,
            f"  Pre-News Consolidation Range [{pre_low:.2f} – {pre_high:.2f}]",
            color='#8c9eff',
            fontsize=8.5,
            fontweight='bold',
            verticalalignment='center'
        )

        # Upper Breakout Trigger (BUY)
        ax.axhline(buy_trigger, color='#00e676', linestyle='--', linewidth=1.6, alpha=0.95, zorder=4)
        ax.text(0.5, buy_trigger, f"  ▲ BUY Breakout Trigger: {buy_trigger:.2f} (+1.50)", color='#00e676', fontsize=9.0, fontweight='bold', verticalalignment='bottom')

        # Lower Breakout Trigger (SELL)
        ax.axhline(sell_trigger, color='#ff1744', linestyle='--', linewidth=1.6, alpha=0.95, zorder=4)
        ax.text(0.5, sell_trigger, f"  ▼ SELL Breakout Trigger: {sell_trigger:.2f} (-1.50)", color='#ff1744', fontsize=9.0, fontweight='bold', verticalalignment='top')

        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), pre_high, pre_low, buy_trigger, sell_trigger]
        min_y, max_y = min(all_vals), max(all_vals)
        pad_y = (max_y - min_y) * 0.08
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        ax.set_title(
            f"[NEWS STRADDLE ARMED]  {symbol} (M1)  |  {event_title}",
            color='#ffffff', fontsize=11.5, fontweight='bold', pad=14, loc='left'
        )

        ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate straddle setup chart: {e}", exc_info=True)
        return None


def generate_trade_close_chart(
    df: Any,
    symbol: str,
    side: str,
    entry_px: float,
    exit_px: float,
    tp: float,
    sl: float,
    pnl: float,
    reason: str = "take_profit",
    engine_name: str = "Straddle",
    event_title: str = "",
    timeframe: str = "M1",
    num_candles: int = 40
) -> Optional[bytes]:
    """
    Renders an in-memory PNG candlestick chart when Take Profit, Breakeven, or Stop Loss is hit.
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        fig.patch.set_facecolor(BG_OUTER)
        ax.set_facecolor(BG_INNER)
        ax.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.7)

        width = 0.60
        wick_width = 1.1

        for i in range(n):
            c_open = float(sub_df.loc[i, 'Open'])
            c_close = float(sub_df.loc[i, 'Close'])
            c_high = float(sub_df.loc[i, 'High'])
            c_low = float(sub_df.loc[i, 'Low'])

            is_bull = c_close >= c_open
            color = GREEN_CANDLE if is_bull else RED_CANDLE

            ax.plot([i, i], [c_low, c_high], color=color, linewidth=wick_width, zorder=2)
            lower = min(c_open, c_close)
            height = max(abs(c_close - c_open), (c_high - c_low) * 0.01)
            rect = patches.Rectangle(
                (i - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3
            )
            ax.add_patch(rect)

        is_long = side.upper() in ["BUY", "LONG"]

        # 1. Entry Line
        ax.axhline(entry_px, color=CYAN_LINE, linestyle='-', linewidth=1.2, alpha=0.8, zorder=4)
        ax.text(0.5, entry_px, f"  Entry: {entry_px:.2f}", color=CYAN_LINE, fontsize=8.5, verticalalignment='bottom' if is_long else 'top')

        # 2. Target Take Profit Line
        if tp and abs(tp - exit_px) > 0.05:
            ax.axhline(tp, color='#00e676', linestyle='--', linewidth=1.2, alpha=0.6, zorder=4)
            ax.text(0.5, tp, f"  Target TP: {tp:.2f}", color='#69f0ae', fontsize=8.0, alpha=0.8, verticalalignment='bottom' if is_long else 'top')

        # 3. Exit Price Level & Badge
        is_tp = pnl > 0
        is_be = pnl == 0
        exit_color = '#00e676' if is_tp else (GOLD_LINE if is_be else '#ff1744')
        badge_text = f"TP HIT: +${pnl:.2f}" if is_tp else (f"BREAKEVEN: $0.00" if is_be else f"SL HIT: -${abs(pnl):.2f}")

        ax.axhline(exit_px, color=exit_color, linestyle='-', linewidth=1.8, alpha=0.95, zorder=5)
        ax.text(0.5, exit_px, f"  [{badge_text}] @ {exit_px:.2f}", color=exit_color, fontsize=9.0, fontweight='bold', verticalalignment='top' if is_long else 'bottom')

        # Marker on final candle
        last_idx = n - 1
        marker_y = exit_px if exit_px > 0 else (sub_df.loc[last_idx, 'Close'])
        marker_sym = '*' if is_tp else ('o' if is_be else 'x')
        ax.scatter([last_idx], [marker_y], color=exit_color, s=180, marker=marker_sym, edgecolor='#ffffff', linewidth=1.4, zorder=6)
        ax.text(
            last_idx, marker_y,
            f"  [{badge_text}]\n  Exit: {exit_px:.2f}",
            color='#ffffff' if not is_be else '#131722',
            fontsize=8.5,
            fontweight='bold',
            verticalalignment='center',
            bbox=dict(boxstyle='round,pad=0.3', facecolor=exit_color, alpha=0.9, edgecolor='none'),
            zorder=7
        )

        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), entry_px, exit_px, tp, sl]
        min_y, max_y = min(all_vals), max(all_vals)
        pad_y = (max_y - min_y) * 0.08
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        header_title = f"[WIN: +${pnl:.2f}]" if is_tp else (f"[BREAKEVEN: $0.00]" if is_be else f"[CLOSED: -${abs(pnl):.2f}]")
        event_tag = f" | {event_title}" if event_title else ""
        ax.set_title(
            f"[{engine_name.upper()} {header_title}]  {symbol}  ({timeframe}){event_tag}",
            color='#ffffff', fontsize=11.5, fontweight='bold', pad=14, loc='left'
        )

        ax.tick_params(colors=TEXT_COLOR, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate trade close chart: {e}", exc_info=True)
        return None

