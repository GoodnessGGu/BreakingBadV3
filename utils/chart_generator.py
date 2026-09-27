"""
utils/chart_generator.py - Advanced TradingView Pro Candlestick Chart Engine

Generates ultra-sleek, modern TradingView-grade dark candlestick charts:
1. ICT Setup Detection (Sweep level + CISD line + FVG retest zone + Target levels)
2. Live Trade Execution (Long ▲ / Short ▼ entry markers, Entry line, SL/TP shaded risk-reward zones)
3. Breakeven / Trailing Stop Lock (Risk-free gold shield transition, SL ratchet, floating profit)
4. News Straddle Breakout Range (Pre-news consolidation zone, dual breakout triggers)
5. Trade Close / TP Hit (Profit / Breakeven / Loss badge, exit marker, PnL banner)
"""

import io
import logging
from typing import Optional, Any, List, Tuple
import pandas as pd
import numpy as np

# Headless Agg backend for background plotting
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.gridspec import GridSpec

logger = logging.getLogger(__name__)

# --- TradingView Pro Dark Theme Palette ---
BG_OUTER = '#0c1017'       # Deep Obsidian
BG_INNER = '#121824'       # Midnight Navy-Slate
GRID_COLOR = '#1c2436'     # Subtle slate grid
TEXT_MAIN = '#f1f5f9'      # Crisp White
TEXT_MUTED = '#94a3b8'     # Slate Muted Silver
BORDER_COLOR = '#1e293b'   # Container Border

# Candles
BULL_COLOR = '#08d48c'     # Emerald Neon Green
BEAR_COLOR = '#ff3d60'     # Crimson Neon Red
WICK_WIDTH = 1.15
BODY_WIDTH = 0.62

# Level Colors
CYAN_ACCENT = '#00f0ff'    # Entry / CISD Cyan
GOLD_ACCENT = '#ffb703'    # Breakeven / Gold Shield
PURPLE_ACCENT = '#c084fc'  # Liquidity Sweep
BLUE_ACCENT = '#3b82f6'    # Pre-News Box


def _candles_to_df(raw_candles: Any) -> Optional[pd.DataFrame]:
    """Helper to ensure raw list of candles or DataFrame is converted to standardized DataFrame."""
    if raw_candles is None:
        return None
    if isinstance(raw_candles, pd.DataFrame):
        df = raw_candles.copy()
        # Normalize column names if needed
        cols = {c: c.capitalize() for c in df.columns}
        df.rename(columns=cols, inplace=True)
        if all(k in df.columns for k in ['Open', 'High', 'Low', 'Close']):
            return df
        return None
    if isinstance(raw_candles, list):
        if not raw_candles:
            return None
        rows = []
        for c in raw_candles:
            o = c.get("open") or c.get("Open") or c.get("from", 0.0)
            h = c.get("max") or c.get("High") or c.get("high", 0.0)
            l = c.get("min") or c.get("Low") or c.get("low", 0.0)
            cl = c.get("close") or c.get("Close") or c.get("to", 0.0)
            v = c.get("volume") or c.get("Volume") or abs(float(cl) - float(o)) * 100
            rows.append({
                "Open": float(o),
                "High": float(h),
                "Low": float(l),
                "Close": float(cl),
                "Volume": float(v)
            })
        return pd.DataFrame(rows)
    return None


def _setup_figure(title_tag: str, tag_color: str, symbol: str, timeframe: str, subtitle: str = ""):
    """Creates a high-DPI figure with header badges and volume underlay layout."""
    fig = plt.figure(figsize=(12, 6.6), dpi=140)
    fig.patch.set_facecolor(BG_OUTER)

    gs = GridSpec(2, 1, height_ratios=[5.2, 1.0], hspace=0.06, figure=fig)
    ax = fig.add_subplot(gs[0])
    ax_vol = fig.add_subplot(gs[1], sharex=ax)

    # Backgrounds & Spines
    for a in (ax, ax_vol):
        a.set_facecolor(BG_INNER)
        a.grid(True, color=GRID_COLOR, linestyle='--', linewidth=0.5, alpha=0.8)
        for spine in a.spines.values():
            spine.set_color(BORDER_COLOR)
            spine.set_linewidth(1.0)

    # Hide volume x-axis ticks
    plt.setp(ax.get_xticklabels(), visible=False)
    ax_vol.tick_params(colors=TEXT_MUTED, labelsize=7.5)
    ax.tick_params(colors=TEXT_MUTED, labelsize=8.5)
    ax_vol.set_yticks([])

    # Top Header Pill Badges
    header_x = 0.02
    header_y = 0.94
    fig.text(
        header_x, header_y, f"  {title_tag}  ",
        color='#0c1017', fontsize=9.0, fontweight='heavy',
        bbox=dict(boxstyle='round,pad=0.35,rounding_size=0.25', facecolor=tag_color, edgecolor='none')
    )

    sub_info = f"  {symbol.upper()}  •  {timeframe.upper()}"
    if subtitle:
        sub_info += f"  •  {subtitle}"
    fig.text(header_x + 0.14 + (len(title_tag) * 0.007), header_y + 0.004, sub_info, color=TEXT_MAIN, fontsize=10.5, fontweight='bold')

    # Watermark
    ax.text(
        0.5, 0.5, "BREAKINGBAD V3",
        transform=ax.transAxes,
        fontsize=24, color='#ffffff', alpha=0.025,
        ha='center', va='center', fontweight='black', rotation=15
    )

    return fig, ax, ax_vol


def _plot_candles_and_volume(ax, ax_vol, df: pd.DataFrame):
    """Draws candlestick bars and volume momentum underlay."""
    n = len(df)
    for i in range(n):
        o = float(df.loc[i, 'Open'])
        h = float(df.loc[i, 'High'])
        l = float(df.loc[i, 'Low'])
        c = float(df.loc[i, 'Close'])
        v = float(df.loc[i, 'Volume']) if 'Volume' in df.columns else abs(c - o)

        is_bull = c >= o
        color = BULL_COLOR if is_bull else BEAR_COLOR

        # Candlestick Wick & Body
        ax.plot([i, i], [l, h], color=color, linewidth=WICK_WIDTH, zorder=2)
        lower = min(o, c)
        height = max(abs(c - o), (h - l) * 0.02 if (h - l) > 0 else 0.01)
        rect = patches.Rectangle(
            (i - BODY_WIDTH / 2, lower), BODY_WIDTH, height,
            facecolor=color, edgecolor=color, zorder=3
        )
        ax.add_patch(rect)

        # Volume Bar Underlay
        ax_vol.bar(i, v, color=color, alpha=0.35, width=BODY_WIDTH, zorder=2)


def _add_price_pill(ax, y_val: float, text: str, color: str, text_color: str = '#ffffff', align: str = 'right', x_pos: float = None):
    """Draws a sleek right-pinned TradingView-style price pill."""
    xlim = ax.get_xlim()
    px = (xlim[1] - 0.2) if x_pos is None else x_pos
    ax.text(
        px, y_val, f"  {text}  ",
        color=text_color, fontsize=8.0, fontweight='bold',
        va='center', ha=align,
        bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.2', facecolor=color, edgecolor='none'),
        zorder=8
    )


def _render_and_close(fig) -> bytes:
    """Exports matplotlib figure to PNG byte buffer."""
    buf = io.BytesIO()
    fig.subplots_adjust(top=0.88, bottom=0.08, left=0.06, right=0.94)
    plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none', dpi=140)
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()


# ============================================================================
# 1. ICT SETUP DETECTION CHART
# ============================================================================
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
    """Renders TradingView Pro ICT setup chart with FVG retest zone and CISD shift."""
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 10:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)
        is_bull = side.upper() in ["BUY", "LONG"]

        tag_color = BULL_COLOR if is_bull else BEAR_COLOR
        fig, ax, ax_vol = _setup_figure(
            title_tag=f"ICT {side.upper()} SETUP",
            tag_color=tag_color,
            symbol=symbol,
            timeframe=timeframe,
            subtitle="FVG Retest Imbalance Zone"
        )

        _plot_candles_and_volume(ax, ax_vol, sub_df)

        # FVG Retest Shaded Area
        fvg_color = BULL_COLOR if is_bull else BEAR_COLOR
        fvg_start_idx = max(0, n - 6)
        fvg_width = (n - fvg_start_idx) + 8
        fvg_height = abs(fvg_high - fvg_low)

        rect_fvg = patches.Rectangle(
            (fvg_start_idx - 0.4, min(fvg_low, fvg_high)),
            fvg_width, fvg_height,
            facecolor=fvg_color, alpha=0.18,
            edgecolor=fvg_color, linestyle='--', linewidth=1.2, zorder=1
        )
        ax.add_patch(rect_fvg)
        fvg_mid = (fvg_low + fvg_high) / 2.0
        ax.text(
            fvg_start_idx + 0.3, fvg_mid, f"FVG ZONE [{fvg_low:.2f} - {fvg_high:.2f}]",
            color=fvg_color, fontsize=8.0, fontweight='bold', va='center', alpha=0.9, zorder=4
        )

        # CISD Line
        ax.axhline(cisd_level, color=CYAN_ACCENT, linestyle='-.', linewidth=1.3, alpha=0.9, zorder=4)
        # Sweep Line
        ax.axhline(sweep_level, color=PURPLE_ACCENT, linestyle=':', linewidth=1.3, alpha=0.9, zorder=4)
        # SL & TP lines
        ax.axhline(sl, color=BEAR_COLOR, linestyle='--', linewidth=1.3, alpha=0.85, zorder=4)
        ax.axhline(tp, color=BULL_COLOR, linestyle='--', linewidth=1.3, alpha=0.85, zorder=4)

        # Set Limits
        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), sweep_level, cisd_level, fvg_low, fvg_high, sl, tp]
        min_y, max_y = min(all_vals), max(all_vals)
        pad = (max_y - min_y) * 0.09
        ax.set_ylim(min_y - pad, max_y + pad)

        # Right Price Pills
        _add_price_pill(ax, cisd_level, f"CISD: {cisd_level:.2f}", CYAN_ACCENT, text_color='#0c1017')
        _add_price_pill(ax, sweep_level, f"SWEEP: {sweep_level:.2f}", PURPLE_ACCENT, text_color='#0c1017')
        _add_price_pill(ax, sl, f"SL: {sl:.2f}", BEAR_COLOR)
        _add_price_pill(ax, tp, f"TARGET TP: {tp:.2f}", BULL_COLOR, text_color='#0c1017')

        return _render_and_close(fig)

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate ICT setup chart: {e}", exc_info=True)
        return None


# ============================================================================
# 2. NEWS STRADDLE ARMED SETUP CHART
# ============================================================================
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
    """Renders Pre-News Consolidation Box and dual upper/lower breakout triggers."""
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax, ax_vol = _setup_figure(
            title_tag="NEWS STRADDLE ARMED",
            tag_color=GOLD_ACCENT,
            symbol=symbol,
            timeframe=timeframe,
            subtitle=event_title or "High-Impact Breakout"
        )

        _plot_candles_and_volume(ax, ax_vol, sub_df)

        # Pre-News Consolidation Box
        box_width = n + 6
        box_height = abs(pre_high - pre_low)
        rect_range = patches.Rectangle(
            (-0.5, pre_low), box_width, box_height,
            facecolor=BLUE_ACCENT, alpha=0.14,
            edgecolor=BLUE_ACCENT, linestyle=':', linewidth=1.4, zorder=1
        )
        ax.add_patch(rect_range)
        ax.text(
            0.5, (pre_high + pre_low) / 2.0,
            f"Pre-News Range [{pre_low:.2f} - {pre_high:.2f}]",
            color='#93c5fd', fontsize=8.5, fontweight='bold', va='center', zorder=4
        )

        # Upper & Lower Triggers
        ax.axhline(buy_trigger, color=BULL_COLOR, linestyle='--', linewidth=1.6, alpha=0.95, zorder=4)
        ax.axhline(sell_trigger, color=BEAR_COLOR, linestyle='--', linewidth=1.6, alpha=0.95, zorder=4)

        # Set Limits
        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), pre_high, pre_low, buy_trigger, sell_trigger]
        min_y, max_y = min(all_vals), max(all_vals)
        pad = (max_y - min_y) * 0.10
        ax.set_ylim(min_y - pad, max_y + pad)

        # Right Price Pills
        _add_price_pill(ax, buy_trigger, f"BUY TRIGGER: {buy_trigger:.2f}", BULL_COLOR, text_color='#0c1017')
        _add_price_pill(ax, sell_trigger, f"SELL TRIGGER: {sell_trigger:.2f}", BEAR_COLOR)

        return _render_and_close(fig)

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate straddle setup chart: {e}", exc_info=True)
        return None


# ============================================================================
# 3. LIVE TRADE EXECUTION CHART
# ============================================================================
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
    Renders instant trade execution chart:
    - Long ▲ / Short ▼ marker on entry candle
    - Shaded green profit expansion & red risk buffer zones
    - Right-aligned price pills for Entry, SL, TP
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)
        is_long = side.upper() in ["BUY", "LONG"]

        tag_color = BULL_COLOR if is_long else BEAR_COLOR
        side_label = "LONG / BUY" if is_long else "SHORT / SELL"
        fig, ax, ax_vol = _setup_figure(
            title_tag=f"{engine_name.upper()} {side_label} EXECUTED",
            tag_color=tag_color,
            symbol=symbol,
            timeframe=timeframe,
            subtitle=event_title or "Trade Filled"
        )

        _plot_candles_and_volume(ax, ax_vol, sub_df)

        # Shaded Risk & Reward Zones
        reward_rect = patches.Rectangle(
            (-0.5, min(entry_px, tp)), n + 6, abs(tp - entry_px),
            facecolor=BULL_COLOR, alpha=0.08, edgecolor='none', zorder=1
        )
        risk_rect = patches.Rectangle(
            (-0.5, min(entry_px, sl)), n + 6, abs(sl - entry_px),
            facecolor=BEAR_COLOR, alpha=0.08, edgecolor='none', zorder=1
        )
        ax.add_patch(reward_rect)
        ax.add_patch(risk_rect)

        # Lines
        ax.axhline(entry_px, color=CYAN_ACCENT, linestyle='-', linewidth=1.6, alpha=0.95, zorder=4)
        ax.axhline(sl, color=BEAR_COLOR, linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.axhline(tp, color=BULL_COLOR, linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)

        # Entry Marker on Latest Execution Candle
        last_idx = n - 1
        marker_y = sub_df.loc[last_idx, 'Low'] if is_long else sub_df.loc[last_idx, 'High']
        marker_sym = '^' if is_long else 'v'

        ax.scatter([last_idx], [marker_y], color=tag_color, s=200, marker=marker_sym, edgecolor='#ffffff', linewidth=1.5, zorder=7)
        offset = -(abs(tp - entry_px) * 0.08) if is_long else (abs(tp - entry_px) * 0.08)
        ax.text(
            last_idx, marker_y + offset,
            f" [ENTRY: {entry_px:.2f}] ",
            color='#ffffff', fontsize=8.5, fontweight='bold',
            va='top' if is_long else 'bottom', ha='center',
            bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.2', facecolor='#1e293b', edgecolor=tag_color, linewidth=1.2),
            zorder=8
        )

        # Set Limits
        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), entry_px, sl, tp]
        min_y, max_y = min(all_vals), max(all_vals)
        pad = (max_y - min_y) * 0.09
        ax.set_ylim(min_y - pad, max_y + pad)

        # Right Price Pills
        _add_price_pill(ax, entry_px, f"ENTRY: {entry_px:.2f}", CYAN_ACCENT, text_color='#0c1017')
        _add_price_pill(ax, sl, f"SL: {sl:.2f}", BEAR_COLOR)
        _add_price_pill(ax, tp, f"TP: {tp:.2f}", BULL_COLOR, text_color='#0c1017')

        return _render_and_close(fig)

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate execution chart: {e}", exc_info=True)
        return None


# ============================================================================
# 4. BREAKEVEN / TRAILING STOP LOCKED CHART
# ============================================================================
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
    Renders chart when Stop Loss is ratcheted to Breakeven (100% Risk Free).
    - Gold Shield line for Breakeven SL
    - Faint crossed-out initial SL line
    - Active floating profit zone
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)
        is_long = side.upper() in ["BUY", "LONG"]

        fig, ax, ax_vol = _setup_figure(
            title_tag="BREAKEVEN LOCKED",
            tag_color=GOLD_ACCENT,
            symbol=symbol,
            timeframe=timeframe,
            subtitle="Trade is 100% Risk-Free"
        )

        _plot_candles_and_volume(ax, ax_vol, sub_df)

        # Protected Profit / Risk-Free Shaded Zone
        profit_zone = patches.Rectangle(
            (-0.5, min(entry_px, cur_px)), n + 6, abs(cur_px - entry_px),
            facecolor=GOLD_ACCENT, alpha=0.09, edgecolor='none', zorder=1
        )
        ax.add_patch(profit_zone)

        # Lines
        ax.axhline(entry_px, color=CYAN_ACCENT, linestyle='-', linewidth=1.3, alpha=0.75, zorder=4)
        ax.axhline(be_sl, color=GOLD_ACCENT, linestyle='-', linewidth=1.8, alpha=0.95, zorder=5)
        ax.axhline(tp, color=BULL_COLOR, linestyle='--', linewidth=1.4, alpha=0.85, zorder=4)

        if initial_sl and abs(initial_sl - be_sl) > 0.05:
            ax.axhline(initial_sl, color=BEAR_COLOR, linestyle=':', linewidth=1.0, alpha=0.35, zorder=3)
            ax.text(
                0.5, initial_sl, "  [Initial Risk Eliminated]  ",
                color='#fca5a5', fontsize=7.5, alpha=0.6, va='center', zorder=4
            )

        # Live Marker on latest candle
        last_idx = n - 1
        live_y = cur_px if cur_px > 0 else float(sub_df.loc[last_idx, 'Close'])
        ax.scatter([last_idx], [live_y], color=GOLD_ACCENT, s=200, marker='*', edgecolor='#ffffff', linewidth=1.4, zorder=7)
        ax.text(
            last_idx, live_y,
            f" [LIVE: {live_y:.2f}] ",
            color='#0c1017', fontsize=8.5, fontweight='heavy',
            va='bottom' if is_long else 'top', ha='center',
            bbox=dict(boxstyle='round,pad=0.25,rounding_size=0.2', facecolor=GOLD_ACCENT, edgecolor='#ffffff', linewidth=1.0),
            zorder=8
        )

        # Set Limits
        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), entry_px, be_sl, initial_sl, tp, cur_px]
        min_y, max_y = min(all_vals), max(all_vals)
        pad = (max_y - min_y) * 0.09
        ax.set_ylim(min_y - pad, max_y + pad)

        # Right Price Pills
        _add_price_pill(ax, entry_px, f"ENTRY: {entry_px:.2f}", CYAN_ACCENT, text_color='#0c1017')
        _add_price_pill(ax, be_sl, f"BREAKEVEN SL: {be_sl:.2f}", GOLD_ACCENT, text_color='#0c1017')
        _add_price_pill(ax, tp, f"TP: {tp:.2f}", BULL_COLOR, text_color='#0c1017')

        return _render_and_close(fig)

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate breakeven chart: {e}", exc_info=True)
        return None


# ============================================================================
# 5. TRADE CLOSE / TAKE PROFIT HIT CHART
# ============================================================================
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
    Renders chart when trade concludes (TP Hit, Breakeven Exit, or SL Hit):
    - Distinct PnL outcome banner & pill badge
    - Exit point star marker
    - Captured profit expansion shading
    """
    try:
        data = _candles_to_df(df)
        if data is None or len(data) < 5:
            return None

        sub_df = data.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)
        is_long = side.upper() in ["BUY", "LONG"]

        is_tp = pnl > 0.1
        is_be = abs(pnl) <= 0.1
        is_sl = pnl < -0.1

        if is_tp:
            badge_title = f"TAKE PROFIT HIT (+${pnl:.2f})"
            tag_color = BULL_COLOR
            exit_pill_color = BULL_COLOR
            exit_text_color = '#0c1017'
        elif is_be:
            badge_title = "BREAKEVEN CLOSED ($0.00)"
            tag_color = GOLD_ACCENT
            exit_pill_color = GOLD_ACCENT
            exit_text_color = '#0c1017'
        else:
            badge_title = f"STOP LOSS HIT (-${abs(pnl):.2f})"
            tag_color = BEAR_COLOR
            exit_pill_color = BEAR_COLOR
            exit_text_color = '#ffffff'

        fig, ax, ax_vol = _setup_figure(
            title_tag=badge_title,
            tag_color=tag_color,
            symbol=symbol,
            timeframe=timeframe,
            subtitle=f"{engine_name.upper()} Outcome"
        )

        _plot_candles_and_volume(ax, ax_vol, sub_df)

        # Shaded Result Zone
        zone_color = BULL_COLOR if is_tp else (GOLD_ACCENT if is_be else BEAR_COLOR)
        res_rect = patches.Rectangle(
            (-0.5, min(entry_px, exit_px)), n + 6, abs(exit_px - entry_px),
            facecolor=zone_color, alpha=0.12, edgecolor='none', zorder=1
        )
        ax.add_patch(res_rect)

        # Lines
        ax.axhline(entry_px, color=CYAN_ACCENT, linestyle='-', linewidth=1.2, alpha=0.75, zorder=4)
        ax.axhline(exit_px, color=exit_pill_color, linestyle='-', linewidth=1.8, alpha=0.95, zorder=5)
        if tp and abs(tp - exit_px) > 0.05:
            ax.axhline(tp, color=BULL_COLOR, linestyle='--', linewidth=1.1, alpha=0.6, zorder=4)

        # Exit Marker on final candle
        last_idx = n - 1
        marker_y = exit_px if exit_px > 0 else float(sub_df.loc[last_idx, 'Close'])
        marker_sym = '*' if is_tp else ('o' if is_be else 'X')

        ax.scatter([last_idx], [marker_y], color=exit_pill_color, s=240, marker=marker_sym, edgecolor='#ffffff', linewidth=1.5, zorder=7)
        ax.text(
            last_idx, marker_y,
            f"  [EXIT: {exit_px:.2f} | {badge_title}]  ",
            color=exit_text_color, fontsize=8.5, fontweight='heavy',
            va='center', ha='right',
            bbox=dict(boxstyle='round,pad=0.28,rounding_size=0.2', facecolor=exit_pill_color, edgecolor='#ffffff', linewidth=1.0),
            zorder=8
        )

        # Set Limits
        ax.set_xlim(-1, n + 6)
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), entry_px, exit_px, tp, sl]
        min_y, max_y = min(all_vals), max(all_vals)
        pad = (max_y - min_y) * 0.09
        ax.set_ylim(min_y - pad, max_y + pad)

        # Right Price Pills
        _add_price_pill(ax, entry_px, f"ENTRY: {entry_px:.2f}", CYAN_ACCENT, text_color='#0c1017')
        _add_price_pill(ax, exit_px, f"EXIT: {exit_px:.2f}", exit_pill_color, text_color=exit_text_color)
        if tp and abs(tp - exit_px) > 0.05:
            _add_price_pill(ax, tp, f"TP: {tp:.2f}", BULL_COLOR, text_color='#0c1017')

        return _render_and_close(fig)

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate trade close chart: {e}", exc_info=True)
        return None
