"""
ICT Candlestick Setup Chart Generator
Generates sleek, dark-themed TradingView style setup charts highlighting:
- Candlestick price action
- Liquidity Sweep levels
- CISD (Market Structure Shift) levels
- FVG Entry Zones (Fair Value Gap)
- Projected Stop Loss and Take Profit
"""

import io
import logging
from typing import Optional, Dict, Any
import pandas as pd
import numpy as np

# Ensure headless Agg backend
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as patches

logger = logging.getLogger(__name__)

def generate_ict_setup_chart(
    df: pd.DataFrame,
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
    """
    Renders an in-memory PNG candlestick chart of the ICT setup.
    Returns bytes or None on error.
    """
    try:
        if df is None or len(df) < 15:
            return None

        # Slice the most recent N candles
        sub_df = df.tail(num_candles).copy().reset_index(drop=True)
        n = len(sub_df)

        fig, ax = plt.subplots(figsize=(11, 6), dpi=140)
        
        # Color Theme: TradingView Pro Dark
        bg_outer = '#131722'
        bg_inner = '#1e222d'
        grid_color = '#2a2e39'
        text_color = '#d1d4dc'
        green_candle = '#089981'
        red_candle = '#f23645'
        
        fig.patch.set_facecolor(bg_outer)
        ax.set_facecolor(bg_inner)
        ax.grid(True, color=grid_color, linestyle='--', linewidth=0.5, alpha=0.7)

        # Plot Candlesticks
        width = 0.60
        wick_width = 1.1
        
        for i in range(n):
            c_open = float(sub_df.loc[i, 'Open'])
            c_close = float(sub_df.loc[i, 'Close'])
            c_high = float(sub_df.loc[i, 'High'])
            c_low = float(sub_df.loc[i, 'Low'])
            
            is_bull = c_close >= c_open
            color = green_candle if is_bull else red_candle
            
            # Wick
            ax.plot([i, i], [c_low, c_high], color=color, linewidth=wick_width, zorder=2)
            
            # Body
            lower = min(c_open, c_close)
            height = max(abs(c_close - c_open), (c_high - c_low) * 0.01)  # small height for dojis
            rect = patches.Rectangle(
                (i - width / 2, lower), width, height,
                facecolor=color, edgecolor=color, zorder=3
            )
            ax.add_patch(rect)

        # Draw FVG Retest Zone (shaded rectangle extending into future)
        fvg_color = '#00e676' if side == 'BUY' else '#ff5252'
        fvg_alpha = 0.22
        fvg_start_idx = max(0, n - 4)
        fvg_span_width = (n - fvg_start_idx) + 6  # extend 6 bars forward for entry anticipation
        
        fvg_height = abs(fvg_high - fvg_low)
        rect_fvg = patches.Rectangle(
            (fvg_start_idx - 0.5, fvg_low),
            fvg_span_width,
            fvg_height,
            facecolor=fvg_color,
            alpha=fvg_alpha,
            edgecolor=fvg_color,
            linestyle='--',
            linewidth=1.2,
            zorder=1
        )
        ax.add_patch(rect_fvg)
        
        # FVG Label
        fvg_mid = (fvg_low + fvg_high) / 2.0
        ax.text(
            fvg_start_idx + fvg_span_width - 1, fvg_mid,
            f"  FVG Zone [{fvg_low:.2f} - {fvg_high:.2f}]",
            color='#69f0ae' if side == 'BUY' else '#ff8a80',
            fontsize=8.5,
            verticalalignment='center',
            fontweight='bold'
        )

        # Draw CISD Level (Market Structure Shift)
        ax.axhline(cisd_level, color='#00e5ff', linestyle='-.', linewidth=1.2, alpha=0.9, zorder=4)
        ax.text(0.5, cisd_level, f" CISD Shift ({cisd_level:.2f})", color='#00e5ff', fontsize=8.5, fontweight='bold', verticalalignment='bottom' if side == 'BUY' else 'top')

        # Draw Liquidity Sweep Level
        ax.axhline(sweep_level, color='#e040fb', linestyle=':', linewidth=1.3, alpha=0.9, zorder=4)
        sweep_tag = f" Low Sweep ({sweep_level:.2f})" if side == 'BUY' else f" High Sweep ({sweep_level:.2f})"
        ax.text(0.5, sweep_level, sweep_tag, color='#e040fb', fontsize=8.5, fontweight='bold', verticalalignment='top' if side == 'BUY' else 'bottom')

        # Draw Planned Stop Loss
        ax.axhline(sl, color='#ff1744', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, sl, f" SL: {sl:.2f}", color='#ff1744', fontsize=8.5, fontweight='bold', verticalalignment='top' if side == 'BUY' else 'bottom')

        # Draw Target Take Profit
        ax.axhline(tp, color='#00e676', linestyle='--', linewidth=1.4, alpha=0.9, zorder=4)
        ax.text(0.5, tp, f" Projected TP: {tp:.2f} (1:2.2 RR)", color='#00e676', fontsize=8.5, fontweight='bold', verticalalignment='bottom' if side == 'BUY' else 'top')

        # Formatting axes
        ax.set_xlim(-1, n + 6)
        
        # Calculate Y range with padding
        all_vals = [sub_df['Low'].min(), sub_df['High'].max(), sweep_level, cisd_level, fvg_low, fvg_high, sl, tp]
        min_y = min(all_vals)
        max_y = max(all_vals)
        pad_y = (max_y - min_y) * 0.08
        ax.set_ylim(min_y - pad_y, max_y + pad_y)

        # Title & Subtitle Badge
        side_badge = f"[ICT {side} SETUP]"
        ax.set_title(
            f"{side_badge}  {symbol}  ({timeframe})  |  Waiting for FVG Retest",
            color='#ffffff',
            fontsize=11.5,
            fontweight='bold',
            pad=14,
            loc='left'
        )

        ax.tick_params(colors=text_color, labelsize=8.5)
        for spine in ax.spines.values():
            spine.set_color('#30363d')

        # Adjust layout & render to buffer
        plt.tight_layout()
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight', facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close(fig)
        buf.seek(0)
        return buf.getvalue()

    except Exception as e:
        logger.error(f"[ChartGenerator] Failed to generate ICT setup chart: {e}", exc_info=True)
        return None
