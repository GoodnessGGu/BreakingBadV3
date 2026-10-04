"""
bot/keyboards.py - Interactive Keyboards for BreakingBad V3 Bot Controller
"""

from telegram import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from typing import Dict, Any, List, Optional


def persistent_reply_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("📊 Status"), KeyboardButton("💰 Balance")],
        [KeyboardButton("🤖 Gold ICT"), KeyboardButton("🕯️ CRT Engine"), KeyboardButton("📐 Trendlines")],
        [KeyboardButton("🏛️ S&D Engine"), KeyboardButton("🎯 Mission & Strategies")],
        [KeyboardButton("📡 Channels"), KeyboardButton("📋 Active Setups")],
        [KeyboardButton("🏆 Prop Firm 5K"), KeyboardButton("📜 History"), KeyboardButton("⚙️ Risk & Sizing")],
        [KeyboardButton("🌐 Market Sessions"), KeyboardButton("📰 Economic News")],
        [KeyboardButton("⏸ Pause"), KeyboardButton("▶ Resume")],
        [KeyboardButton("🛑 Close All")]
    ]
    return ReplyKeyboardMarkup(keyboard, resize_keyboard=True)


def main_menu_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("📊 Balance & Status", callback_data="btn_status"),
            InlineKeyboardButton("📡 Channels Menu", callback_data="btn_channels_menu")
        ],
        [
            InlineKeyboardButton("🤖 ICT Multi-Engine", callback_data="btn_ict_menu"),
            InlineKeyboardButton("🕯️ CRT Engine", callback_data="btn_crt_menu")
        ],
        [
            InlineKeyboardButton("📐 Trendline Engine", callback_data="btn_trendline_menu"),
            InlineKeyboardButton("🏛️ S&D Imbalance Engine", callback_data="btn_snd_menu")
        ],
        [
            InlineKeyboardButton("🎯 Mission & Strategies", callback_data="btn_mission_strategies"),
            InlineKeyboardButton("📋 Active Setups", callback_data="btn_active_trades")
        ],
        [
            InlineKeyboardButton("🏆 Prop Firm 5K Sim", callback_data="btn_prop_menu"),
            InlineKeyboardButton("📜 Trade History", callback_data="history_cat_all")
        ],
        [
            InlineKeyboardButton("⚙️ Risk & Sizing", callback_data="btn_settings_menu"),
            InlineKeyboardButton("🌐 Market Sessions", callback_data="btn_sessions")
        ],
        [
            InlineKeyboardButton("📰 Economic News", callback_data="btn_news")
        ],
        [
            InlineKeyboardButton("🛑 EMERGENCY CLOSE ALL", callback_data="btn_close_all_confirm")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def history_menu_keyboard(category: str = "all") -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton(f"{'✅ ' if category == 'all' else ''}📊 All Trades", callback_data="history_cat_all"),
            InlineKeyboardButton(f"{'✅ ' if category == 'cfd' else ''}📈 CFD Copiers", callback_data="history_cat_cfd")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if category == 'ict' else ''}🤖 ICT Engine", callback_data="history_cat_ict"),
            InlineKeyboardButton(f"{'✅ ' if category == 'crt' else ''}🕯️ CRT Engine", callback_data="history_cat_crt")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if category == 'trendline' else ''}📐 Trendlines", callback_data="history_cat_trendline"),
            InlineKeyboardButton(f"{'✅ ' if category == 'blitz' else ''}⚡ Blitz Options", callback_data="history_cat_blitz")
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data=f"history_cat_{category}"),
            InlineKeyboardButton("🔙 Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def active_setups_keyboard(is_live: bool = True, open_positions: Optional[List[Dict[str, Any]]] = None) -> InlineKeyboardMarkup:
    status_txt = "🟢 Live (Auto 4s)" if is_live else "⚪ Static"
    keyboard = []

    if open_positions:
        for p in open_positions:
            pid = p.get("position_id") or p.get("id")
            sym = p.get("symbol") or p.get("asset_name") or f"Asset {p.get('asset_id')}"
            side = str(p.get("side") or p.get("type", "BUY")).upper()
            if side == "LONG": side = "BUY"
            if side == "SHORT": side = "SELL"
            pnl_val = p.get("pnl")
            pnl_txt = ""
            if pnl_val is not None:
                pnl_sign = "+" if pnl_val > 0 else ("" if pnl_val == 0 else "-")
                pnl_txt = f" ({pnl_sign}${abs(pnl_val):.2f})"
            keyboard.append([
                InlineKeyboardButton(
                    f"❌ Close {sym} {side} #{pid}{pnl_txt}",
                    callback_data=f"close_trade_{pid}"
                )
            ])
        keyboard.append([
            InlineKeyboardButton("🛑 Close ALL Positions", callback_data="btn_close_all_confirm")
        ])

    keyboard.append([
        InlineKeyboardButton(f"🔄 {status_txt}", callback_data="btn_active_trades_refresh"),
        InlineKeyboardButton("🔙 Main Menu", callback_data="btn_main_menu")
    ])
    return InlineKeyboardMarkup(keyboard)


def individual_close_keyboard(open_positions: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    keyboard = []
    for p in open_positions:
        pid = p.get("position_id") or p.get("id")
        sym = p.get("symbol") or p.get("asset_name") or f"Asset {p.get('asset_id')}"
        side = str(p.get("side") or p.get("type", "BUY")).upper()
        if side == "LONG": side = "BUY"
        if side == "SHORT": side = "SELL"
        pnl_val = p.get("pnl")
        pnl_txt = ""
        if pnl_val is not None:
            pnl_sign = "+" if pnl_val > 0 else ("" if pnl_val == 0 else "-")
            pnl_txt = f" ({pnl_sign}${abs(pnl_val):.2f})"
        keyboard.append([
            InlineKeyboardButton(
                f"❌ Close {sym} {side} #{pid}{pnl_txt}",
                callback_data=f"close_trade_{pid}"
            )
        ])
    if open_positions:
        keyboard.append([
            InlineKeyboardButton("⚠️ Emergency Close ALL", callback_data="btn_close_all_confirm")
        ])
    keyboard.append([
        InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
    ])
    return InlineKeyboardMarkup(keyboard)


def channels_menu_keyboard(copiers_status: List[Dict[str, Any]]) -> InlineKeyboardMarkup:
    keyboard = []
    for c in copiers_status:
        name = c["name"]
        enabled = c["enabled"]
        icon = "🟢" if enabled else "🔴"
        status_txt = "ON" if enabled else "OFF"
        keyboard.append([
            InlineKeyboardButton(
                f"{icon} {name}: {status_txt}",
                callback_data=f"toggle_channel_{name.lower()}"
            )
        ])
    keyboard.append([InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")])
    return InlineKeyboardMarkup(keyboard)


def ict_menu_keyboard(ict_status: Dict[str, Any]) -> InlineKeyboardMarkup:
    enabled = ict_status.get("enabled", False)
    icon = "🟢" if enabled else "🔴"
    status_txt = "ON" if enabled else "OFF"
    hybrid_on = ict_status.get("use_hybrid_trailing", True)
    hybrid_icon = "🟢" if hybrid_on else "🔴"
    hybrid_txt = "ON" if hybrid_on else "OFF"
    active_syms = set(ict_status.get("enabled_symbols", []))
    cur_rr = ict_status.get("rr_ratio", 2.2)
    cur_lots = ict_status.get("lots", 1.0)
    cur_lev = ict_status.get("leverage", 100)

    def s_icon(sym: str) -> str:
        return "🟢" if sym in active_syms else "⚪"

    keyboard = [
        [
            InlineKeyboardButton(
                f"{icon} ICT Master Switch: {status_txt}",
                callback_data="toggle_ict_engine"
            )
        ],
        [
            InlineKeyboardButton(
                f"{hybrid_icon} Hybrid Profit Lock: {hybrid_txt}",
                callback_data="toggle_ict_hybrid_trail"
            )
        ],
        [
            InlineKeyboardButton(f"{s_icon('XAUUSD')} Gold (XAU)", callback_data="toggle_inst_xauusd"),
            InlineKeyboardButton(f"{s_icon('XAGUSD')} Silver (XAG)", callback_data="toggle_inst_xagusd")
        ],
        [
            InlineKeyboardButton(f"{s_icon('BTCUSD')} Bitcoin (BTC)", callback_data="toggle_inst_btcusd"),
            InlineKeyboardButton(f"{s_icon('EURUSD')} EUR/USD", callback_data="toggle_inst_eurusd")
        ],
        [
            InlineKeyboardButton(f"{s_icon('GBPUSD')} GBP/USD", callback_data="toggle_inst_gbpusd"),
            InlineKeyboardButton(f"{s_icon('USDJPY')} USD/JPY", callback_data="toggle_inst_usdjpy"),
            InlineKeyboardButton(f"{s_icon('AUDUSD')} AUD/USD", callback_data="toggle_inst_audusd")
        ],
        [
            InlineKeyboardButton(f"📊 Lot Size: {cur_lots:.2f}", callback_data="noop_ict_lots")
        ],
        [
            InlineKeyboardButton("➖ 0.1", callback_data="ict_lots_minus"),
            InlineKeyboardButton("0.1", callback_data="set_ict_lots_0.1"),
            InlineKeyboardButton("0.5", callback_data="set_ict_lots_0.5"),
            InlineKeyboardButton("1.0", callback_data="set_ict_lots_1.0"),
            InlineKeyboardButton("2.0", callback_data="set_ict_lots_2.0"),
            InlineKeyboardButton("➕ 0.1", callback_data="ict_lots_plus")
        ],
        [
            InlineKeyboardButton(f"⚡ Leverage: {cur_lev}x", callback_data="noop_ict_lev")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 20 else ''}20x", callback_data="set_ict_lev_20"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 50 else ''}50x", callback_data="set_ict_lev_50"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 100 else ''}100x", callback_data="set_ict_lev_100"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 200 else ''}200x", callback_data="set_ict_lev_200")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 1.5 else ''}1:1.5 RR", callback_data="set_rr_1.5"),
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 2.0 else ''}1:2.0 RR", callback_data="set_rr_2.0"),
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 2.2 else ''}1:2.2 RR", callback_data="set_rr_2.2"),
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 3.0 else ''}1:3.0 RR", callback_data="set_rr_3.0")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def crt_menu_keyboard(crt_status: Dict[str, Any]) -> InlineKeyboardMarkup:
    """Dedicated interactive keyboard for Candle Range Theory (CRT) Engine."""
    enabled = crt_status.get("enabled", False)
    icon = "🟢" if enabled else "🔴"
    status_txt = "ON" if enabled else "OFF"
    active_syms = set(crt_status.get("enabled_symbols", []))
    cur_lots = crt_status.get("lots", 1.0)
    cur_lev = crt_status.get("leverage", 100)

    def s_icon(sym: str) -> str:
        return "🟢" if sym in active_syms else "⚪"

    keyboard = [
        [
            InlineKeyboardButton(
                f"{icon} CRT Master Switch: {status_txt}",
                callback_data="toggle_crt_master"
            )
        ],
        [
            InlineKeyboardButton(f"{s_icon('EURUSD')} EUR/USD (Asian Judas)", callback_data="toggle_crt_eurusd"),
            InlineKeyboardButton(f"{s_icon('GBPUSD')} GBP/USD (Asian Judas)", callback_data="toggle_crt_gbpusd")
        ],
        [
            InlineKeyboardButton(f"{s_icon('BTCUSD')} Bitcoin (H1 Anchor)", callback_data="toggle_crt_btcusd")
        ],
        [
            InlineKeyboardButton(f"📊 Lot Size: {cur_lots:.2f}", callback_data="noop_crt_lots")
        ],
        [
            InlineKeyboardButton("➖ 0.1", callback_data="crt_lots_minus"),
            InlineKeyboardButton("0.1", callback_data="set_crt_lots_0.1"),
            InlineKeyboardButton("0.5", callback_data="set_crt_lots_0.5"),
            InlineKeyboardButton("1.0", callback_data="set_crt_lots_1.0"),
            InlineKeyboardButton("2.0", callback_data="set_crt_lots_2.0"),
            InlineKeyboardButton("➕ 0.1", callback_data="crt_lots_plus")
        ],
        [
            InlineKeyboardButton(f"⚡ Leverage: {cur_lev}x", callback_data="noop_crt_lev")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 20 else ''}20x", callback_data="set_crt_lev_20"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 50 else ''}50x", callback_data="set_crt_lev_50"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 100 else ''}100x", callback_data="set_crt_lev_100")
        ],
        [
            InlineKeyboardButton("🎯 Mission & Logic", callback_data="btn_mission_strategies"),
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def trendline_menu_keyboard(trendline_status: Dict[str, Any]) -> InlineKeyboardMarkup:
    """Dedicated interactive keyboard for 15M Trendline Bounce Engine."""
    enabled = trendline_status.get("enabled", False)
    icon = "🟢" if enabled else "🔴"
    status_txt = "ON" if enabled else "OFF"
    active_syms = set(trendline_status.get("enabled_symbols", []))
    cur_lots = trendline_status.get("lots", 2.0)
    cur_lev = trendline_status.get("leverage", 100)

    def s_icon(sym: str) -> str:
        return "🟢" if sym in active_syms else "⚪"

    keyboard = [
        [
            InlineKeyboardButton(
                f"{icon} Trendline Master Switch: {status_txt}",
                callback_data="toggle_trendline_master"
            )
        ],
        [
            InlineKeyboardButton(f"{s_icon('XAUUSD')} Gold (XAUUSD)", callback_data="toggle_trendline_xauusd"),
            InlineKeyboardButton(f"{s_icon('NAS100')} US Tech 100 (NAS100)", callback_data="toggle_trendline_nas100")
        ],
        [
            InlineKeyboardButton(f"{s_icon('BTCUSD')} Bitcoin (BTCUSD)", callback_data="toggle_trendline_btcusd"),
            InlineKeyboardButton(f"{s_icon('EURUSD')} EUR/USD", callback_data="toggle_trendline_eurusd")
        ],
        [
            InlineKeyboardButton(f"⚙️ Lot Sizing: {cur_lots} Lots (Dual)", callback_data="btn_settings_menu"),
            InlineKeyboardButton(f"⚡ Lev: {cur_lev}x", callback_data="btn_settings_menu")
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data="btn_trendline_menu"),
            InlineKeyboardButton("🔙 Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def snd_menu_keyboard(snd_status: Dict[str, Any]) -> InlineKeyboardMarkup:
    """Dedicated interactive keyboard for Supply & Demand Imbalance Engine (Config E)."""
    enabled = snd_status.get("enabled", False)
    icon = "🟢" if enabled else "🔴"
    status_txt = "ON" if enabled else "OFF"
    active_syms = set(snd_status.get("enabled_symbols", []))
    cur_lots = snd_status.get("lots", 1.0)
    cur_lev = snd_status.get("leverage", 100)

    def s_icon(sym: str) -> str:
        return "🟢" if sym in active_syms else "⚪"

    keyboard = [
        [
            InlineKeyboardButton(
                f"{icon} S&D Master Switch: {status_txt}",
                callback_data="toggle_snd_master"
            )
        ],
        [
            InlineKeyboardButton(f"{s_icon('NZDUSD')} NZD/USD (+27.0R)", callback_data="toggle_snd_nzdusd"),
            InlineKeyboardButton(f"{s_icon('USDJPY')} USD/JPY (+17.0R)", callback_data="toggle_snd_usdjpy")
        ],
        [
            InlineKeyboardButton(f"{s_icon('AUDUSD')} AUD/USD (+15.0R)", callback_data="toggle_snd_audusd"),
            InlineKeyboardButton(f"{s_icon('USDCAD')} USD/CAD (+7.0R)", callback_data="toggle_snd_usdcad")
        ],
        [
            InlineKeyboardButton(f"📊 Lot Size: {cur_lots:.2f}", callback_data="noop_snd_lots")
        ],
        [
            InlineKeyboardButton("➖ 0.1", callback_data="snd_lots_minus"),
            InlineKeyboardButton("0.1", callback_data="set_snd_lots_0.1"),
            InlineKeyboardButton("0.5", callback_data="set_snd_lots_0.5"),
            InlineKeyboardButton("1.0", callback_data="set_snd_lots_1.0"),
            InlineKeyboardButton("2.0", callback_data="set_snd_lots_2.0"),
            InlineKeyboardButton("➕ 0.1", callback_data="snd_lots_plus")
        ],
        [
            InlineKeyboardButton(f"⚡ Leverage: {cur_lev}x", callback_data="noop_snd_lev")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 20 else ''}20x", callback_data="set_snd_lev_20"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 50 else ''}50x", callback_data="set_snd_lev_50"),
            InlineKeyboardButton(f"{'✅ ' if cur_lev == 100 else ''}100x", callback_data="set_snd_lev_100")
        ],
        [
            InlineKeyboardButton("🎯 Strategy Mission & Rules", callback_data="btn_mission_strategies"),
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def mission_menu_keyboard() -> InlineKeyboardMarkup:
    """Navigation keyboard for the Mission & Strategy Description Center."""
    keyboard = [
        [
            InlineKeyboardButton("🤖 ICT Engine (Metals)", callback_data="btn_ict_menu"),
            InlineKeyboardButton("🕯️ CRT Engine (Forex/BTC)", callback_data="btn_crt_menu")
        ],
        [
            InlineKeyboardButton("🏛️ S&D Engine (Config E)", callback_data="btn_snd_menu"),
            InlineKeyboardButton("⚡ News Straddle (NFP)", callback_data="btn_news")
        ],
        [
            InlineKeyboardButton("📊 Balance & Status", callback_data="btn_status"),
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def settings_menu_keyboard(account_type: str, lots: float, leverage: int, blitz_stake: float) -> InlineKeyboardMarkup:
    is_training = account_type.lower() == "training"
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if is_training else ''}Practice (Training)",
                callback_data="set_acc_training"
            ),
            InlineKeyboardButton(
                f"{'✅ ' if not is_training else ''}Real (Regular)",
                callback_data="set_acc_regular"
            )
        ],
        [
            InlineKeyboardButton(f"⚡ Blitz Base Stake: ${blitz_stake:.2f}", callback_data="noop_stake")
        ],
        [
            InlineKeyboardButton("➖ $1", callback_data="stake_minus"),
            InlineKeyboardButton("$1.00", callback_data="set_stake_1"),
            InlineKeyboardButton("$2.00", callback_data="set_stake_2"),
            InlineKeyboardButton("$5.00", callback_data="set_stake_5"),
            InlineKeyboardButton("$10.00", callback_data="set_stake_10"),
            InlineKeyboardButton("➕ $1", callback_data="stake_plus")
        ],
        [
            InlineKeyboardButton(f"📊 Global Lots: {lots:.2f} | Lev: {leverage}x", callback_data="noop_fx")
        ],
        [
            InlineKeyboardButton("➖ 0.1", callback_data="global_lots_minus"),
            InlineKeyboardButton("0.1", callback_data="set_global_lots_0.1"),
            InlineKeyboardButton("0.5", callback_data="set_global_lots_0.5"),
            InlineKeyboardButton("1.0", callback_data="set_global_lots_1.0"),
            InlineKeyboardButton("2.0", callback_data="set_global_lots_2.0"),
            InlineKeyboardButton("➕ 0.1", callback_data="global_lots_plus")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if leverage == 20 else ''}20x", callback_data="set_global_lev_20"),
            InlineKeyboardButton(f"{'✅ ' if leverage == 50 else ''}50x", callback_data="set_global_lev_50"),
            InlineKeyboardButton(f"{'✅ ' if leverage == 100 else ''}100x", callback_data="set_global_lev_100"),
            InlineKeyboardButton(f"{'✅ ' if leverage == 200 else ''}200x", callback_data="set_global_lev_200")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def news_menu_keyboard(is_straddle_enabled: bool = False, is_shield_enabled: bool = True) -> InlineKeyboardMarkup:
    straddle_icon = "🟢" if is_straddle_enabled else "🔴"
    straddle_txt = "ON" if is_straddle_enabled else "OFF"
    shield_icon = "🟢" if is_shield_enabled else "🔴"
    shield_txt = "ON" if is_shield_enabled else "OFF"
    keyboard = [
        [
            InlineKeyboardButton(f"{straddle_icon} ⚡ NFP/News Straddle Engine: {straddle_txt}", callback_data="toggle_news_straddle")
        ],
        [
            InlineKeyboardButton(f"{shield_icon} 🛡️ News Freeze Shield: {shield_txt}", callback_data="toggle_news_shield"),
            InlineKeyboardButton("🔄 Refresh Calendar", callback_data="btn_news_refresh")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def close_all_confirm_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("⚠️ YES, CLOSE ALL TRADES NOW", callback_data="action_close_all_execute")
        ],
        [
            InlineKeyboardButton("❌ CANCEL", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)


def prop_firm_menu_keyboard(is_enabled: bool = False, phase: int = 1, risk_pct: float = 0.75) -> InlineKeyboardMarkup:
    status_icon = "🟢" if is_enabled else "🔴"
    status_text = "ENABLED" if is_enabled else "DISABLED"
    
    keyboard = [
        [
            InlineKeyboardButton(f"{status_icon} Prop Firm Guard: {status_text}", callback_data="toggle_prop_master")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if phase == 1 else ''}Phase 1 (+8%)", callback_data="set_prop_phase_1"),
            InlineKeyboardButton(f"{'✅ ' if phase == 2 else ''}Phase 2 (+5%)", callback_data="set_prop_phase_2")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if abs(risk_pct - 0.5) < 0.05 else ''}0.50% ($25)", callback_data="set_prop_risk_0.5"),
            InlineKeyboardButton(f"{'✅ ' if abs(risk_pct - 0.75) < 0.05 else ''}0.75% ($37.5)", callback_data="set_prop_risk_0.75"),
            InlineKeyboardButton(f"{'✅ ' if abs(risk_pct - 1.0) < 0.05 else ''}1.00% ($50)", callback_data="set_prop_risk_1.0")
        ],
        [
            InlineKeyboardButton("🔄 Refresh Dashboard", callback_data="btn_prop_menu"),
            InlineKeyboardButton("⚠️ Reset $5K Challenge", callback_data="reset_prop_challenge")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

