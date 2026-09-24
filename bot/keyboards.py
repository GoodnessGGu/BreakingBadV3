from telegram import InlineKeyboardMarkup, InlineKeyboardButton, ReplyKeyboardMarkup, KeyboardButton
from typing import Dict, Any, List

def persistent_reply_keyboard() -> ReplyKeyboardMarkup:
    keyboard = [
        [KeyboardButton("📊 Status"), KeyboardButton("💰 Balance")],
        [KeyboardButton("🤖 Gold ICT"), KeyboardButton("📡 Channels")],
        [KeyboardButton("📜 History"), KeyboardButton("📋 Active Setups")],
        [KeyboardButton("🌐 Market Sessions"), KeyboardButton("📰 Economic News")],
        [KeyboardButton("⚙️ Risk & Sizing"), KeyboardButton("ℹ️ Help")],
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
            InlineKeyboardButton("⚙️ Risk & Sizing", callback_data="btn_settings_menu")
        ],
        [
            InlineKeyboardButton("📋 Active Setups", callback_data="btn_active_trades"),
            InlineKeyboardButton("📜 Trade History", callback_data="history_cat_all")
        ],
        [
            InlineKeyboardButton("🌐 Market Sessions", callback_data="btn_sessions"),
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
            InlineKeyboardButton(f"{'✅ ' if category == 'blitz' else ''}⚡ Blitz Options", callback_data="history_cat_blitz")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if category == 'cfd' else ''}📈 CFD Copiers", callback_data="history_cat_cfd"),
            InlineKeyboardButton(f"{'✅ ' if category == 'ict' else ''}🤖 ICT Engine", callback_data="history_cat_ict")
        ],
        [
            InlineKeyboardButton("🔄 Refresh", callback_data=f"history_cat_{category}"),
            InlineKeyboardButton("🔙 Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def active_setups_keyboard(is_live: bool = True) -> InlineKeyboardMarkup:
    status_txt = "🟢 Live (Auto 4s)" if is_live else "⚪ Static"
    keyboard = [
        [
            InlineKeyboardButton(f"🔄 {status_txt}", callback_data="btn_active_trades_refresh"),
            InlineKeyboardButton("🔙 Main Menu", callback_data="btn_main_menu")
        ]
    ]
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
    active_syms = set(ict_status.get("enabled_symbols", []))
    cur_rr = ict_status.get("rr_ratio", 2.2)
    cur_lots = ict_status.get("lots", 1.0)
    cur_lev = ict_status.get("leverage", 100)

    # Dynamic status icon helper
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
            InlineKeyboardButton(f"{s_icon('XAUUSD')} Gold (XAU)", callback_data="toggle_inst_xauusd"),
            InlineKeyboardButton(f"{s_icon('BTCUSD')} Bitcoin (BTC)", callback_data="toggle_inst_btcusd")
        ],
        [
            InlineKeyboardButton(f"{s_icon('EURUSD')} EUR/USD", callback_data="toggle_inst_eurusd"),
            InlineKeyboardButton(f"{s_icon('GBPUSD')} GBP/USD", callback_data="toggle_inst_gbpusd"),
            InlineKeyboardButton(f"{s_icon('USDJPY')} USD/JPY", callback_data="toggle_inst_usdjpy")
        ],
        [
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
