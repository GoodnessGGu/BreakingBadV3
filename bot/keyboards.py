"""
bot/keyboards.py - Inline Keyboard Layouts for Telegram Controller
"""

from telegram import InlineKeyboardMarkup, InlineKeyboardButton
from typing import Dict, Any, List

def main_menu_keyboard() -> InlineKeyboardMarkup:
    keyboard = [
        [
            InlineKeyboardButton("📊 Balance & Status", callback_data="btn_status"),
            InlineKeyboardButton("📡 Channels Menu", callback_data="btn_channels_menu")
        ],
        [
            InlineKeyboardButton("🤖 ICT Strategy Engine", callback_data="btn_ict_menu"),
            InlineKeyboardButton("⚙️ Account Settings", callback_data="btn_account_menu")
        ],
        [
            InlineKeyboardButton("📋 Active Setups & Trades", callback_data="btn_active_trades"),
            InlineKeyboardButton("🛑 EMERGENCY CLOSE ALL", callback_data="btn_close_all_confirm")
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
    cur_sym = ict_status.get("symbol", "XAUUSD")
    cur_rr = ict_status.get("rr_ratio", 2.0)

    keyboard = [
        [
            InlineKeyboardButton(
                f"{icon} ICT Engine: {status_txt}",
                callback_data="toggle_ict_engine"
            )
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_sym == 'XAUUSD' else ''}Gold", callback_data="set_inst_xauusd"),
            InlineKeyboardButton(f"{'✅ ' if cur_sym == 'EURUSD' else ''}EUR/USD", callback_data="set_inst_eurusd"),
            InlineKeyboardButton(f"{'✅ ' if cur_sym == 'GBPUSD' else ''}GBP/USD", callback_data="set_inst_gbpusd")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_sym == 'USDJPY' else ''}USD/JPY", callback_data="set_inst_usdjpy"),
            InlineKeyboardButton(f"{'✅ ' if cur_sym == 'AUDUSD' else ''}AUD/USD", callback_data="set_inst_audusd")
        ],
        [
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 1.5 else ''}1:1.5 RR", callback_data="set_rr_1.5"),
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 2.0 else ''}1:2.0 RR", callback_data="set_rr_2.0"),
            InlineKeyboardButton(f"{'✅ ' if cur_rr == 3.0 else ''}1:3.0 RR", callback_data="set_rr_3.0")
        ],
        [
            InlineKeyboardButton("🔙 Back to Main Menu", callback_data="btn_main_menu")
        ]
    ]
    return InlineKeyboardMarkup(keyboard)

def account_menu_keyboard(account_type: str) -> InlineKeyboardMarkup:
    is_training = account_type.lower() == "training"
    keyboard = [
        [
            InlineKeyboardButton(
                f"{'✅ ' if is_training else ''}Practice Account (Training)",
                callback_data="set_acc_training"
            )
        ],
        [
            InlineKeyboardButton(
                f"{'✅ ' if not is_training else ''}Real Money Account (Regular)",
                callback_data="set_acc_regular"
            )
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
