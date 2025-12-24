# telegram_bot.py
import os
import asyncio
import logging
import time
import tempfile
from datetime import datetime, date, timedelta
from collections import defaultdict
from telegram import Update
from telegram.ext import (
    ApplicationBuilder, CommandHandler, MessageHandler,
    ContextTypes, filters
)
from telegram.request import HTTPXRequest
from telegram import ReplyKeyboardMarkup, KeyboardButton
from iqclient import IQOptionAPI, run_trade
from signal_parser import parse_signals_from_text, parse_signals_from_file
from settings import config, TIMEZONE_MANUAL, update_env_variable
from keep_alive import keep_alive
from channel_monitor import ChannelMonitor
from strategies import analyze_strategy, reload_ai_model
from collect_data import run_collection_cycle
from ml_utils import train_model
from smart_trade import smart_trade_manager
import pytz

# --- Logging ---
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logging.getLogger("httpx").setLevel(logging.WARNING) # Suppress polling noise
logger = logging.getLogger(__name__)

# --- Environment Variables ---
EMAIL = os.getenv("IQ_EMAIL")
PASSWORD = os.getenv("IQ_PASSWORD")
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")
API_ID = os.getenv("TELEGRAM_API_ID")
API_HASH = os.getenv("TELEGRAM_API_HASH")

# Support multiple channels
CHANNELS = {
    "1": os.getenv("CHANNEL_ID_1"),
    "2": os.getenv("CHANNEL_ID_2")
}
active_channel_key = "1" # Default to channel 1

# --- Start Time (for uptime reporting) ---
START_TIME = time.time()

# --- Initialize IQ Option API (without connecting) ---
api = IQOptionAPI(email=EMAIL, password=PASSWORD)
monitor = None
# Defer monitor init to async loop due to Telethon requirements

# --- Auto-Trading Tasks ---
active_auto_trades = {}  # Stores asyncio tasks: { "EURUSD": task_object }


# --- Ensure IQ Option connection ---
async def ensure_connection():
    """Ensures the API is connected before executing a command."""
    if api.check_connect():
        return

    logger.warning("🔌 IQ Option API disconnected — attempting to reconnect...")
    
    max_retries = 3
    for attempt in range(1, max_retries + 1):
        try:
            await api._connect()
            if getattr(api, "_connected", False):
                logger.info("🔁 Reconnected to IQ Option API.")
                return
        except Exception as e:
            logger.warning(f"⚠️ Connection attempt {attempt}/{max_retries} failed: {e}")
            if attempt < max_retries:
                await asyncio.sleep(2)  # Wait before retrying
    
    # If we get here, all retries failed
    raise ConnectionError("Failed to connect to IQ Option after multiple attempts. Check credentials.")

# --- Command Handlers ---
async def start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if str(update.effective_chat.id) != str(ADMIN_ID):
        await update.message.reply_text(f"⛔ Unauthorized access. Your ID is: `{update.effective_chat.id}`", parse_mode="Markdown")
        logger.warning(f"Unauthorized access attempt from ID: {update.effective_chat.id}")
        return
    
    keyboard = [
        [KeyboardButton("📊 Status"), KeyboardButton("💰 Balance")],
        [KeyboardButton("🧠 AI Toggle"), KeyboardButton("🛡️ Smart Gale"), KeyboardButton("🧠 Retrain")],
        [KeyboardButton("⏸ Pause"), KeyboardButton("▶ Resume"), KeyboardButton("⚙️ Settings")],
        [KeyboardButton("📡 Monitor"), KeyboardButton("🔄 Channel"), KeyboardButton("🔄 Mode")],
        [KeyboardButton("ℹ️ Help")]
    ]
    reply_markup = ReplyKeyboardMarkup(keyboard, resize_keyboard=True)
    
    await update.message.reply_text("🤖 *Breaking Bad Bot v3* Online!\nReady to cook. 💎", reply_markup=reply_markup, parse_mode="Markdown")

async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        "ℹ️ *IQ Bot Command Reference*\n\n"
        "🤖 *Core & Automation*\n"
        "• /start - Wake up the bot\n"
        "• /autotrade <ASSET> <TF> - Start strategy\n"
        "• /stoptrade <ASSET> - Stop strategy\n"
        "• /retrain - 🧠 Force AI Retraining\n"
        "• /toggle\\_ai <on/off> - 🧠 AI Filter\n"
        "• /smart\\_gale <on/off> - 🛡️ Signal-Based Martingale\n\n"
        "⚙️ *Configuration*\n"
        "• /set\\_amount <n> - Trade Amount\n"
        "• /set\\_martingale <n> - Max Gales\n"
        "• /settings - View current config\n"
        "• /mode <BINARY/DIGITAL> - Switch mode\n\n"
        "📡 *Signals*\n"
        "• /switch\\_channel - Toggle Monitor Channel\n"
        "• /signals <text> - Parse signal text\n"
        "• /suppress <on/off> - Prevent overlap\n\n"
        "🕹️ *Controls*\n"
        "• /pause - Stop taking trades\n"
        "• /resume - Resume trading\n"
        "• /status - System Check\n"
        "• /balance - Check Funds\n"
        "• /refill - Refill Practice\n"
        "• /shutdown - Kill Bot"
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def settings_info(update: Update, context: ContextTypes.DEFAULT_TYPE):
    msg = (
        f"⚙️ *Current Settings*\n"
        f"💵 Amount: ${config.trade_amount}\n"
        f"🔄 Max Gales: {config.max_martingale_gales}\n"
        f"✖️ Martingale Multiplier: {config.martingale_multiplier}x\n"
        f"💼 Account: {config.account_type}\n"
        f"🚫 Suppression: {'ON' if config.suppress_overlapping_signals else 'OFF'}\n"
        f"⏸️ Paused: {'YES' if config.paused else 'NO'}\n\n"
        "To change these, use the /set commands (see ℹ️ Help)."
    )
    await update.message.reply_text(msg, parse_mode="Markdown")

async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    text = update.message.text
    
    if text == "📊 Status":
        await status(update, context)
    elif text == "💰 Balance":
        await balance(update, context)
    elif text == "📡 Monitor":
        # Just show status of monitor
        mon_status = "ACTIVE" if monitor.is_running else "INACTIVE"
        curr_chan = CHANNELS.get(active_channel_key, "Unknown")
        await update.message.reply_text(f"📡 *Auto-Monitor Status*: {mon_status}\n🎧 Listening to: `{curr_chan}`", parse_mode="Markdown")
        
    elif text == "🔄 Channel":
        await switch_channel(update, context)
    elif text == "🔄 Mode":
        await toggle_mode(update, context)
        
    elif text == "🧠 AI Toggle":
        await toggle_ai(update, context)
    elif text == "🛡️ Smart Gale":
        await toggle_smart_gale(update, context)
    elif text == "🧠 Retrain":
        await retrain_command(update, context)
        
    elif text == "⏸ Pause":
        await pause_bot(update, context)
    elif text == "▶ Resume":
        await resume_bot(update, context)
        
    elif text == "⚙️ Settings":
        await settings_info(update, context)
    elif text == "ℹ️ Help":
        await help_command(update, context)
    else:
        # Ignore other text or treat as signal input if you prefer
        pass

async def balance(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await ensure_connection()
        bal = api.get_current_account_balance()
        acc_type = getattr(api, "account_mode", "unknown").capitalize()
        await update.message.reply_text(
            f"💼 *{acc_type}* Account\n💰 Balance: *${bal:.2f}*",
            parse_mode="Markdown"
        )
    except Exception as e:
        await update.message.reply_text(f"⚠️ Could not fetch balance: {e}")

async def refill(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await ensure_connection()
        api.refill_practice_balance()
        await update.message.reply_text("✅ Practice balance refilled!")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to refill balance: {e}")

async def status(update: Update, context: ContextTypes.DEFAULT_TYPE):
    try:
        await ensure_connection()
        bal = api.get_current_account_balance()
        acc_type = getattr(api, "account_mode", "unknown").capitalize()
        connected = getattr(api, "_connected", False)
        uptime_sec = int(time.time() - START_TIME)
        uptime_str = f"{uptime_sec//3600}h {(uptime_sec%3600)//60}m"

        # Fetch open positions
        open_trades = []
        try:
            positions = await api.get_open_positions()
            if positions:
                for p in positions:
                    direction = p.get('direction', 'N/A').upper()
                    asset = p.get('asset', 'N/A')
                    amount = p.get('amount', 0)
                    open_trades.append(f"{asset} ({direction}) @ ${amount}")
        except Exception as e:
            logger.warning(f"⚠️ Failed to get open positions: {e}")

        trades_info = "\n".join(open_trades) if open_trades else "No open trades."

        msg = (
            f"🔌 Connection: {'✅ Connected' if connected else '❌ Disconnected'}\n"
            f"📡 Auto-Monitor: {'✅ Running' if monitor and monitor.is_running else '❌ Off'}\n"
            f"💼 Account Type: *{acc_type}*\n"
            f"💰 Balance: *${bal:.2f}*\n\n"
            f"🕒 Uptime: {uptime_str}\n\n"
            f"⚙️ *Settings:*\n"
            f"💵 Amount: ${config.trade_amount} | 🔄 Gales: {config.max_martingale_gales}\n"
            f"⏸️ Paused: {config.paused} | 🚫 Suppress: {config.suppress_overlapping_signals}\n"
            f"🧠 AI Filter: {'ON' if config.use_ai_filter else 'OFF'}\n\n"
            f"📈 *Open Trades:*{trades_info}"
        )
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        await update.message.reply_text(f"⚠️ Failed to fetch status: {e}")

async def process_and_schedule_signals(update: Update, parsed_signals: list):
    """Schedules and executes trades based on parsed signals."""
    if not parsed_signals:
        await update.message.reply_text("⚠️ No valid signals found to process.")
        return

    # Convert time strings to datetime objects aware of timezone
    tz = pytz.timezone(TIMEZONE_MANUAL)
    now_tz = datetime.now(tz)
    
    # Process signals relative to target timezone
    processed_signals = []
    
    for sig in parsed_signals:
        hh, mm = map(int, sig["time"].split(":"))
        
        # Create a datetime for today at HH:MM in the target timezone
        sched_time = now_tz.replace(hour=hh, minute=mm, second=0, microsecond=0)
        
        # If time passed, assume next day
        if sched_time < now_tz:
            sched_time += timedelta(days=1)
            
        sig["time"] = sched_time
        processed_signals.append(sig)

    # Group signals by scheduled time
    grouped = defaultdict(list)
    for sig in processed_signals:
        grouped[sig["time"]].append(sig)

    await update.message.reply_text(f"✅ Found {len(processed_signals)} signals. Scheduling trades (Timezone: {TIMEZONE_MANUAL})...")

    all_trade_tasks = []
    for sched_time in sorted(grouped.keys()):
        # Recalculate 'now' inside loop to be precise
        now_runtime = datetime.now(tz)
        delay = (sched_time - now_runtime).total_seconds()

        if delay > 0:
            msg = f"⏳ Waiting {int(delay)}s until {sched_time.strftime('%H:%M')} for {len(grouped[sched_time])} signal(s)..."
            logger.info(msg)
            await update.message.reply_text(msg)
            await asyncio.sleep(delay)

        exec_msg = f"🚀 Executing {len(grouped[sched_time])} signal(s) at {sched_time.strftime('%H:%M')}"
        logger.info(exec_msg)
        await update.message.reply_text(exec_msg)

        async def notify(msg):
            try:
                await update.message.reply_text(msg)
            except Exception as e:
                logger.error(f"Failed to send notification: {e}")

        for s in grouped[sched_time]:
            # execute trade
            task = asyncio.create_task(run_trade(api, s["pair"], s["direction"], s["expiry"], config.trade_amount, notification_callback=notify))
            all_trade_tasks.append(task)

    # Wait for all trades to complete and generate report
    if all_trade_tasks:
        results = await asyncio.gather(*all_trade_tasks)
        
        report_lines = ["📊 *Trade Session Report*"]
        total_profit = 0.0
        wins = 0
        losses = 0

        for res in results:
            if not res: continue # Handle potential None returns if any
            
            icon = "✅" if res['result'] == "WIN" else "❌" if res['result'] == "LOSS" else "⚠️"
            
            result_text = res['result']
            if res['result'] == "ERROR" and 'error_message' in res:
                result_text = f"ERROR: {res['error_message']}"
                
            line = f"{icon} {res['asset']} {res['direction']} | {result_text} (Gale {res['gales']})"
            report_lines.append(line)
            
            if res['result'] == "WIN":
                wins += 1
                total_profit += res['profit']
            elif res['result'] == "LOSS":
                losses += 1
                total_profit += res['profit'] # profit is negative or 0 on loss

        report_lines.append(f"\n🏆 Wins: {wins} | 💀 Losses: {losses}")
        report_lines.append(f"💰 Total Profit: ${total_profit:.2f}")
        
        await update.message.reply_text("\n".join(report_lines), parse_mode="Markdown")

async def signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text(
            "⚠️ Usage: /signals followed by text or attach a file with signals."
        )
        return

    text = " ".join(context.args)
    parsed_signals = parse_signals_from_text(text)
    
    # Schedule and process signals
    asyncio.create_task(process_and_schedule_signals(update, parsed_signals))

async def handle_file(update: Update, context: ContextTypes.DEFAULT_TYPE):
    document = update.message.document
    if not document:
        return

    file = await document.get_file()
    # Use a temporary file path that is safe
    file_path = os.path.join(tempfile.gettempdir(), document.file_name)
    await file.download_to_drive(file_path)

    parsed_signals = parse_signals_from_file(file_path)
    
    # Schedule and process signals
    asyncio.create_task(process_and_schedule_signals(update, parsed_signals))

# --- Settings Commands ---
async def set_amount(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /set_amount <amount>")
        return
    try:
        amount = float(context.args[0])
        if amount < 1:
            await update.message.reply_text("⚠️ Amount must be at least 1.")
            return
        config.trade_amount = amount
        await update.message.reply_text(f"✅ Trade amount set to ${config.trade_amount}")
    except ValueError:
        await update.message.reply_text("⚠️ Invalid amount.")

async def set_account(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /set_account <real/demo>")
        return
    target_type = context.args[0].upper()
    valid_types = ['REAL', 'DEMO', 'TOURNAMENT']
    
    # Map common terms
    if target_type == 'PRACTICE': target_type = 'DEMO'

    if target_type not in valid_types:
        await update.message.reply_text(f"⚠️ Invalid account type. Use: {', '.join(valid_types)}")
        return

    try:
        await ensure_connection()
        api.switch_account(target_type)
        config.account_type = target_type # Update config to reflect change
        await update.message.reply_text(f"✅ Switched to {target_type} account.")
    except Exception as e:
        await update.message.reply_text(f"❌ Failed to switch account: {e}")

async def set_martingale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /set_martingale <count>")
        return
    try:
        count = int(context.args[0])
        if count < 0:
            await update.message.reply_text("⚠️ Count must be non-negative.")
            return
        config.max_martingale_gales = count
        await update.message.reply_text(f"✅ Max martingale gales set to {config.max_martingale_gales}")
    except ValueError:
        await update.message.reply_text("⚠️ Invalid number.")

async def switch_channel(update: Update, context: ContextTypes.DEFAULT_TYPE):
    global active_channel_key
    
    if not monitor:
        await update.message.reply_text("⚠️ Auto-Monitor not initialized (check API_ID/HASH).")
        return

    # Toggle
    new_key = "2" if active_channel_key == "1" else "1"
    new_channel = CHANNELS.get(new_key)
    
    if not new_channel:
        await update.message.reply_text(f"⚠️ Channel {new_key} not configured in .env (CHANNEL_ID_{new_key}).")
        return

    active_channel_key = new_key
    
    # Restart monitor if it was running or should run
    if monitor.is_running:
         await monitor.stop()
         await asyncio.sleep(1) # grace period
         asyncio.create_task(monitor.start(new_channel))
         await update.message.reply_text(f"🔄 Switched to Channel {new_key}: `{new_channel}` (Monitor Restarted)")
    else:
         # Just set the new target for next start
         # Or should we auto-start? Let's just switch the target.
         # But usually switch implies "listen to this now".
         asyncio.create_task(monitor.start(new_channel))
         await update.message.reply_text(f"🔄 Switched and Started Channel {new_key}: `{new_channel}`")

async def pause_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config.paused = True
    await update.message.reply_text("⏸️ Bot PAUSED. No new trades will be taken.")

async def resume_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    config.paused = False
    await update.message.reply_text("▶️ Bot RESUMED. Trading enabled.")

async def toggle_suppression(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        status = "ON" if config.suppress_overlapping_signals else "OFF"
        await update.message.reply_text(f"ℹ️ Signal suppression is currently {status}.\nUsage: /suppress <on/off>")
        return
    
    mode = context.args[0].lower()
    if mode in ['on', 'true', '1', 'yes']:
        config.suppress_overlapping_signals = True
        await update.message.reply_text("✅ Signal suppression enabled.")
    elif mode in ['off', 'false', '0', 'no']:
        config.suppress_overlapping_signals = False
        await update.message.reply_text("✅ Signal suppression disabled.")
    else:
        await update.message.reply_text("⚠️ Invalid option. Use 'on' or 'off'.")

async def toggle_ai(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        # Toggle if no args
        config.use_ai_filter = not config.use_ai_filter
        status = "ENABLED ✅" if config.use_ai_filter else "DISABLED ❌"
        await update.message.reply_text(f"🧠 AI Filter toggled: {status}")
        return
    
    mode = context.args[0].lower()
    if mode in ['on', 'true', '1', 'yes']:
        config.use_ai_filter = True
        await update.message.reply_text("✅ AI Filter ENABLED. Signals will be verified by model.")
    elif mode in ['off', 'false', '0', 'no']:
        config.use_ai_filter = False
        await update.message.reply_text("⚠️ AI Filter DISABLED. Signals will be executed blindly.")
    else:
        await update.message.reply_text("⚠️ Invalid option. Use 'on' or 'off'.")

async def toggle_mode(update: Update, context: ContextTypes.DEFAULT_TYPE):
    modes = ['AUTO', 'BINARY', 'DIGITAL']
    current = config.preferred_trading_type
    
    # Find next mode
    try:
        idx = modes.index(current)
    except ValueError:
        idx = 0 # Default to AUTO if unknown
    
    new_mode = modes[(idx + 1) % len(modes)]
    
    # Update Config
    config.preferred_trading_type = new_mode
    
    # Persist to .env
    update_env_variable("PREFERRED_TRADING_TYPE", new_mode)
    
    msg = f"🔄 Mode switched to: *{new_mode}*\n"
    if new_mode == "BINARY":
        msg += "⚡ _Fastest for OTC (Skips Digital check)_"
    elif new_mode == "DIGITAL":
        msg += "⚠️ _Digital Only (May fail on OTC)_"
    else:
        msg += "🧠 _Smart Auto-Switching_"
    
    await update.message.reply_text(msg, parse_mode="Markdown")

async def shutdown_bot(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Gracefully shuts down the bot."""
    if str(update.effective_chat.id) != str(ADMIN_ID):
        return

    await update.message.reply_text("🛑 Shutting down system... Bye!")
    logger.info("🛑 Received shutdown command. Exiting...")
    
    if monitor and monitor.is_running:
        await monitor.stop()
    
    # Give time for reply to send
    await asyncio.sleep(1)
    
    # Kill process
    os._exit(0)

# --- Auto-Trading Logic ---
async def auto_trade_loop(asset, timeframe, context, chat_id):
    """Background task that runs the strategy loop for a specific asset."""
    # Sanitize Asset Name (Common User Typo: 0TC -> OTC)
    asset = asset.replace('0TC', 'OTC')
    
    logger.info(f"🚀 Starting Auto-Trade loop for {asset} ({timeframe}s)")
    
    # Map timeframe string to seconds if needed, assuming input is seconds (e.g., 60)
    try:
        tf_seconds = int(timeframe)
    except:
        tf_seconds = 60 # Default 1m

    while True:
        try:
            if config.paused:
                await asyncio.sleep(5)
                continue

            await ensure_connection()
            
            # Fetch candles (Need enough for MA34 + buffer)
            # Assuming api.get_candles returns list of dicts
            candles = api.get_candle_history(asset, 50, tf_seconds)
            
            signal = analyze_strategy(candles)
            
            if signal:
                msg = f"🎯 Strategy Signal found for *{asset}*: *{signal}*\n🚀 Executing trade..."
                logger.info(f"🎯 Strategy Signal found for {asset}: {signal}")
                try:
                    await context.bot.send_message(chat_id=chat_id, text=msg, parse_mode="Markdown")
                except Exception as e:
                    logger.error(f"Failed to send entry message: {e}")

                async def notify_result(text):
                    try:
                        await context.bot.send_message(chat_id=chat_id, text=text)
                    except Exception as e:
                        logger.error(f"Failed to send trade result: {e}")

                # Execute trade (1 min expiry default for strategy)
                # Smart Martingale Check
                base_amount = config.trade_amount
                trade_amount, max_gales = smart_trade_manager.get_trade_details(asset, base_amount)

                # Execute
                result = await run_trade(api, asset, signal, 1, trade_amount, max_gales=max_gales, notification_callback=notify_result)
                
                # Update Smart Martingale State
                if result:
                    trade_outcome = result.get("result", "ERROR")
                    if trade_outcome in ["WIN", "LOSS"]:
                        smart_trade_manager.update_result(asset, trade_outcome)
                        
                # Wait for next candle to avoid duplicate signals on same candle
                await asyncio.sleep(tf_seconds)
            
            # Wait a bit before next check (e.g., check every 5 seconds)
            await asyncio.sleep(5)
            
        except asyncio.CancelledError:
            logger.info(f"🛑 Auto-Trade loop for {asset} stopped.")
            break
        except Exception as e:
            logger.error(f"⚠️ Error in auto-trade loop for {asset}: {e}")
            await asyncio.sleep(10)

async def start_auto_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args or len(context.args) < 2:
        await update.message.reply_text("⚠️ Usage: /autotrade <ASSET> <TIMEFRAME_SEC>\nExample: /autotrade EURUSD-OTC 60")
        return
        
    asset = context.args[0].upper().replace('0TC', 'OTC')
    timeframe = context.args[1]
    
    if asset in active_auto_trades:
        await update.message.reply_text(f"⚠️ Auto-trade already running for {asset}")
        return
        
    chat_id = update.effective_chat.id
    task = asyncio.create_task(auto_trade_loop(asset, timeframe, context, chat_id))
    active_auto_trades[asset] = task
    
    await update.message.reply_text(f"✅ Started Auto-Trade strategy for *{asset}* ({timeframe}s)", parse_mode="Markdown")

async def toggle_smart_gale(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Toggle Smart Martingale mode."""
    if not context.args:
        # Toggle if no args
        config.smart_martingale = not config.smart_martingale
        status = "ENABLED ✅" if config.smart_martingale else "DISABLED ❌"
        await update.message.reply_text(f"🧠 Smart Martingale toggled: {status}")
        return

    mode = context.args[0].lower()
    if mode in ['on', 'true', '1', 'yes']:
        config.smart_martingale = True
        await update.message.reply_text("✅ Smart Martingale ENABLED (Signal-Based Recovery).")
    elif mode in ['off', 'false', '0', 'no']:
        config.smart_martingale = False
        await update.message.reply_text("🚫 Smart Martingale DISABLED (Immediate Recovery).")
    else:
        await update.message.reply_text("⚠️ Invalid option. Use 'on' or 'off'.")

async def stop_auto_trade(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("⚠️ Usage: /stoptrade <ASSET>")
        return
        
    asset = context.args[0].upper()
    
    if asset in active_auto_trades:
        active_auto_trades[asset].cancel()
        del active_auto_trades[asset]
        await update.message.reply_text(f"🛑 Stopped Auto-Trade for {asset}")
    else:
        await update.message.reply_text(f"⚠️ No active strategy found for {asset}")

async def confirm_signals(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """
    Checks past signals against historical data to verify accuracy.
    Usage: /confirm_signals <signals_text>
    Supports format: HH:MM;ASSET;DIR;EXPIRY
    """
    if not context.args and not update.message.text:
       await update.message.reply_text("⚠️ No signals found.")
       return

    full_text = update.message.text.replace("/confirm_signals", "").strip()
    if not full_text:
        await update.message.reply_text("⚠️ No signals found.")
        return
        
    await update.message.reply_text("🔍 Analyzing signals... This may take a moment.")
    
    # Use Robust Parser
    from signal_parser import parse_signals_from_text
    signals = parse_signals_from_text(full_text)
    
    if not signals:
        await update.message.reply_text("⚠️ Could not parse any signals. Format: 00:00;PAIR;PUT;5")
        return

    results = []
    wins = 0
    losses = 0
    total = 0
    TARGET_DATE = datetime.now().date()
    # Or check if signal has full date parsing? Parser returns HH:MM usually.
    
    MAX_GALES = 2
    tz_local = pytz.timezone(TIMEZONE_MANUAL)
    
    for sig in signals:
        try:
            # Parse Time
            sig_dt_str = f"{TARGET_DATE} {sig['time']}"
            try:
                # 1. Parse Naive (Today + HH:MM)
                sig_naive = datetime.strptime(sig_dt_str, "%Y-%m-%d %H:%M")
                
                # 2. Localize to Configured Timezone (e.g. Sao Paulo)
                sig_dt = tz_local.localize(sig_naive)
                
            except ValueError:
                continue 
                
            ts = int(sig_dt.timestamp())
            
            # Map Expiry to Timeframe
            # If User says 5 -> 5 minutes -> 300s candles?
            # Or 1m candles but checked 5m later?
            # Standard: Use Expiry as Candle Size
            timeframe = sig['expiry'] * 60
            
            end_query = ts + (MAX_GALES + 2) * timeframe
            
            # Fetch Candles 
            # Logic: Try Regular, fallback to OTC if empty
            candles = api.get_candle_history(sig['pair'], 10, timeframe, end_time=end_query)
            
            if not candles:
                if "OTC" not in sig['pair']:
                    sig['pair'] += "-OTC"
                    candles = api.get_candle_history(sig['pair'], 10, timeframe, end_time=end_query)
            
            outcome = "NO DATA"
            gale_win = -1
            target_ts = ts
            
            if candles:
                for g in range(MAX_GALES + 1):
                    # Find candle opening at target_ts
                    candle = next((c for c in candles if c['from'] == target_ts), None)
                    
                    if not candle:
                        break
                    
                    open_p = candle['open']
                    close_p = candle['close']
                    
                    is_win = False
                    if sig['direction'] == 'CALL' and close_p > open_p: is_win = True
                    if sig['direction'] == 'PUT' and close_p < open_p: is_win = True
                    
                    if is_win:
                        gale_win = g
                        break
                    else:
                        target_ts += timeframe
            
            if gale_win == 0: 
                outcome = "WIN"
                wins += 1
            elif gale_win > 0: 
                outcome = f"WIN-G{gale_win}"
                wins += 1
            elif gale_win == -1 and candles: 
                outcome = "LOSS"
                losses += 1
            
            results.append(f"`{sig['time']:<5} {sig['pair']:<10} {sig['direction']:<4} -> {outcome}`")
            total += 1
            
            await asyncio.sleep(0.1)
            
        except Exception as e:
            logger.error(f"Error checking signal {sig}: {e}")
            results.append(f"{sig.get('time', '?')} Error")

    # Generate Report
    accuracy = (wins / total * 100) if total > 0 else 0
    report_header = f"📊 **Signal Check Report ({TARGET_DATE})**\nAccuracy: {accuracy:.1f}% ({wins}/{total})\n\n"
    report_body = "\n".join(results)
    
    if len(report_body) > 3800:
        report_body = report_body[:3800] + "\n...(truncated)"
        
    await update.message.reply_text(report_header + report_body, parse_mode="Markdown")

async def list_auto_trades(update: Update, context: ContextTypes.DEFAULT_TYPE):
    active = list(active_auto_trades.keys())
    msg = f"🤖 *Active Strategies:*\n{', '.join(active) if active else 'None'}"
    await update.message.reply_text(msg, parse_mode="Markdown")

# --- Auto-Retraining ---
async def retrain_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Trigger manual retraining."""
    await update.message.reply_text("🧠 Starting Auto-Retraining Sequence...\n1. Collecting Data... (This takes a minute)")
    
    # Run collection
    success = await run_collection_cycle(api)
    if not success:
        await update.message.reply_text("❌ Data collection failed.")
        return

    await update.message.reply_text("2. Training Model (Gradient Boosting)...")
    
    # Train in thread to avoid blocking bot
    loop = asyncio.get_running_loop()
    try:
        await loop.run_in_executor(None, train_model)
    except Exception as e:
        await update.message.reply_text(f"❌ Training failed: {e}")
        return
    
    await update.message.reply_text("3. Reloading Brain...")
    reload_success = reload_ai_model()
    
    if reload_success:
        await update.message.reply_text("✅ Retraining Complete! Model updated.")
    else:
        await update.message.reply_text("⚠️ Retraining finished but reload failed.")

async def start_auto_retrain_loop(context: ContextTypes.DEFAULT_TYPE):
    """Background task to retrain every 6 hours."""
    while True:
        logger.info("⏳ Auto-Retrain Timer: Waiting 6 hours...")
        await asyncio.sleep(6 * 60 * 60) # 6 hours
        
        logger.info("🧠 Auto-Retrain Triggered!")
        try:
            success = await run_collection_cycle(api)
            if success:
                loop = asyncio.get_running_loop()
                await loop.run_in_executor(None, train_model)
                reload_ai_model()
                logger.info("✅ Auto-Retrain Cycle Complete.")
        except Exception as e:
            logger.error(f"Auto-Retrain Failed: {e}")

async def connection_watchdog(context: ContextTypes.DEFAULT_TYPE):
    """Checks and restores connection periodically."""
    while True:
        await asyncio.sleep(30) # Check every 30s
        if not api.check_connect():
            logger.warning("🔍 Watchdog: Connection lost. Restoring...")
            await api.ensure_connect()

# --- Startup Notification ---
async def notify_admin_startup(app):
    """
    Notify admin on startup with account balance and info.
    """
    try:
        if not ADMIN_ID:
            logger.warning("⚠️ TELEGRAM_ADMIN_ID not set. Skipping startup notification.")
            return

        # Connection is now handled in post_init before this is called.
        bal = api.get_current_account_balance()
        acc_type = getattr(api, "account_mode", "unknown").capitalize()

        message = (
            f"🤖 *Trading Bot Online*\n"
            f"📧 Account: `{EMAIL}`\n"
            f"🔌 Connection: ✅ Connected\n"
            f"📡 Auto-Monitor: {'✅ Running' if monitor and monitor.is_running else '❌ Off'}\n"
            f"💼 Account Type: *{acc_type}*\n"
            f"💰 Balance: *${bal:.2f}*\n\n"
            f"✅ Ready to receive signals!"
        )
        await app.bot.send_message(chat_id=int(ADMIN_ID), text=message, parse_mode="Markdown")
        logger.info("✅ Startup notification sent to admin.")
    except Exception as e:
        logger.error(f"❌ Failed to send startup notification: {e}")

async def error_handler(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Log the error and be gentle with Timeouts."""
    from telegram.error import TimedOut, NetworkError
    
    try:
        raise context.error
    except (TimedOut, NetworkError):
        # Just log a warning, don't spam stack trace
        logger.warning(f"📉 Network Error/Timeout: {context.error}")
    except Exception:
        logger.error(f"❌ Exception while handling an update:", exc_info=context.error)

# --- Main Entrypoint ---
def main():
    # Resilient Request Config - Increased to handle network jitter
    t_request = HTTPXRequest(
        connection_pool_size=16, 
        read_timeout=60.0, 
        write_timeout=30.0, 
        connect_timeout=30.0
    )
    app = ApplicationBuilder().token(TELEGRAM_TOKEN).request(t_request).build()

    # Commands
    app.add_handler(CommandHandler("start", start))
    app.add_handler(CommandHandler("help", help_command))
    app.add_handler(CommandHandler("balance", balance))
    app.add_handler(CommandHandler("refill", refill))
    app.add_handler(CommandHandler("status", status))
    app.add_handler(CommandHandler("signals", signals))
    app.add_handler(MessageHandler(filters.Document.ALL, handle_file))
    
    # Text Handler for Keyboard
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Settings Commands
    app.add_handler(CommandHandler("set_amount", set_amount))
    app.add_handler(CommandHandler("set_account", set_account))
    app.add_handler(CommandHandler("set_martingale", set_martingale))
    app.add_handler(CommandHandler("pause", pause_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("resume", resume_bot))
    app.add_handler(CommandHandler("suppress", toggle_suppression))
    app.add_handler(CommandHandler("toggle_ai", toggle_ai))
    app.add_handler(CommandHandler("mode", toggle_mode))
    app.add_handler(CommandHandler("shutdown", shutdown_bot))
    
    app.add_handler(CommandHandler("autotrade", start_auto_trade))
    app.add_handler(CommandHandler("stoptrade", stop_auto_trade))
    app.add_handler(CommandHandler("confirm_signals", confirm_signals))
    app.add_handler(CommandHandler("retrain", retrain_command)) # Manual trigger
    app.add_handler(CommandHandler("smart_gale", toggle_smart_gale))

    app.add_error_handler(error_handler)

    logger.info("🌐 Initializing bot...")

    async def post_init(app):
        """Function to run after initialization and before polling starts."""
        global monitor
        try:
             # Initialize auto-monitor here (inside loop)
            if API_ID and API_HASH and not monitor:
                 try:
                     monitor = ChannelMonitor(API_ID, API_HASH, api)
                 except Exception as e:
                     logger.error(f"❌ Failed to init ChannelMonitor: {e}")

            # Initialize the bot and connect to IQ Option
            await app.bot.initialize()
            await app.bot.delete_webhook()
            logger.info("✅ Deleted old webhook before polling.")

            logger.info("📡 Connecting to IQ Option API...")
            await api._connect()
            logger.info("✅ Connected to IQ Option API.")

            # Notify admin that the bot is online
            await notify_admin_startup(app)

            # Start Auto-Monitor if configured
            default_chan = CHANNELS.get(active_channel_key)
            if monitor and default_chan:
                asyncio.create_task(monitor.start(default_chan))
            
            # Start Auto-Retraining Loop
            asyncio.create_task(start_auto_retrain_loop(None))
            logger.info("🧠 Auto-Retraining System: ONLINE") 

            # Start Connection Watchdog
            asyncio.create_task(connection_watchdog(None))
            logger.info("🛡️ Connection Watchdog: ONLINE") 

        except Exception as e:
            logger.error(f"❌ An error occurred during startup: {e}")

    app.post_init = post_init
    app.run_polling(close_loop=False)

if __name__ == "__main__":
    #keep_alive()
    main()
