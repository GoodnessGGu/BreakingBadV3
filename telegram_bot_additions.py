# telegram_bot_additions.py
"""
This file contains all the new code to add to telegram_bot.py for the analytics features.
Copy and integrate these sections into telegram_bot.py
"""

# ========== IMPORTS TO ADD (after existing imports) ==========
from trade_database import db
from chart_generator import generate_pnl_chart, generate_winrate_chart, generate_asset_performance_chart, generate_summary_dashboard
from trade_exporter import export_to_csv, export_to_excel
from health_monitor import HealthMonitor
import os


# ========== GLOBAL VARIABLES TO ADD (after channel_monitor = None) ==========
# Initialize health monitor (will be started in post_init)
health_monitor_instance = None


# ========== UPDATE KEYBOARD IN start() FUNCTION ==========
# Replace the keyboard definition with this:
keyboard = [
    [KeyboardButton("📊 Status"), KeyboardButton("💰 Balance")],
    [KeyboardButton("⏸ Pause"), KeyboardButton("▶ Resume")],
    [KeyboardButton("📡 Channels"), KeyboardButton("⚙️ Settings")],
    [KeyboardButton("📈 Charts"), KeyboardButton("📋 History")],
    [KeyboardButton("📊 Stats"), KeyboardButton("ℹ️ Help")]
]


# ========== UPDATE help_command() FUNCTION ==========
# Replace the help message with this:
msg = (
    "ℹ️ *Bot Commands*\\n\\n"
    "🖱 *Quick Actions:*\\n"
    "Use the keyboard buttons for common tasks.\\n\\n"
    "🛠 *Configuration:*\\n"
    "`/set_amount <n>` - Set trade amount\\n"
    "`/set_account <type>` - REAL, DEMO, TOURNAMENT\\n"
    "`/set_martingale <n>` - Max martingale steps\\n"
    "`/suppress <on/off>` - Toggle signal suppression\\n"
    "`/pause` / `/resume` - Control trading\\n\\n"
    "📡 *Signals:*\\n"
    "`/signals <text>` - Parse text signals\\n"
    "Or upload a text file with signals.\\n\\n"
    "📢 *Channel Monitoring:*\\n"
    "`/channels` - Toggle channel signal monitoring\\n\\n"
    "📊 *Analytics:*\\n"
    "`/stats` - View trading statistics\\n"
    "`/history [days]` - View trade history\\n"
    "`/charts [days]` - Generate performance charts\\n"
    "`/export [days]` - Export trades to Excel\\n\\n"
    "🔧 *System:*\\n"
    "`/health` - Check bot health status"
)


# ========== UPDATE handle_message() FUNCTION ==========
# Add these cases to the if/elif chain:
elif text == "📈 Charts":
    await charts_command(update, context)
elif text == "📋 History":
    await history_command(update, context)
elif text == "📊 Stats":
    await stats_command(update, context)


# ========== NEW COMMAND HANDLERS TO ADD ==========

async def stats_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display trading statistics."""
    try:
        # Get days parameter (default 7)
        days = 7
        if context.args and context.args[0].isdigit():
            days = int(context.args[0])
        
        # Get statistics
        stats = db.get_statistics(days=days)
        best_pairs = db.get_best_pairs(days=days, limit=3)
        
        if stats['total_trades'] == 0:
            await update.message.reply_text(f"📊 No trades found in the last {days} days.")
            return
        
        # Build message
        msg = f"📊 *Trading Statistics* ({days} days)\\n\\n"
        msg += f"📈 Total Trades: {stats['total_trades']}\\n"
        msg += f"✅ Wins: {stats['wins']} | ❌ Losses: {stats['losses']}\\n"
        msg += f"🎯 Win Rate: {stats['win_rate']:.1f}%\\n"
        msg += f"💰 Total Profit: ${stats['total_profit']:.2f}\\n"
        msg += f"📊 Avg Profit/Trade: ${stats['avg_profit']:.2f}\\n\\n"
        
        if best_pairs:
            msg += "*🏆 Top Performing Assets:*\\n"
            for pair in best_pairs:
                msg += f"• {pair['asset']}: ${pair['total_profit']:.2f} ({pair['win_rate']:.0f}% WR)\\n"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to get statistics: {e}")
        await update.message.reply_text(f"❌ Error getting statistics: {e}")


async def history_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Display trade history."""
    try:
        # Get days parameter (default 7)
        days = 7
        if context.args and context.args[0].isdigit():
            days = int(context.args[0])
        
        # Get trades
        trades = db.get_trades(days=days)
        
        if not trades:
            await update.message.reply_text(f"📋 No trades found in the last {days} days.")
            return
        
        # Build message (show last 10 trades)
        msg = f"📋 *Trade History* (Last {min(10, len(trades))} of {len(trades)} trades)\\n\\n"
        
        for trade in trades[:10]:
            icon = "✅" if trade['result'] == 'WIN' else "❌" if trade['result'] == 'LOSS' else "⚠️"
            timestamp = trade['timestamp'][:16].replace('T', ' ')  # Format: YYYY-MM-DD HH:MM
            msg += f"{icon} {timestamp}\\n"
            msg += f"   {trade['asset']} {trade['direction']} | ${trade['profit']:.2f} (G{trade['gale_level']})\\n\\n"
        
        if len(trades) > 10:
            msg += f"_...and {len(trades) - 10} more trades_\\n\\n"
        
        msg += f"Use `/export {days}` to download full history"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to get history: {e}")
        await update.message.reply_text(f"❌ Error getting history: {e}")


async def charts_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Generate and send performance charts."""
    try:
        # Get days parameter (default 7)
        days = 7
        if context.args and context.args[0].isdigit():
            days = int(context.args[0])
        
        await update.message.reply_text(f"📈 Generating charts for last {days} days...")
        
        # Get data
        trades = db.get_trades(days=days)
        stats = db.get_statistics(days=days)
        best_pairs = db.get_best_pairs(days=days, limit=5)
        
        if not trades:
            await update.message.reply_text(f"❌ No trades found in the last {days} days.")
            return
        
        # Generate dashboard
        chart_path = generate_summary_dashboard(trades, stats, best_pairs)
        
        if chart_path and os.path.exists(chart_path):
            # Send chart
            with open(chart_path, 'rb') as photo:
                await update.message.reply_photo(
                    photo=photo,
                    caption=f"📊 Performance Dashboard ({days} days)"
                )
            
            # Clean up
            os.remove(chart_path)
        else:
            await update.message.reply_text("❌ Failed to generate chart")
    except Exception as e:
        logger.error(f"Failed to generate charts: {e}")
        await update.message.reply_text(f"❌ Error generating charts: {e}")


async def export_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Export trade history to Excel."""
    try:
        # Get days parameter (default 30)
        days = 30
        if context.args and context.args[0].isdigit():
            days = int(context.args[0])
        
        await update.message.reply_text(f"📊 Exporting trades from last {days} days...")
        
        # Get data
        trades = db.get_trades(days=days)
        stats = db.get_statistics(days=days)
        best_pairs = db.get_best_pairs(days=days, limit=10)
        
        if not trades:
            await update.message.reply_text(f"❌ No trades found in the last {days} days.")
            return
        
        # Export to Excel
        filepath = export_to_excel(trades, stats, best_pairs)
        
        if filepath and os.path.exists(filepath):
            # Send file
            with open(filepath, 'rb') as doc:
                await update.message.reply_document(
                    document=doc,
                    filename=os.path.basename(filepath),
                    caption=f"📊 Trade Export ({len(trades)} trades, {days} days)"
                )
            
            # Clean up
            os.remove(filepath)
        else:
            await update.message.reply_text("❌ Failed to export trades")
    except Exception as e:
        logger.error(f"Failed to export trades: {e}")
        await update.message.reply_text(f"❌ Error exporting trades: {e}")


async def health_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Check bot health status."""
    global health_monitor_instance
    
    try:
        if not health_monitor_instance:
            await update.message.reply_text("⚠️ Health monitoring not initialized")
            return
        
        health_status = await health_monitor_instance.check_health()
        
        # Build message
        msg = "🏥 *Health Check Report*\\n\\n"
        
        for check_name, check_data in health_status['checks'].items():
            icon = "✅" if check_data['healthy'] else "❌"
            name = check_name.replace('_', ' ').title()
            msg += f"{icon} {name}: {check_data['message']}\\n"
        
        overall_icon = "✅" if health_status['overall_healthy'] else "❌"
        msg += f"\\n{overall_icon} *Overall Status:* {'Healthy' if health_status['overall_healthy'] else 'Unhealthy'}"
        
        await update.message.reply_text(msg, parse_mode="Markdown")
    except Exception as e:
        logger.error(f"Failed to check health: {e}")
        await update.message.reply_text(f"❌ Error checking health: {e}")


# ========== UPDATE post_init() FUNCTION ==========
# Add this code inside post_init(), after connecting to IQ Option:

# Initialize health monitor
global health_monitor_instance
health_monitor_instance = HealthMonitor(api, app)
asyncio.create_task(health_monitor_instance.monitor_loop())
logger.info("✅ Health monitoring started")


# ========== ADD COMMAND HANDLERS IN main() FUNCTION ==========
# Add these lines after the existing command handlers:

app.add_handler(CommandHandler("stats", stats_command))
app.add_handler(CommandHandler("history", history_command))
app.add_handler(CommandHandler("charts", charts_command))
app.add_handler(CommandHandler("export", export_command))
app.add_handler(CommandHandler("health", health_command))
