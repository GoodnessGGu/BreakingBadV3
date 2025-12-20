# bot_supervisor.py
import subprocess
import time
import logging
import os
import sys
from datetime import datetime
import asyncio
from telegram import Bot

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("supervisor.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Configuration
BOT_SCRIPT = "telegram_bot.py"
MAX_RESTART_ATTEMPTS = int(os.getenv("MAX_RESTART_ATTEMPTS", "5"))
RESTART_DELAY = int(os.getenv("RESTART_DELAY", "10"))
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")

class BotSupervisor:
    """Supervises the bot process and handles auto-restart on crashes."""
    
    def __init__(self):
        self.process = None
        self.restart_count = 0
        self.last_restart_time = None
        self.running = True
    
    async def send_crash_alert(self, error_info: str):
        """Send crash notification to admin."""
        if not TELEGRAM_TOKEN or not ADMIN_ID:
            logger.warning("Cannot send crash alert - missing credentials")
            return
        
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            message = (
                f"🚨 *Bot Crashed!*\n\n"
                f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔄 Restart Attempt: {self.restart_count}/{MAX_RESTART_ATTEMPTS}\n"
                f"📝 Error: {error_info}\n\n"
                f"♻️ Attempting auto-restart in {RESTART_DELAY}s..."
            )
            await bot.send_message(chat_id=int(ADMIN_ID), text=message, parse_mode="Markdown")
            logger.info("✅ Crash alert sent to admin")
        except Exception as e:
            logger.error(f"❌ Failed to send crash alert: {e}")
    
    async def send_restart_success(self):
        """Send successful restart notification."""
        if not TELEGRAM_TOKEN or not ADMIN_ID:
            return
        
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            message = (
                f"✅ *Bot Restarted Successfully*\n\n"
                f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
                f"🔄 Restart Count: {self.restart_count}\n\n"
                f"Bot is now running normally."
            )
            await bot.send_message(chat_id=int(ADMIN_ID), text=message, parse_mode="Markdown")
            logger.info("✅ Restart success notification sent")
        except Exception as e:
            logger.error(f"❌ Failed to send restart notification: {e}")
    
    def start_bot(self):
        """Start the bot process."""
        try:
            logger.info(f"🚀 Starting bot: {BOT_SCRIPT}")
            self.process = subprocess.Popen(
                [sys.executable, BOT_SCRIPT],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            logger.info(f"✅ Bot started with PID: {self.process.pid}")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to start bot: {e}")
            return False
    
    def stop_bot(self):
        """Stop the bot process gracefully."""
        if self.process:
            logger.info("⏹️ Stopping bot...")
            self.process.terminate()
            try:
                self.process.wait(timeout=10)
                logger.info("✅ Bot stopped gracefully")
            except subprocess.TimeoutExpired:
                logger.warning("⚠️ Bot didn't stop gracefully, forcing...")
                self.process.kill()
                self.process.wait()
                logger.info("✅ Bot force-stopped")
    
    async def monitor_bot(self):
        """Monitor the bot process and restart on crash."""
        logger.info("👁️ Starting bot monitoring...")
        
        # Start the bot initially
        if not self.start_bot():
            logger.error("❌ Failed to start bot initially")
            return
        
        # Send startup notification
        await self.send_restart_success()
        
        while self.running:
            try:
                # Check if process is still running
                if self.process.poll() is not None:
                    # Process has terminated
                    exit_code = self.process.returncode
                    logger.warning(f"⚠️ Bot process terminated with exit code: {exit_code}")
                    
                    # Check restart attempts
                    if self.restart_count >= MAX_RESTART_ATTEMPTS:
                        logger.error(f"❌ Max restart attempts ({MAX_RESTART_ATTEMPTS}) reached. Giving up.")
                        await self.send_crash_alert(f"Max restart attempts reached. Exit code: {exit_code}")
                        break
                    
                    # Increment restart count
                    self.restart_count += 1
                    self.last_restart_time = datetime.now()
                    
                    # Send crash alert
                    await self.send_crash_alert(f"Exit code: {exit_code}")
                    
                    # Wait before restarting
                    logger.info(f"⏳ Waiting {RESTART_DELAY}s before restart...")
                    await asyncio.sleep(RESTART_DELAY)
                    
                    # Restart the bot
                    logger.info(f"🔄 Restarting bot (attempt {self.restart_count}/{MAX_RESTART_ATTEMPTS})...")
                    if self.start_bot():
                        await self.send_restart_success()
                    else:
                        logger.error("❌ Failed to restart bot")
                        break
                
                # Wait before next check
                await asyncio.sleep(5)
                
            except KeyboardInterrupt:
                logger.info("⌨️ Keyboard interrupt received")
                self.running = False
                break
            except Exception as e:
                logger.error(f"❌ Error in monitoring loop: {e}")
                await asyncio.sleep(5)
        
        # Cleanup
        self.stop_bot()
        logger.info("👋 Supervisor shutting down")
    
    def run(self):
        """Run the supervisor."""
        try:
            asyncio.run(self.monitor_bot())
        except KeyboardInterrupt:
            logger.info("⌨️ Supervisor interrupted by user")
        except Exception as e:
            logger.error(f"❌ Supervisor error: {e}")


if __name__ == "__main__":
    logger.info("=" * 60)
    logger.info("🤖 Bot Supervisor Starting")
    logger.info(f"📝 Script: {BOT_SCRIPT}")
    logger.info(f"🔄 Max Restarts: {MAX_RESTART_ATTEMPTS}")
    logger.info(f"⏱️ Restart Delay: {RESTART_DELAY}s")
    logger.info("=" * 60)
    
    supervisor = BotSupervisor()
    supervisor.run()
