# health_monitor.py
import asyncio
import logging
import time
from datetime import datetime
from telegram import Bot
import os

logger = logging.getLogger(__name__)

# Configuration
HEALTH_CHECK_INTERVAL = int(os.getenv("HEALTH_CHECK_INTERVAL", "300"))  # 5 minutes
HEARTBEAT_TIMEOUT = int(os.getenv("HEARTBEAT_TIMEOUT", "600"))  # 10 minutes
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
ADMIN_ID = os.getenv("ADMIN_ID")


class HealthMonitor:
    """Monitors bot health and sends alerts if issues detected."""
    
    def __init__(self, iq_api, telegram_app):
        self.iq_api = iq_api
        self.telegram_app = telegram_app
        self.last_heartbeat = time.time()
        self.is_healthy = True
        self.running = False
        self.alert_sent = False
    
    def update_heartbeat(self):
        """Update the last heartbeat timestamp."""
        self.last_heartbeat = time.time()
    
    async def check_health(self) -> dict:
        """
        Perform health checks on all bot components.
        
        Returns:
            Dictionary with health status
        """
        health_status = {
            'timestamp': datetime.now().isoformat(),
            'overall_healthy': True,
            'checks': {}
        }
        
        # Check 1: Heartbeat
        heartbeat_age = time.time() - self.last_heartbeat
        heartbeat_healthy = heartbeat_age < HEARTBEAT_TIMEOUT
        health_status['checks']['heartbeat'] = {
            'healthy': heartbeat_healthy,
            'age_seconds': int(heartbeat_age),
            'message': 'OK' if heartbeat_healthy else f'No heartbeat for {int(heartbeat_age)}s'
        }
        
        # Check 2: IQ Option API Connection
        try:
            iq_connected = getattr(self.iq_api, "_connected", False)
            health_status['checks']['iq_api'] = {
                'healthy': iq_connected,
                'message': 'Connected' if iq_connected else 'Disconnected'
            }
        except Exception as e:
            health_status['checks']['iq_api'] = {
                'healthy': False,
                'message': f'Error: {str(e)}'
            }
        
        # Check 3: Telegram Bot
        try:
            # Simple check - if we can access the bot, it's healthy
            telegram_healthy = self.telegram_app is not None
            health_status['checks']['telegram_bot'] = {
                'healthy': telegram_healthy,
                'message': 'Running' if telegram_healthy else 'Not running'
            }
        except Exception as e:
            health_status['checks']['telegram_bot'] = {
                'healthy': False,
                'message': f'Error: {str(e)}'
            }
        
        # Overall health
        health_status['overall_healthy'] = all(
            check['healthy'] for check in health_status['checks'].values()
        )
        
        return health_status
    
    async def send_health_alert(self, health_status: dict):
        """Send health alert to admin."""
        if not TELEGRAM_TOKEN or not ADMIN_ID:
            logger.warning("Cannot send health alert - missing credentials")
            return
        
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            
            # Build alert message
            message = "🚨 *Health Check Failed!*\n\n"
            message += f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
            message += "*Failed Checks:*\n"
            
            for check_name, check_data in health_status['checks'].items():
                if not check_data['healthy']:
                    icon = "❌"
                    message += f"{icon} {check_name.replace('_', ' ').title()}: {check_data['message']}\n"
            
            message += "\n⚠️ Bot may not be functioning properly!"
            
            await bot.send_message(chat_id=int(ADMIN_ID), text=message, parse_mode="Markdown")
            logger.info("✅ Health alert sent to admin")
            self.alert_sent = True
        except Exception as e:
            logger.error(f"❌ Failed to send health alert: {e}")
    
    async def send_recovery_notification(self):
        """Send notification when health is restored."""
        if not TELEGRAM_TOKEN or not ADMIN_ID or not self.alert_sent:
            return
        
        try:
            bot = Bot(token=TELEGRAM_TOKEN)
            message = (
                f"✅ *Health Restored*\n\n"
                f"⏰ Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                f"All systems are now functioning normally."
            )
            await bot.send_message(chat_id=int(ADMIN_ID), text=message, parse_mode="Markdown")
            logger.info("✅ Recovery notification sent")
            self.alert_sent = False
        except Exception as e:
            logger.error(f"❌ Failed to send recovery notification: {e}")
    
    async def monitor_loop(self):
        """Main monitoring loop."""
        logger.info(f"👁️ Health monitoring started (interval: {HEALTH_CHECK_INTERVAL}s)")
        self.running = True
        
        while self.running:
            try:
                # Update heartbeat
                self.update_heartbeat()
                
                # Perform health check
                health_status = await self.check_health()
                
                # Log status
                if health_status['overall_healthy']:
                    logger.debug("✅ Health check passed")
                    
                    # Send recovery notification if we previously sent an alert
                    if self.alert_sent:
                        await self.send_recovery_notification()
                else:
                    logger.warning("⚠️ Health check failed")
                    logger.warning(f"Failed checks: {health_status['checks']}")
                    
                    # Send alert if not already sent
                    if not self.alert_sent:
                        await self.send_health_alert(health_status)
                
                # Update health status
                self.is_healthy = health_status['overall_healthy']
                
                # Wait for next check
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
                
            except Exception as e:
                logger.error(f"❌ Error in health monitoring loop: {e}")
                await asyncio.sleep(HEALTH_CHECK_INTERVAL)
    
    def stop(self):
        """Stop the health monitor."""
        logger.info("⏹️ Stopping health monitor...")
        self.running = False


# Global health monitor instance (will be initialized in telegram_bot.py)
health_monitor = None
