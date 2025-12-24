
import os
import asyncio
import logging
import pytz
from datetime import datetime, timedelta
from typing import Optional
from telethon import TelegramClient, events

from settings import config, TIMEZONE_AUTO
from iqclient import run_trade
from signal_parser import parse_signals_from_text
from channel_signal_parser import parse_channel_signal, is_signal_message
from strategies import confirm_trade_with_ai
from smart_trade import smart_trade_manager

logger = logging.getLogger(__name__)


class ChannelMonitor:
    """Unified ChannelMonitor that supports multiple signal formats.

    - Supports legacy `signal_parser.parse_signals_from_text` (returns list of signals
      with time strings like 'HH:MM').
    - Supports `channel_signal_parser.parse_channel_signal` (returns a dict
      containing a `datetime`-typed `time`).
    """

    def __init__(
        self,
        api_id: str,
        api_hash: str,
        api_instance=None,
        channel_id: Optional[str] = None,
        notification_callback=None,
    ):
        self.api_id = api_id
        self.api_hash = api_hash
        self.api_instance = api_instance
        self.channel_id = int(channel_id) if channel_id is not None else None
        self.notification_callback = notification_callback
        self.client: Optional[TelegramClient] = None
        self.is_running = False

    async def start(self, channel_identifier: Optional[str] = None):
        """Start monitoring a channel. If `channel_identifier` is provided it overrides
        the stored `channel_id`.
        """
        if self.is_running:
            logger.warning("⚠️ Channel monitoring is already running")
            return

        # Resolve channel id
        if channel_identifier is None and self.channel_id is None:
            logger.error("❌ No channel ID provided to start monitoring")
            return

        if channel_identifier is not None:
            try:
                if isinstance(channel_identifier, str) and channel_identifier.lstrip('-').isdigit():
                    channel_identifier = int(channel_identifier)
            except Exception:
                pass
        else:
            channel_identifier = self.channel_id

        try:
            if not self.client:
                self.client = TelegramClient('bot_session', self.api_id, self.api_hash)

            await self.client.start()
            logger.info(f"✅ Telethon client started. Resolving entity for: {channel_identifier}")

            # Force resolve entity to ensure it's in cache/known
            try:
                entity = await self.client.get_entity(channel_identifier)
                channel_name = getattr(entity, 'title', str(channel_identifier))
                logger.info(f"✅ Resolved Channel: {channel_name} (ID: {entity.id})")
            except ValueError:
                logger.warning(f"⚠️ Could not resolve entity for {channel_identifier} directly. Fetching dialogs...")
                # Fallback: Sync dialogs
                async for dialog in self.client.iter_dialogs():
                    if dialog.id == channel_identifier:
                        logger.info(f"✅ Found channel in dialogs: {dialog.title}")
                        break
            except Exception as e:
                logger.error(f"❌ Failed to resolve channel entity: {e}")
                # We proceed anyway, but it might fail to listen

            @self.client.on(events.NewMessage(chats=channel_identifier))
            async def _on_message(event):
                await self._process_message(event)

            self.is_running = True
            logger.info(f"📡 Started monitoring channel: {channel_identifier} (TZ: {TIMEZONE_AUTO})")

            if self.notification_callback:
                await self.notification_callback(
                    f"📡 *Channel Monitoring Started*\nMonitoring: `{channel_identifier}`"
                )

            await self.client.run_until_disconnected()

        except Exception as e:
            logger.error(f"❌ Failed to start channel monitoring: {e}")
            self.is_running = False
            if self.notification_callback:
                await self.notification_callback(f"❌ Failed to start monitoring: {e}")

    async def stop(self):
        """Stop monitoring the channel."""
        if not self.is_running:
            logger.warning("⚠️ Channel monitoring is not running")
            return

        try:
            self.is_running = False
            if self.client:
                await self.client.disconnect()
                logger.info("✅ Telethon client disconnected")

            logger.info("📡 Stopped monitoring channel")
            if self.notification_callback:
                await self.notification_callback("📡 *Channel Monitoring Stopped*")

        except Exception as e:
            logger.error(f"❌ Failed to stop channel monitoring: {e}")

    def is_monitoring(self) -> bool:
        return self.is_running

    async def _process_message(self, event):
        """Process incoming Telethon message event and handle signals from
        whichever parser matches the message format.
        """
        try:
            # Prefer message text fields compatibly
            message_text = None
            if hasattr(event.message, 'message'):
                message_text = event.message.message
            elif hasattr(event.message, 'text'):
                message_text = event.message.text
            else:
                # Fallback to string representation
                message_text = str(event.message)

            if not message_text:
                return

            logger.info(f"📨 Auto-Signal Received: {message_text[:80]}...")

            # If message matches channel signal parser format, use that
            if is_signal_message(message_text):
                signal = parse_channel_signal(message_text)
                if not signal:
                    logger.warning("⚠️ Failed to parse channel signal")
                    if self.notification_callback:
                        await self.notification_callback(
                            "⚠️ Signal detected but failed to parse channel format"
                        )
                    return

                # Execute parsed signal (expects signal['time'] as datetime)
                await self._execute_signal(signal)
                return

            # Otherwise try legacy parser which may return multiple signals
            signals = parse_signals_from_text(message_text)
            if not signals:
                return

            # For legacy signals (time as 'HH:MM'), schedule delayed trades
            # Configured to use LAGOS time for legacy auto-signals
            try:
                tz = pytz.timezone(TIMEZONE_AUTO) 
            except Exception:
                tz = pytz.timezone('Africa/Lagos')
                
            now_tz = datetime.now(tz)

            for sig in signals:
                try:
                    hh, mm = map(int, sig.get('time', '00:00').split(':'))
                    sched_time = now_tz.replace(hour=hh, minute=mm, second=0, microsecond=0)
                    if sched_time < now_tz:
                        sched_time += timedelta(days=1)
                    delay = (sched_time - now_tz).total_seconds()
                    
                    if delay < 0:
                         # Shouldn't happen given logic above, but just in case
                         delay = 0

                    logger.info(f"⏳ Scheduled Auto-Trade: {sig.get('pair')} {sig.get('direction')} in {int(delay)}s")
                    asyncio.create_task(self._delayed_trade(sig, delay))
                except Exception as e:
                    logger.error(f"Error scheduling legacy auto-trade: {e}")

        except Exception as e:
            logger.error(f"❌ Error processing channel message: {e}")
            if self.notification_callback:
                await self.notification_callback(f"❌ Error processing signal: {e}")

    async def _delayed_trade(self, sig, delay: float):
        if delay > 0:
            await asyncio.sleep(delay)

        pair = sig.get('pair')
        direction = sig.get('direction')
        expiry = sig.get('expiry')

        logger.info(f"🚀 Executing Auto-Trade: {pair} {direction}")

        # Build a simple notification wrapper
        async def trade_notification(msg):
            if self.notification_callback:
                await self.notification_callback(msg)

        # Try to pass a notification callback
        api = getattr(self, 'api_instance', None) or getattr(self, 'iq_api', None) or self.api_instance

        # --- Smart Martingale Integration ---
        base_amount = config.trade_amount
        trade_amount, max_gales = smart_trade_manager.get_trade_details(pair, base_amount)

        try:
            # Execute
            result = await run_trade(
                api, 
                pair, 
                direction, 
                expiry, 
                trade_amount, 
                max_gales=max_gales, 
                notification_callback=trade_notification
            )
            
            # Result Handler
            trade_outcome = result.get("result", "ERROR")
            if trade_outcome in ["WIN", "LOSS"]:
                smart_trade_manager.update_result(pair, trade_outcome)
            
        except TypeError:
            # Fallback for old run_trade signature
            result = await run_trade(api, pair, direction, expiry, trade_amount)

        return result

    async def _execute_signal(self, signal):
        """Execute a parsed `signal` where `signal['time']` is a datetime."""
        try:
            from timezone_utils import now

            if config.paused:
                logger.info("⏸️ Bot is paused, skipping trade execution")
                if self.notification_callback:
                    await self.notification_callback("⏸️ Trade skipped - Bot is paused")
                return

            current_time = now()
            entry_time = signal['time']
            delay = (entry_time - current_time).total_seconds()

            if delay > 0:
                logger.info(f"⏳ Waiting {int(delay)}s until {entry_time.strftime('%H:%M')} to execute trade")
                if self.notification_callback:
                    await self.notification_callback(f"⏳ Waiting {int(delay)}s until {entry_time.strftime('%I:%M %p')} to enter trade...")
                await asyncio.sleep(delay)
            elif delay < -60: # Allow 60s buffer for network lag, otherwise reject
                msg = f"⚠️ Signal Expired: Time {entry_time.strftime('%H:%M')} has passed. (Delay: {int(abs(delay))}s)"
                logger.warning(msg)
                if self.notification_callback:
                    await self.notification_callback(msg)
                return # Skip execution

            logger.info(f"🚀 Executing trade: {signal.get('pair')} {signal.get('direction')}")

            async def trade_notification(msg):
                if self.notification_callback:
                    await self.notification_callback(msg)

            api = getattr(self, 'api_instance', None) or getattr(self, 'iq_api', None) or self.api_instance
            
            # --- AI Check ---
            # Attempt to verify signal with AI
            try:
                # Need to use specific API methods to get candles. 
                # Assuming api object has get_candle_history (from iqclient wrapper)
                if config.use_ai_filter and hasattr(api, "get_candle_history"):
                     # Fetch 60 candles to be safe for feature calc (SMA50 etc)
                     # Using 60s timeframe as default for AI check
                     candles = api.get_candle_history(signal['pair'], 60, 60)
                     if candles:
                         is_approved = confirm_trade_with_ai(candles, signal['direction'])
                         if not is_approved:
                             msg = f"🛑 AI Filter Blocked: {signal.get('pair')} {signal.get('direction')} (Conditions Unfavorable)"
                             logger.info(msg)
                             if self.notification_callback:
                                 await self.notification_callback(msg)
                             return # STOP EXECUTION
            except Exception as e:
                logger.warning(f"⚠️ AI Check failed (proceeding anyway): {e}")
            # ----------------

            # Smart Martingale Integration
            base_amount = config.trade_amount
            trade_amount, max_gales = smart_trade_manager.get_trade_details(signal['pair'], base_amount)

            try:
                result = await run_trade(
                    api, 
                    signal['pair'], 
                    signal['direction'], 
                    signal['expiry'], 
                    trade_amount, 
                    max_gales=max_gales,
                    notification_callback=trade_notification
                )
                
                # Update Smart Martingale State
                if result:
                    trade_outcome = result.get("result", "ERROR")
                    if trade_outcome in ["WIN", "LOSS"]:
                        smart_trade_manager.update_result(signal['pair'], trade_outcome)

            except TypeError:
                logger.warning("Falling back to legacy run_trade signature (no max_gales support??)")
                result = await run_trade(api, signal['pair'], signal['direction'], signal['expiry'], config.trade_amount, notification_callback=trade_notification)

            # Notifying about entry is done inside run_trade via callback usually, 
            # but if we want an explicit entry log:
            if self.notification_callback:
                entry_msg = (
                    f"✅ *Trade Entered!*\n\n"
                    f"📊 Asset: `{signal.get('pair')}`\n"
                    f"📈 Direction: *{signal.get('direction')}*\n"
                    f"💰 Amount: ${config.trade_amount}\n"
                    f"⏳ Expiry: {signal.get('expiry')}m\n"
                    f"🕒 Entry Time: {datetime.now().strftime('%I:%M:%S %p')}"
                )
                # await self.notification_callback(entry_msg) # run_trade sends log, this might be duplicate

            return result

        except Exception as e:
            logger.error(f"❌ Failed to execute signal: {e}")
            if self.notification_callback:
                await self.notification_callback(f"❌ Failed to execute trade: {e}")
