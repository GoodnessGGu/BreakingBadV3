
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
from strategies import analyze_strategy

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
        self.gale_levels = {} # { "EURUSD": 0 }

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
            logger.info(f"✅ Telethon client started (listening: {channel_identifier})")

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

        api = getattr(self, 'api_instance', None) or getattr(self, 'iq_api', None) or self.api_instance

        # --- NEW: Strategy Validation & Feature Generation ---
        if config.use_ai_filter_for_channel:
            try:
                # Use timeframe matching the expiry (e.g., 300s for 5m)
                analysis_tf = 300 if expiry >= 5 else 60
                candles = api.get_candle_history(pair, 280, analysis_tf)
                
                # Fetch orderbook for confirmation (if available)
                active_id = api.market_manager.get_marginal_asset_id(pair)
                orderbook = api.message_handler.orderbook.get(active_id)
                
                strategy_signal, entry_features = analyze_strategy(candles, expiry=expiry, return_features=True, orderbook=orderbook)
                
                if strategy_signal != direction.upper():
                    # Instead of blocking immediately, we log that strategy doesn't see a signal,
                    # but we'll let the AI model (inside run_trade) decide if the probability is high enough.
                    logger.info(f"⚠️ Strategy {strategy_signal} differs from Channel {direction.upper()} for {pair}. AI will decide.")
                else:
                    logger.info(f"✅ Strategy confirms Channel Signal for {pair}.")
            except Exception as e:
                logger.error(f"Error generating features for channel signal: {e}")
                entry_features = None
        else:
            entry_features = None  # No features if filter disabled

        try:
            # Assuming run_trade is imported from iqclient
            current_gale = self.gale_levels.get(pair, 0)
            amount = config.trade_amount * (config.martingale_multiplier ** current_gale)
            
            if current_gale > 0:
                 sym = api.get_currency_symbol()
                 logger.info(f"🔄 Smart Martingale Recovery (Channel): {pair} at Gale {current_gale} ({sym}{amount:.2f})")

            # run_trade call
            result = await run_trade(
                api, pair, direction, expiry, amount, 
                notification_callback=trade_notification,
                auto_martingale=not config.smart_martingale_channel,
                features=entry_features
            )
            
            # Update state
            if config.smart_martingale_channel:
                if result['result'] == "WIN":
                    self.gale_levels[pair] = 0
                elif result['result'] == "LOSS":
                    new_gale = current_gale + 1
                    if new_gale > config.max_martingale_gales:
                        logger.warning(f"💀 Max gales reached on {pair} (Channel). Resetting.")
                        self.gale_levels[pair] = 0
                    else:
                        self.gale_levels[pair] = new_gale

        except Exception as e:
            logger.error(f"Error executing channel trade: {e}")
            result = None

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

            logger.info(f"🚀 Executing trade: {signal.get('pair')} {signal.get('direction')}")

            async def trade_notification(msg):
                if self.notification_callback:
                    await self.notification_callback(msg)

            api = getattr(self, 'api_instance', None) or getattr(self, 'iq_api', None) or self.api_instance

            # --- NEW: Strategy Validation & Feature Generation ---
            if config.use_ai_filter_for_channel:
                try:
                    # Use timeframe matching the expiry (e.g., 300s for 5m)
                    analysis_tf = 300 if signal['expiry'] >= 5 else 60
                    candles = api.get_candle_history(signal['pair'], 280, analysis_tf)
                    
                    # Fetch orderbook for confirmation (if available)
                    pair = signal.get('pair')
                    active_id = api.market_manager.get_marginal_asset_id(pair)
                    orderbook = api.message_handler.orderbook.get(active_id)
                    
                    strategy_signal, entry_features = analyze_strategy(candles, expiry=signal['expiry'], return_features=True, orderbook=orderbook)
                    
                    if strategy_signal != signal['direction'].upper():
                        logger.info(f"⚠️ Strategy {strategy_signal} differs from Channel {signal['direction'].upper()} for {signal['pair']}. AI will decide.")
                    else:
                        logger.info(f"✅ Strategy confirms Channel Signal for {signal['pair']}.")
                except Exception as e:
                    logger.error(f"Error generating features for channel signal: {e}")
                    entry_features = None
            else:
                entry_features = None

            try:
                pair = signal.get('pair')
                current_gale = self.gale_levels.get(pair, 0)
                amount = config.trade_amount * (config.martingale_multiplier ** current_gale)
                
                if current_gale > 0:
                     sym = api.get_currency_symbol()
                     logger.info(f"🔄 Smart Martingale Recovery (Channel): {pair} at Gale {current_gale} ({sym}{amount:.2f})")

                result = await run_trade(
                    api, pair, signal['direction'], signal['expiry'], amount, 
                    notification_callback=trade_notification,
                    auto_martingale=not config.smart_martingale_channel,
                    features=entry_features
                )

                # Update state
                if config.smart_martingale_channel:
                    logger.info(f"🔍 DEBUG: Channel Martingale Check. Result: {result['result']} Pair: {pair}")
                    if result['result'] == "WIN":
                        self.gale_levels[pair] = 0
                        logger.info(f"✅ Reset Gale for {pair} to 0")
                    elif result['result'] == "LOSS":
                        new_gale = current_gale + 1
                        if new_gale > config.max_martingale_gales:
                            logger.warning(f"💀 Max gales reached on {pair} (Channel). Resetting.")
                            self.gale_levels[pair] = 0
                        else:
                            self.gale_levels[pair] = new_gale
                            logger.info(f"📈 Incremented Gale for {pair} to {new_gale}")
            except Exception as e:
                logger.error(f"Error executing signal: {e}")
                result = None

            return result

        except Exception as e:
            logger.error(f"❌ Failed to execute signal: {e}")
            if self.notification_callback:
                await self.notification_callback(f"❌ Failed to execute trade: {e}")
