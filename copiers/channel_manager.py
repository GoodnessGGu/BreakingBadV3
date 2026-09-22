"""
copiers/channel_manager.py - Unified Multi-Channel Telethon Router & Manager

Runs a single Telethon client session to listen to ALL active channels simultaneously.
Completely eliminates SQLite session file locking by avoiding multiple competing processes.
Allows registering new channels dynamically with zero architecture changes.
"""

import os
import sys
import time
import logging
import asyncio
from typing import Dict, Any, Optional, List, Callable
from telethon import TelegramClient, events
from telethon.sessions import StringSession
from dotenv import load_dotenv

from copiers.base_copier import BaseCopier

logger = logging.getLogger("ChannelManager")

class ChannelManager:
    def __init__(self, session_name: str = "user_desktop_session", lookback_mins: int = 10):
        load_dotenv()
        self.api_id = os.getenv("TELEGRAM_API_ID")
        self.api_hash = os.getenv("TELEGRAM_API_HASH")
        self.session_name = session_name
        self.lookback_mins = lookback_mins

        self.client: Optional[TelegramClient] = None
        self.copiers: Dict[str, BaseCopier] = {}
        self.channel_to_copier: Dict[int, BaseCopier] = {}
        self.is_running = False
        self.notify_cb: Optional[Callable] = None

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb
        for c in self.copiers.values():
            c.set_notification_callback(cb)

    def register_copier(self, copier: BaseCopier):
        """Register a channel copier into the manager."""
        self.copiers[copier.name.lower()] = copier
        self.channel_to_copier[copier.channel_id] = copier
        if self.notify_cb:
            copier.set_notification_callback(self.notify_cb)
        logger.info(f"Registered Copier: '{copier.name}' for Channel ID: {copier.channel_id}")

    def get_copier(self, name: str) -> Optional[BaseCopier]:
        return self.copiers.get(name.lower())

    def toggle_copier(self, name: str) -> bool:
        copier = self.get_copier(name)
        if copier:
            return copier.toggle()
        return False

    def list_copiers(self) -> List[Dict[str, Any]]:
        return [c.get_status() for c in self.copiers.values()]

    async def start(self):
        """Start the unified Telethon client listening to all registered channels."""
        if self.is_running:
            return

        if not self.api_id or not self.api_hash:
            logger.error("❌ TELEGRAM_API_ID or TELEGRAM_API_HASH missing in .env")
            return

        session_string = os.getenv("TELEGRAM_STRING_SESSION")
        if session_string:
            logger.info("Connecting unified Telethon listener using StringSession from environment...")
            self.client = TelegramClient(StringSession(session_string), int(self.api_id), self.api_hash)
        else:
            logger.info(f"Connecting unified Telethon listener using session file '{self.session_name}'...")
            self.client = TelegramClient(self.session_name, int(self.api_id), self.api_hash)

        await self.client.connect()

        if not await self.client.is_user_authorized():
            logger.error("❌ Telethon user session is not authorized!")
            return

        me = await self.client.get_me()
        logger.info(f"✅ Connected as: {me.first_name} (@{me.username})")

        # Startup Lookback Scan for each registered channel
        if self.lookback_mins > 0:
            await self._run_startup_lookback()

        # Listen to all channels
        channel_ids = list(self.channel_to_copier.keys())
        logger.info(f"📡 Listening to {len(channel_ids)} channels simultaneously: {channel_ids}")

        @self.client.on(events.NewMessage(chats=channel_ids))
        async def _message_handler(event):
            try:
                chat_id = event.chat_id
                copier = self.channel_to_copier.get(chat_id)
                if not copier:
                    return

                msg_text = getattr(event.message, 'message', None) or getattr(event.message, 'text', '')
                if not msg_text:
                    return

                msg_id = getattr(event.message, 'id', int(time.time()))
                msg_date = getattr(event.message, 'date', None)
                logger.info(f"📨 [{copier.name}] New message (#{msg_id}): {msg_text[:60]}...")
                await copier.handle_message(msg_text, msg_id, event, msg_date)
            except Exception as e:
                logger.error(f"Error handling channel event: {e}")

        self.is_running = True
        logger.info("🚀 ChannelManager running!")
        while self.is_running:
            try:
                await self.client.run_until_disconnected()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning(f"⚠️ Telethon update loop error ({e}). Reconnecting in 3s...")
                await asyncio.sleep(3)
                if not self.client.is_connected():
                    try:
                        await self.client.connect()
                    except Exception as ce:
                        logger.error(f"Error reconnecting Telethon: {ce}")

    async def _run_startup_lookback(self):
        """Fetch recent messages from each registered channel to catch recent setups."""
        cutoff_sec = max(self.lookback_mins, 240) * 60  # Look back at least 4 hours for active zones
        now = time.time()
        logger.info(f"🔍 Running startup scan on registered channels (last {int(cutoff_sec/60)} mins)...")

        for cid, copier in self.channel_to_copier.items():
            if not copier.is_enabled:
                continue
            try:
                msgs = await self.client.get_messages(cid, limit=50)
                recent_msgs = [
                    m for m in msgs
                    if m.date and (now - m.date.timestamp()) <= cutoff_sec
                ]
                recent_msgs.reverse() # Oldest to newest
                logger.info(f"   [{copier.name}] Found {len(recent_msgs)} recent messages within {int(cutoff_sec/60)}m")
                for m in recent_msgs:
                    txt = getattr(m, 'message', None) or getattr(m, 'text', '')
                    if txt:
                        await copier.handle_message(txt, m.id, None, m.date)
            except Exception as e:
                logger.warning(f"Startup lookback warning for {copier.name} ({cid}): {e}")

    async def stop(self):
        self.is_running = False
        if self.client:
            await self.client.disconnect()
            logger.info("Unified Telethon client disconnected.")
