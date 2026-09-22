"""
copiers/base_copier.py - Base interface for modular channel copiers
"""

import abc
import logging
from typing import Dict, Any, Optional, Callable

logger = logging.getLogger("BaseCopier")

class BaseCopier(abc.ABC):
    def __init__(self, name: str, channel_id: int, enabled: bool = True):
        self.name = name
        self.channel_id = int(channel_id)
        self.is_enabled = enabled
        self.notify_cb: Optional[Callable] = None

    def set_notification_callback(self, cb: Callable):
        self.notify_cb = cb

    async def notify(self, message: str):
        if self.notify_cb:
            try:
                await self.notify_cb(message)
            except Exception as e:
                logger.warning(f"[{self.name}] Notification error: {e}")

    def enable(self):
        self.is_enabled = True
        logger.info(f"🟢 [{self.name}] Enabled")

    def disable(self):
        self.is_enabled = False
        logger.info(f"🔴 [{self.name}] Disabled")

    def toggle(self) -> bool:
        self.is_enabled = not self.is_enabled
        logger.info(f"🔄 [{self.name}] Toggled -> {'ENABLED' if self.is_enabled else 'DISABLED'}")
        return self.is_enabled

    @abc.abstractmethod
    async def handle_message(self, text: str, message_id: int, event: Any = None, msg_date: Any = None):
        """Process incoming message from the channel."""
        pass

    @abc.abstractmethod
    def get_status(self) -> Dict[str, Any]:
        """Return status dictionary for UI/dashboard."""
        pass
