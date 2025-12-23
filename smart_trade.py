import logging
from settings import config

logger = logging.getLogger(__name__)

class SmartTradeManager:
    """
    Manages the state of "Smart Martingale" (Signal-Based Martingale).
    Tracks the current martingale level for each asset across independent signals.
    """
    def __init__(self):
        self.states = {} # { "ASSET": { "level": 0, "last_result": None } }

    def get_trade_details(self, asset, base_amount):
        """
        Returns the amount and max_gales for the next trade.
        If Smart Mode is ON:
            - amount = base_amount * multiplier^level
            - max_gales = 0 (Single Shot)
        If Smart Mode is OFF:
            - amount = base_amount
            - max_gales = config.max_martingale_gales (Classic Loop)
        """
        if not config.smart_martingale:
            return base_amount, config.max_martingale_gales

        # Smart Mode Logic
        state = self.states.get(asset, {"level": 0})
        level = state["level"]
        
        # Calculate amount based on level
        if level == 0:
            amount = base_amount
        else:
            amount = base_amount * (config.martingale_multiplier ** level)
            logger.info(f"🧠 Smart Gale: Trading {asset} at Level {level} (${amount:.2f})")
            
        return amount, 0 # Always 0 gales for single shot

    def update_result(self, asset, result):
        """
        Updates the state based on trade result.
        """
        if not config.smart_martingale:
            return

        state = self.states.get(asset, {"level": 0})
        level = state["level"]

        if result == "WIN":
            if level > 0:
                logger.info(f"🧠 Smart Gale: WIN at Level {level}. Resetting to Level 0.")
            self.states[asset] = {"level": 0, "last_result": "WIN"}
            
        elif result == "LOSS":
            max_levels = config.max_martingale_gales
            
            if level < max_levels:
                new_level = level + 1
                self.states[asset] = {"level": new_level, "last_result": "LOSS"}
                logger.info(f"🧠 Smart Gale: LOSS. Saving state for next signal -> Level {new_level}")
            else:
                self.states[asset] = {"level": 0, "last_result": "RESET"}
                logger.info(f"💀 Smart Gale: Max Levels ({max_levels}) hit. Resetting to Level 0.")

# Global instance
smart_trade_manager = SmartTradeManager()
