#settings.py
import os
from dotenv import load_dotenv
load_dotenv()


LOGIN_URL = 'https://api.iqoption.com/v2/login'
LOGOUT_URL = "https://auth.iqoption.com/api/v1.0/logout"
WS_URL = 'wss://ws.iqoption.com/echo/websocket'


EMAIL = os.getenv('email')
PASSWORD = os.getenv('password')


DEFAULT_ACCOUNT_TYPE = 'demo' # REAL/DEMO/real/demo

# Constants for account types
ACCOUNT_REAL = 1
ACCOUNT_TOURNAMENT = 2
ACCOUNT_DEMO = 4
ACCOUNT_CFD = 6



# Trade settings
DEFAULT_TRADE_AMOUNT = 1
MAX_MARTINGALE_GALES = 2
MARTINGALE_MULTIPLIER = 2

# Signal Suppression
SUPPRESS_OVERLAPPING_SIGNALS = True
PAUSED = False

# Timezones
TIMEZONE_MANUAL = "America/Sao_Paulo"
TIMEZONE_AUTO = "Africa/Lagos"

class TradingConfig:
    def __init__(self):
        self.trade_amount = DEFAULT_TRADE_AMOUNT
        self.max_martingale_gales = MAX_MARTINGALE_GALES
        self.martingale_multiplier = float(os.getenv("MARTINGALE_MULTIPLIER", 2.0))
        self.suppress_overlapping_signals = SUPPRESS_OVERLAPPING_SIGNALS
        self.paused = PAUSED
        self.suppress_overlapping_signals = SUPPRESS_OVERLAPPING_SIGNALS
        self.paused = PAUSED
        self.account_type = DEFAULT_ACCOUNT_TYPE
        # Optimization: AUTO, DIGITAL, BINARY
        self.preferred_trading_type = os.getenv("PREFERRED_TRADING_TYPE", "AUTO").upper()
        self.use_ai_filter = True # Filter trades with AI model?
        self.smart_martingale = False # Wait for new signal before martingale?

    def __str__(self):
        return (f"TradingConfig(amount={self.trade_amount}, "
                f"gales={self.max_martingale_gales}, "
                f"multiplier={self.martingale_multiplier}, "
                f"paused={self.paused}, "
                f"suppress={self.suppress_overlapping_signals}, "
                f"account={self.account_type})")

config = TradingConfig()

def update_env_variable(key: str, value: str):
    """Updates a variable in the .env file."""
    try:
        env_file_path = os.path.join(os.path.dirname(__file__), '.env')
        
        # Read all lines
        lines = []
        if os.path.exists(env_file_path):
            with open(env_file_path, 'r') as f:
                lines = f.readlines()
        
        key_found = False
        new_lines = []
        
        for line in lines:
            if line.strip().startswith(f"{key}="):
                new_lines.append(f"{key}={value}\n")
                key_found = True
            else:
                new_lines.append(line)
        
        if not key_found:
            if new_lines and not new_lines[-1].endswith('\n'):
                 new_lines.append('\n')
            new_lines.append(f"{key}={value}\n")
            
        with open(env_file_path, 'w') as f:
            f.writelines(new_lines)
            
        # Update os.environ as well for good measure (though config object is main source)
        os.environ[key] = value
        
    except Exception as e:
        print(f"Failed to update .env: {e}")
