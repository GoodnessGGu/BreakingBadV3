import os
import sys
from dotenv import load_dotenv
load_dotenv()
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

print("--- DEBUGGING MONITOR ---")
try:
    from channel_monitor import ChannelMonitor
    print("Import successful")
except Exception as e:
    print(f"Import Failed: {e}")
    sys.exit(1)

api_id = os.getenv("TELEGRAM_API_ID")
api_hash = os.getenv("TELEGRAM_API_HASH")

print(f"API_ID from env: '{api_id}' (Type: {type(api_id)})")
print(f"API_HASH from env: '{api_hash}' (Type: {type(api_hash)})")

if not api_id or not api_hash:
    print("X Credentials missing in .env")
else:
    print("V Credentials found")

print("Attempting to init ChannelMonitor...")
try:
    # Pass None for api_instance as we are just testing init
    monitor = ChannelMonitor(api_id, api_hash, api_instance=None)
    print("V ChannelMonitor initialized successfully")
    print(f"Monitor attributes: {monitor.__dict__}")
except Exception as e:
    print(f"X Initialization Failed: {e}")
    import traceback
    traceback.print_exc()
