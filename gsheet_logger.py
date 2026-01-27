import os
import gspread
import logging
from oauth2client.service_account import ServiceAccountCredentials
from datetime import datetime

logger = logging.getLogger(__name__)

# Constants from environment
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")

class GSheetLogger:
    def __init__(self, sheet_id=GOOGLE_SHEET_ID, credentials_file=SERVICE_ACCOUNT_FILE):
        self.sheet_id = sheet_id
        self.credentials_file = credentials_file
        self.client = None
        self.sheet = None
        self._connected = False
        
        if self.sheet_id and os.path.exists(self.credentials_file):
            self._connect()
        else:
            logger.warning("⚠️ Google Sheets ID or Service Account file missing. GSheet sync disabled.")

    def _connect(self):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, scope)
            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(self.sheet_id)
            # Use the first sheet or create 'Trades' if it doesn't exist
            try:
                self.sheet = self.spreadsheet.worksheet("Trades")
            except gspread.WorksheetNotFound:
                self.sheet = self.spreadsheet.add_worksheet(title="Trades", rows="100", cols="10")
                # Add Header
                self.sheet.append_row([
                    "Timestamp", "Asset", "Direction", "Amount", "Expiry", 
                    "Result", "Profit", "Gale Level", "Source"
                ])
            
            self._connected = True
            logger.info("✅ Connected to Google Sheets successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Google Sheets: {e}")
            self._connected = False

    def log_trade(self, trade_data):
        """Append a trade row to Google Sheets."""
        if not self._connected:
            return False
            
        try:
            row = [
                trade_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                trade_data.get('asset', 'N/A'),
                trade_data.get('direction', 'N/A').upper(),
                trade_data.get('amount', 0),
                trade_data.get('expiry', 0),
                trade_data.get('result', 'N/A'),
                round(trade_data.get('profit', 0), 2),
                trade_data.get('gale_level', 0),
                trade_data.get('signal_source', 'bot')
            ]
            self.sheet.append_row(row)
            return True
        except Exception as e:
            logger.error(f"❌ Failed to log trade to GSheets: {e}")
            # Try to reconnect once
            if "token" in str(e).lower() or "auth" in str(e).lower():
                self._connect()
            return False

# Global instance
gsheet_logger = GSheetLogger()
