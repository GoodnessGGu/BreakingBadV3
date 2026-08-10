import os
import gspread
import logging
import json
from google.oauth2.service_account import Credentials
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

logger = logging.getLogger(__name__)

# Constants from environment
GOOGLE_SHEET_ID = os.getenv("GOOGLE_SHEET_ID")
SERVICE_ACCOUNT_FILE = os.getenv("SERVICE_ACCOUNT_FILE", "service_account.json")
GOOGLE_SERVICE_ACCOUNT_JSON = os.getenv("GOOGLE_SERVICE_ACCOUNT_JSON")

# 20 Rich retraining feature columns
HEADER_COLUMNS = [
    "Timestamp", "Asset", "Direction", "Amount", "Expiry", 
    "Result", "Profit", "Gale Level", "Source",
    "RSI", "ADX", "BB_Width", "ATR", "EMA_Trend",
    "Orderbook_Ratio", "Win_Rate_Gate", "Loss_Prob", "NN_Prob",
    "Entry_Latency", "Close_Price"
]

# Google Sheets Row Color Styles
COLOR_WIN = {"backgroundColor": {"red": 0.82, "green": 0.94, "blue": 0.84}}      # Soft Green
COLOR_LOSS = {"backgroundColor": {"red": 0.96, "green": 0.82, "blue": 0.82}}     # Soft Red
COLOR_EQUAL = {"backgroundColor": {"red": 0.98, "green": 0.96, "blue": 0.82}}    # Soft Yellow
COLOR_HEADER = {
    "backgroundColor": {"red": 0.15, "green": 0.25, "blue": 0.4},
    "textFormat": {"bold": True, "foregroundColor": {"red": 1.0, "green": 1.0, "blue": 1.0}}
}

class GSheetLogger:
    def __init__(self, sheet_id=GOOGLE_SHEET_ID, credentials_file=SERVICE_ACCOUNT_FILE):
        self.sheet_id = sheet_id
        self.credentials_file = credentials_file
        self.client = None
        self.spreadsheet = None
        self.worksheets = {}
        self._connected = False
        
        if self.sheet_id and (GOOGLE_SERVICE_ACCOUNT_JSON or os.path.exists(self.credentials_file)):
            self._connect()
        else:
            missing = []
            if not self.sheet_id: missing.append("GOOGLE_SHEET_ID")
            if not (GOOGLE_SERVICE_ACCOUNT_JSON or os.path.exists(self.credentials_file)): 
                missing.append("GOOGLE_SERVICE_ACCOUNT_JSON or service_account.json")
            logger.warning(f"⚠️ GSheet sync disabled. Missing: {', '.join(missing)}")

    def _connect(self):
        try:
            scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
            
            if GOOGLE_SERVICE_ACCOUNT_JSON:
                try:
                    creds_dict = json.loads(GOOGLE_SERVICE_ACCOUNT_JSON)
                    creds = Credentials.from_service_account_info(creds_dict, scopes=scope)
                    logger.info("🔑 Using Google Service Account from environment variable.")
                except json.JSONDecodeError:
                    logger.error("❌ GOOGLE_SERVICE_ACCOUNT_JSON is not valid JSON.")
                    return
            else:
                creds = Credentials.from_service_account_file(self.credentials_file, scopes=scope)
                logger.info(f"🔑 Using Google Service Account file: {self.credentials_file}")

            self.client = gspread.authorize(creds)
            self.spreadsheet = self.client.open_by_key(self.sheet_id)
            self._connected = True
            logger.info("✅ Connected to Google Sheets successfully.")
        except Exception as e:
            logger.error(f"❌ Failed to connect to Google Sheets: {e}")
            self._connected = False

    def get_or_create_worksheet(self, title):
        """Fetches an existing worksheet or creates it with header and styling."""
        if not self._connected or not self.spreadsheet:
            self._connect()
            if not self._connected:
                return None

        if title in self.worksheets:
            return self.worksheets[title]

        try:
            ws = self.spreadsheet.worksheet(title)
        except gspread.WorksheetNotFound:
            ws = self.spreadsheet.add_worksheet(title=title, rows="100", cols="20")
            ws.append_row(HEADER_COLUMNS)
            try:
                ws.format("A1:T1", COLOR_HEADER)
            except Exception as fe:
                logger.warning(f"Failed to set header formatting on {title}: {fe}")

        self.worksheets[title] = ws
        return ws

    def log_trade(self, trade_data, worksheet_name=None):
        """
        Append a detailed trade row to Google Sheets with automatic row color formatting.
        Formats: Green for WIN / VIRTUAL_WIN, Red for LOSS / VIRTUAL_LOSS.
        Routes realtime_bot trades to 'Realtime_Bot_Trades' sheet.
        """
        source = str(trade_data.get('signal_source', 'bot')).lower()
        if not worksheet_name:
            if 'realtime' in source or trade_data.get('is_realtime_bot'):
                worksheet_name = "Realtime_Bot_Trades"
            else:
                worksheet_name = "Trades"

        logger.info(f"📊 Logging trade to Google Sheet ({worksheet_name}): {trade_data.get('asset')} {trade_data.get('result')}")

        ws = self.get_or_create_worksheet(worksheet_name)
        if not ws:
            logger.warning("⚠️ Could not access Google Sheet worksheet.")
            return False

        try:
            result_str = str(trade_data.get('result', 'N/A')).upper()
            
            # Format row values matching the 20 HEADER_COLUMNS
            row = [
                trade_data.get('timestamp', datetime.now().strftime("%Y-%m-%d %H:%M:%S")),
                trade_data.get('asset', 'N/A'),
                trade_data.get('direction', 'N/A').upper(),
                trade_data.get('amount', 0),
                trade_data.get('expiry', 0),
                result_str,
                round(trade_data.get('profit', 0), 2),
                trade_data.get('gale_level', 0),
                trade_data.get('signal_source', 'bot'),
                trade_data.get('rsi', ''),
                trade_data.get('adx', ''),
                trade_data.get('bb_width', ''),
                trade_data.get('atr', ''),
                trade_data.get('ema200_diff', ''),
                trade_data.get('orderbook_ratio', ''),
                trade_data.get('recent_win_rate', ''),
                trade_data.get('loss_prob', ''),
                trade_data.get('nn_prob', ''),
                trade_data.get('entry_latency', ''),
                trade_data.get('close', trade_data.get('entry_price', ''))
            ]

            # Append row to sheet
            ws.append_row(row)
            
            # Determine row index for formatting (last row)
            row_idx = len(ws.get_all_values())
            cell_range = f"A{row_idx}:T{row_idx}"

            # Apply row background color based on win/loss
            if "WIN" in result_str:
                ws.format(cell_range, COLOR_WIN)
            elif "LOSS" in result_str:
                ws.format(cell_range, COLOR_LOSS)
            elif "EQUAL" in result_str or "DRAW" in result_str or "TIE" in result_str:
                ws.format(cell_range, COLOR_EQUAL)

            return True

        except Exception as e:
            logger.error(f"❌ Failed to log trade row to GSheets: {e}")
            if "token" in str(e).lower() or "auth" in str(e).lower():
                self._connect()
            return False

# Global instance
gsheet_logger = GSheetLogger()
