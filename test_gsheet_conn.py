import sys
import os
from dotenv import load_dotenv

load_dotenv()

from gsheet_logger import GSheetLogger

def main():
    print("Testing Google Sheets Connection...")
    logger = GSheetLogger()
    if logger._connected:
        print(f"[SUCCESS] Connected to Google Sheet ID: {logger.sheet_id}")
        print(f"Worksheet Name: {logger.sheet.title}")
    else:
        print("[FAILED] Could not connect to Google Sheets.")
        print("\nTroubleshooting steps:")
        print("1. Replace service_account.json with a newly generated key from Google Cloud Console.")
        print("2. Ensure your Google Sheet is shared with the client_email as 'Editor'.")

if __name__ == "__main__":
    main()
