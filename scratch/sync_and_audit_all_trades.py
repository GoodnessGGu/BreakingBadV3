import sys, os, time
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from gsheet_logger import gsheet_logger
from clients.forex_mcp_client import IQForexMCPClient

def audit_and_sync():
    print("Auditing Google Sheets vs Broker History...")
    ws = None
    for attempt in range(5):
        try:
            ws = gsheet_logger.get_or_create_forex_worksheet("Forex_Margin_Trades")
            if ws:
                break
        except Exception as e:
            print(f"Connection attempt {attempt+1} failed: {e}")
            time.sleep(2)

    if not ws:
        print("Failed to reach Google Sheet.")
        return

    rows = ws.get_all_values()
    logged_pos_ids = set()
    for r in rows[1:]:
        if len(r) >= 14 and r[13]:
            logged_pos_ids.add(str(r[13]).strip())

    print(f"Existing rows in GSheet: {len(rows)}, Unique Position IDs: {len(logged_pos_ids)}")

    c = IQForexMCPClient()
    hist = c.get_trade_history(balance_id=1237481096, limit=50) or []
    print(f"Broker history items fetched: {len(hist)}")

    missing_trades = []
    for h in hist:
        pid = str(h.get('position_id') or h.get('id') or '').strip()
        if pid and pid not in logged_pos_ids:
            missing_trades.append(h)

    print(f"\nFound {len(missing_trades)} unlogged trades:")
    for m in missing_trades:
        pid = str(m.get('position_id') or m.get('id'))
        asset = m.get('asset_name') or str(m.get('asset_id'))
        side = str(m.get('type') or m.get('side', 'BUY')).upper()
        pnl = float(m.get('pnl', 0.0))
        reason = m.get('close_reason', 'closed')
        ctime = m.get('close_time', '')
        open_px = float(m.get('open_price', 0.0))
        close_px = float(m.get('close_price', 0.0))
        lots = float(m.get('count') or m.get('lots', 1.0))
        pips = round(abs(close_px - open_px), 2)

        # Build trade payload
        payload = {
            "timestamp": ctime.replace("T", " ").replace("Z", "")[:19] if ctime else time.strftime("%Y-%m-%d %H:%M:%S"),
            "asset": f"CFD {asset}",
            "side": side,
            "lots": lots,
            "entry_price": open_px,
            "stop_loss": 0.0,
            "take_profit": 0.0,
            "exit_price": close_px,
            "pnl": pnl,
            "pips": pips,
            "risk_reward": "Broker Market Order",
            "exit_reason": reason,
            "position_id": pid,
            "balance_equity": 0.0
        }

        success = gsheet_logger.log_forex_margin_trade(payload)
        print(f" -> Backfilled #{pid} ({asset}) PnL=${pnl:.2f} Reason={reason} -> Success: {success}")
        logged_pos_ids.add(pid)
        time.sleep(0.5)

    print("\nAudit and Sync Complete!")

if __name__ == "__main__":
    audit_and_sync()
