import os
import sys
from dotenv import load_dotenv

if sys.stdout.encoding != 'utf-8':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

from clients.forex_mcp_client import IQForexMCPClient

load_dotenv()
fx = IQForexMCPClient()
fx.initialize()

bal = fx.get_training_balance()
bid = bal.get("id") or bal.get("balance_id")
print(f"Current Broker Training Equity: ${bal.get('equity')} | Balance: ${bal.get('balance')}")

hist = fx.get_trade_history(balance_id=bid, limit=40) or []
print(f"Retrieved {len(hist)} trades:\n")

total_pnl = 0.0
win_cnt = 0
loss_cnt = 0

for h in hist:
    pid = h.get("position_id") or h.get("order_id") or h.get("id")
    aid = h.get("asset_id")
    sym = "Gold" if aid == 74 else ("BTC" if aid == 816 else ("EURUSD" if aid == 1 else str(aid)))
    side = str(h.get("side") or h.get("direction") or "BUY").upper()
    lots = h.get("lots") or h.get("count") or 1.0
    pnl = float(h.get("pnl") or 0.0)
    reason = h.get("close_reason") or "closed"
    open_px = float(h.get("open_price") or 0.0)
    close_px = float(h.get("close_price") or 0.0)
    t = h.get("close_time") or h.get("open_time")

    total_pnl += pnl
    if pnl > 0:
        win_cnt += 1
    elif pnl < 0:
        loss_cnt += 1

    print(f"{t} | {sym} {side} | Lots: {lots} | PnL: ${pnl:+8.2f} | Reason: {reason:<12} | Open: {open_px:.2f} -> Close: {close_px:.2f} | ID: #{pid}")

print(f"\nTotal PnL across last {len(hist)} trades: ${total_pnl:+.2f}")
print(f"Record: {win_cnt}W - {loss_cnt}L")
