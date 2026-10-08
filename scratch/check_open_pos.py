import sys, os
sys.path.insert(0, os.getcwd())
from clients.forex_mcp_client import IQForexMCPClient

c = IQForexMCPClient()
pos = c.list_positions(balance_id=1237481096)
print(f"Open positions count: {len(pos)}")
for p in pos:
    pid = p.get('position_id')
    inst = p.get('instrument_id')
    side = p.get('direction') or p.get('side')
    lots = p.get('count')
    entry = p.get('open_price')
    cur = p.get('close_price')
    pnl = float(p.get('pnl', 0.0))
    sl = p.get('stop_loss')
    tp = p.get('take_profit')
    otime = p.get('open_time')
    print(f"#{pid} | Inst: {inst} | Side: {side} | Lots: {lots} | Entry: {entry} | PnL: ${pnl:.2f} | SL: {sl} | TP: {tp} | Opened: {otime}")
