import os
import sys
from datetime import datetime

sys.path.insert(0, '/sdcard/Documents/AGY Files/BreakingBadV3')
from dotenv import load_dotenv
load_dotenv('/sdcard/Documents/AGY Files/BreakingBadV3/.env')
from clients.forex_mcp_client import IQForexMCPClient

client = IQForexMCPClient()
bid = 1237481096
all_trades = []
for ep in ['cfd', 'forex', 'crypto']:
    hist = client.get_trade_history(balance_id=bid, endpoint=ep, limit=100)
    if hist:
        for t in hist:
            t['_ep'] = ep
            all_trades.append(t)

oct18_ts = '2026-10-18T00:00:00'
filtered = [t for t in all_trades if (t.get('close_time') or '') >= oct18_ts]

by_asset = {}
total_pnl = 0.0
wins = 0
losses = 0
bes = 0

for t in filtered:
    asset = t.get('asset_name') or t.get('instrument_id') or t.get('_ep')
    pnl = float(t.get('pnl') or t.get('pnl_net') or 0.0)
    total_pnl += pnl
    if pnl > 0.1:
        wins += 1
    elif pnl < -0.1:
        losses += 1
    else:
        bes += 1
    if asset not in by_asset:
        by_asset[asset] = {'trades': 0, 'wins': 0, 'losses': 0, 'pnl': 0.0}
    by_asset[asset]['trades'] += 1
    by_asset[asset]['pnl'] += pnl
    if pnl > 0.1:
        by_asset[asset]['wins'] += 1
    elif pnl < -0.1:
        by_asset[asset]['losses'] += 1

print('=== OVERALL SUMMARY (Oct 18 - Oct 25) ===')
print(f'Total Trades: {len(filtered)}')
print(f'Wins: {wins} | Losses: {losses} | Breakevens: {bes}')
wr = (wins / (wins + losses) * 100) if (wins + losses) > 0 else 0
print(f'Win Rate: {wr:.1f}%')
print(f'Net PnL: ${total_pnl:.2f}')

print('\n=== BREAKDOWN BY ASSET ===')
for asset, st in sorted(by_asset.items(), key=lambda x: x[1]['pnl']):
    awr = (st['wins'] / st['trades'] * 100) if st['trades'] > 0 else 0
    pnl_str = f"{st['pnl']:+.2f}"
    print(f"{asset:15} | Trades: {st['trades']:2} | Wins: {st['wins']:2} | Losses: {st['losses']:2} | WinRate: {awr:5.1f}% | Net PnL: ${pnl_str}")
