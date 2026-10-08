import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from gsheet_logger import gsheet_logger

trades = [
    {
        'timestamp': '2026-10-07 00:52:21',
        'asset': 'Trendline US100 (Leg 1 TP1)',
        'side': 'SHORT',
        'lots': 0.5,
        'entry_price': 31252.86,
        'stop_loss': 31267.86,
        'take_profit': 31234.07,
        'exit_price': 31234.07,
        'pnl': 9.40,
        'pips': 188.0,
        'risk_reward': 'Leg 1 (+1.0R Plan)',
        'exit_reason': 'take_profit',
        'position_id': '100167446576',
        'balance_equity': 5031.0
    },
    {
        'timestamp': '2026-10-07 02:03:18',
        'asset': 'Trendline US100 (Leg 1 TP1)',
        'side': 'SHORT',
        'lots': 0.5,
        'entry_price': 31247.90,
        'stop_loss': 31262.90,
        'take_profit': 31208.88,
        'exit_price': 31208.88,
        'pnl': 19.51,
        'pips': 390.0,
        'risk_reward': 'Leg 1 (+1.0R Plan)',
        'exit_reason': 'take_profit',
        'position_id': '100167425033',
        'balance_equity': 5050.51
    },
    {
        'timestamp': '2026-10-07 02:04:37',
        'asset': 'Trendline US100 (Leg 2 Runner)',
        'side': 'SHORT',
        'lots': 0.5,
        'entry_price': 31252.96,
        'stop_loss': 31250.00,
        'take_profit': 31204.76,
        'exit_price': 31204.76,
        'pnl': 24.10,
        'pips': 482.0,
        'risk_reward': 'Leg 2 (+2.5R Runner)',
        'exit_reason': 'take_profit',
        'position_id': '100167446577',
        'balance_equity': 5074.61
    },
    {
        'timestamp': '2026-10-07 09:04:15',
        'asset': 'Trendline US100 (Leg 2 Runner)',
        'side': 'SHORT',
        'lots': 0.5,
        'entry_price': 31246.72,
        'stop_loss': 31244.00,
        'take_profit': 31148.45,
        'exit_price': 31148.45,
        'pnl': 49.14,
        'pips': 982.7,
        'risk_reward': 'Leg 2 (+2.5R Runner)',
        'exit_reason': 'take_profit',
        'position_id': '100167425034',
        'balance_equity': 5123.75
    }
]

for t in trades:
    success = gsheet_logger.log_forex_margin_trade(t)
    print(f"Logged {t['asset']} #{t['position_id']} PnL=+${t['pnl']:.2f} -> Success: {success}")

print("Done backfilling!")
