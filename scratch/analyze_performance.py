import sys, os
sys.path.insert(0, os.getcwd())
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from datetime import datetime
import pandas as pd
from gsheet_logger import gsheet_logger

def main():
    ws = gsheet_logger.get_or_create_forex_worksheet('Forex_Margin_Trades')
    values = ws.get_all_values()
    headers = values[0]
    rows = values[1:]
    
    unique_headers = []
    seen = {}
    for h in headers:
        if h in seen:
            seen[h] += 1
            unique_headers.append(f"{h}_{seen[h]}")
        else:
            seen[h] = 0
            unique_headers.append(h)
            
    df = pd.DataFrame(rows, columns=unique_headers)
    
    # Locate PnL column
    pnl_col = next((c for c in df.columns if 'pnl' in c.lower()), None)
    if not pnl_col:
        print("No PnL column found")
        return
        
    df['PnL'] = pd.to_numeric(df[pnl_col].astype(str).str.replace('$', '').str.replace(',', '').str.strip(), errors='coerce').fillna(0.0)
    
    ts_col = next((c for c in df.columns if 'time' in c.lower() or 'date' in c.lower()), None)
    asset_col = next((c for c in df.columns if 'asset' in c.lower() or 'pair' in c.lower()), None)
    
    print(f"Total Logged Trades: {len(df)}")
    print(f"Earliest: {df[ts_col].iloc[0]} | Latest: {df[ts_col].iloc[-1]}")
    
    # 1. Overall stats
    wins = df[df['PnL'] > 0]
    losses = df[df['PnL'] < 0]
    evens = df[df['PnL'] == 0]
    
    gross_win = wins['PnL'].sum()
    gross_loss = losses['PnL'].sum()
    net_pnl = df['PnL'].sum()
    win_rate = (len(wins) / len(df) * 100) if len(df) > 0 else 0
    pf = abs(gross_win / gross_loss) if abs(gross_loss) > 0 else 999.0
    
    print(f"\n================ OVERALL STATS ================")
    print(f"Net PnL:        ${net_pnl:,.2f}")
    print(f"Win Rate:       {win_rate:.1f}% ({len(wins)} W / {len(losses)} L / {len(evens)} BE)")
    print(f"Gross Profit:   ${gross_win:,.2f}")
    print(f"Gross Loss:     ${gross_loss:,.2f}")
    print(f"Profit Factor:  {pf:.2f}")
    if len(wins) > 0:
        print(f"Avg Win:        ${wins['PnL'].mean():.2f}")
    if len(losses) > 0:
        print(f"Avg Loss:       ${losses['PnL'].mean():.2f}")
        
    # 2. Focus on recent "Prop Firm Mode" (Oct 4 - Oct 7)
    df['dt'] = pd.to_datetime(df[ts_col], errors='coerce')
    prop_era = df[df['dt'] >= '2026-10-04'].copy()
    if not prop_era.empty:
        p_wins = prop_era[prop_era['PnL'] > 0]
        p_losses = prop_era[prop_era['PnL'] < 0]
        p_gross_win = p_wins['PnL'].sum()
        p_gross_loss = p_losses['PnL'].sum()
        p_net = prop_era['PnL'].sum()
        p_wr = (len(p_wins) / len(prop_era) * 100)
        p_pf = abs(p_gross_win / p_gross_loss) if abs(p_gross_loss) > 0 else 999.0
        
        print(f"\n================ PROP FIRM ERA (Oct 4 - Oct 7) ================")
        print(f"Total Trades:   {len(prop_era)}")
        print(f"Net PnL:        ${p_net:,.2f}")
        print(f"Win Rate:       {p_wr:.1f}% ({len(p_wins)} W / {len(p_losses)} L)")
        print(f"Gross Profit:   ${p_gross_win:,.2f}")
        print(f"Gross Loss:     ${p_gross_loss:,.2f}")
        print(f"Profit Factor:  {p_pf:.2f}")
        if len(p_wins) > 0:
            print(f"Avg Win:        ${p_wins['PnL'].mean():.2f}")
        if len(p_losses) > 0:
            print(f"Avg Loss:       ${p_losses['PnL'].mean():.2f}")
            
    # 3. Strategy / Asset Breakdown in Prop Firm Era
    if not prop_era.empty and asset_col:
        print(f"\n================ BREAKDOWN BY STRATEGY / ASSET (Prop Firm Era) ================")
        grp = prop_era.groupby(asset_col)['PnL'].agg(
            Trades='count',
            Net_PnL='sum',
            Wins=lambda x: (x > 0).sum(),
            Losses=lambda x: (x < 0).sum(),
            Avg_Win=lambda x: round(x[x > 0].mean(), 2) if (x > 0).sum() > 0 else 0,
            Avg_Loss=lambda x: round(x[x < 0].mean(), 2) if (x < 0).sum() > 0 else 0,
            Win_Rate=lambda x: f"{(x > 0).mean()*100:.1f}%"
        ).sort_values(by='Net_PnL', ascending=False)
        print(grp.to_string())

    # 4. Check broker live balance
    from clients.forex_mcp_client import IQForexMCPClient
    try:
        c = IQForexMCPClient()
        bals = c.list_balances() or []
        for b in bals:
            if b.get('balance_id') == 1237481096:
                print(f"\nCurrent Prop Account Balance: ${b.get('amount') or b.get('balance')} | Equity: ${b.get('equity')}")
    except Exception as e:
        print(f"Broker query error: {e}")

if __name__ == '__main__':
    main()
