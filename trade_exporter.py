# trade_exporter.py
import pandas as pd
import logging
from datetime import datetime
from typing import List, Dict
import os

logger = logging.getLogger(__name__)

EXPORTS_DIR = "exports"
os.makedirs(EXPORTS_DIR, exist_ok=True)


def export_to_csv(trades: List[Dict], filename: str = None) -> str:
    """
    Export trades to CSV file.
    
    Args:
        trades: List of trade dictionaries
        filename: Optional custom filename
    
    Returns:
        Path to generated CSV file
    """
    try:
        if not trades:
            logger.warning("No trades to export")
            return None
        
        # Convert to DataFrame
        df = pd.DataFrame(trades)
        
        # Select and order columns
        columns = ['timestamp', 'asset', 'direction', 'amount', 'expiry', 
                  'result', 'profit', 'gale_level', 'signal_source']
        df = df[[col for col in columns if col in df.columns]]
        
        # Format timestamp
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
        
        # Generate filename
        if not filename:
            filename = f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv'
        
        filepath = os.path.join(EXPORTS_DIR, filename)
        
        # Export to CSV
        df.to_csv(filepath, index=False)
        
        logger.info(f"✅ Exported {len(trades)} trades to CSV: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"❌ Failed to export to CSV: {e}")
        return None


def export_to_excel(trades: List[Dict], stats: Dict, best_pairs: List[Dict], filename: str = None) -> str:
    """
    Export trades to Excel file with multiple sheets.
    
    Args:
        trades: List of trade dictionaries
        stats: Statistics dictionary
        best_pairs: Best performing pairs
        filename: Optional custom filename
    
    Returns:
        Path to generated Excel file
    """
    try:
        if not trades:
            logger.warning("No trades to export")
            return None
        
        # Generate filename
        if not filename:
            filename = f'trades_{datetime.now().strftime("%Y%m%d_%H%M%S")}.xlsx'
        
        filepath = os.path.join(EXPORTS_DIR, filename)
        
        # Create Excel writer
        with pd.ExcelWriter(filepath, engine='openpyxl') as writer:
            # Sheet 1: All Trades
            df_trades = pd.DataFrame(trades)
            columns = ['timestamp', 'asset', 'direction', 'amount', 'expiry', 
                      'result', 'profit', 'gale_level', 'signal_source']
            df_trades = df_trades[[col for col in columns if col in df_trades.columns]]
            
            if 'timestamp' in df_trades.columns:
                df_trades['timestamp'] = pd.to_datetime(df_trades['timestamp']).dt.strftime('%Y-%m-%d %H:%M:%S')
            
            df_trades.to_excel(writer, sheet_name='Trades', index=False)
            
            # Sheet 2: Summary Statistics
            summary_data = {
                'Metric': ['Total Trades', 'Wins', 'Losses', 'Win Rate (%)', 'Total Profit ($)', 'Avg Profit ($)'],
                'Value': [
                    stats.get('total_trades', 0),
                    stats.get('wins', 0),
                    stats.get('losses', 0),
                    f"{stats.get('win_rate', 0):.2f}",
                    f"{stats.get('total_profit', 0):.2f}",
                    f"{stats.get('avg_profit', 0):.2f}"
                ]
            }
            df_summary = pd.DataFrame(summary_data)
            df_summary.to_excel(writer, sheet_name='Summary', index=False)
            
            # Sheet 3: Best Pairs
            if best_pairs:
                df_pairs = pd.DataFrame(best_pairs)
                df_pairs.to_excel(writer, sheet_name='Best Pairs', index=False)
            
            # Sheet 4: Daily Breakdown
            df_trades_full = pd.DataFrame(trades)
            if 'timestamp' in df_trades_full.columns and 'profit' in df_trades_full.columns:
                df_trades_full['date'] = pd.to_datetime(df_trades_full['timestamp']).dt.date
                daily = df_trades_full.groupby('date').agg({
                    'profit': ['sum', 'count'],
                    'result': lambda x: (x == 'WIN').sum()
                }).reset_index()
                daily.columns = ['Date', 'Total Profit', 'Total Trades', 'Wins']
                daily['Losses'] = daily['Total Trades'] - daily['Wins']
                daily['Win Rate (%)'] = (daily['Wins'] / daily['Total Trades'] * 100).round(2)
                daily.to_excel(writer, sheet_name='Daily Breakdown', index=False)
        
        logger.info(f"✅ Exported {len(trades)} trades to Excel: {filepath}")
        return filepath
    except Exception as e:
        logger.error(f"❌ Failed to export to Excel: {e}")
        return None
