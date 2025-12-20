# chart_generator.py
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timedelta
from typing import List, Dict
import logging
import os

logger = logging.getLogger(__name__)

CHARTS_DIR = "charts"
os.makedirs(CHARTS_DIR, exist_ok=True)


def generate_pnl_chart(trades: List[Dict], days: int = 7) -> str:
    """
    Generate profit/loss timeline chart.
    
    Args:
        trades: List of trade dictionaries
        days: Number of days to show
    
    Returns:
        Path to generated chart image
    """
    try:
        # Group trades by date
        daily_pnl = {}
        for trade in trades:
            if not trade.get('timestamp'):
                continue
            
            date = datetime.fromisoformat(trade['timestamp']).date()
            profit = trade.get('profit', 0) or 0
            
            if date in daily_pnl:
                daily_pnl[date] += profit
            else:
                daily_pnl[date] = profit
        
        # Sort by date
        dates = sorted(daily_pnl.keys())
        profits = [daily_pnl[d] for d in dates]
        
        # Calculate cumulative profit
        cumulative = []
        total = 0
        for p in profits:
            total += p
            cumulative.append(total)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.plot(dates, cumulative, marker='o', linewidth=2, markersize=6, color='#2196F3')
        ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
        ax.fill_between(dates, cumulative, 0, alpha=0.3, color='#2196F3')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Cumulative Profit ($)', fontsize=12)
        ax.set_title('Profit/Loss Timeline', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(CHARTS_DIR, f'pnl_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ P&L chart generated: {chart_path}")
        return chart_path
    except Exception as e:
        logger.error(f"❌ Failed to generate P&L chart: {e}")
        return None


def generate_winrate_chart(trades: List[Dict]) -> str:
    """Generate win rate chart."""
    try:
        # Group by date
        daily_stats = {}
        for trade in trades:
            if not trade.get('timestamp') or not trade.get('result'):
                continue
            
            date = datetime.fromisoformat(trade['timestamp']).date()
            
            if date not in daily_stats:
                daily_stats[date] = {'wins': 0, 'losses': 0}
            
            if trade['result'] == 'WIN':
                daily_stats[date]['wins'] += 1
            elif trade['result'] == 'LOSS':
                daily_stats[date]['losses'] += 1
        
        # Calculate win rates
        dates = sorted(daily_stats.keys())
        win_rates = []
        for date in dates:
            stats = daily_stats[date]
            total = stats['wins'] + stats['losses']
            win_rate = (stats['wins'] / total * 100) if total > 0 else 0
            win_rates.append(win_rate)
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        bars = ax.bar(dates, win_rates, color=['#4CAF50' if wr >= 50 else '#F44336' for wr in win_rates])
        ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% Break-even')
        
        ax.set_xlabel('Date', fontsize=12)
        ax.set_ylabel('Win Rate (%)', fontsize=12)
        ax.set_title('Daily Win Rate', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend()
        
        # Format x-axis
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%d'))
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(CHARTS_DIR, f'winrate_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Win rate chart generated: {chart_path}")
        return chart_path
    except Exception as e:
        logger.error(f"❌ Failed to generate win rate chart: {e}")
        return None


def generate_asset_performance_chart(best_pairs: List[Dict]) -> str:
    """Generate asset performance chart."""
    try:
        if not best_pairs:
            return None
        
        assets = [pair['asset'] for pair in best_pairs]
        profits = [pair['total_profit'] for pair in best_pairs]
        
        # Create chart
        fig, ax = plt.subplots(figsize=(10, 6))
        colors = ['#4CAF50' if p >= 0 else '#F44336' for p in profits]
        bars = ax.barh(assets, profits, color=colors)
        
        ax.set_xlabel('Total Profit ($)', fontsize=12)
        ax.set_ylabel('Asset', fontsize=12)
        ax.set_title('Top Performing Assets', fontsize=14, fontweight='bold')
        ax.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
        ax.grid(True, alpha=0.3, axis='x')
        
        plt.tight_layout()
        
        # Save chart
        chart_path = os.path.join(CHARTS_DIR, f'asset_chart_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Asset performance chart generated: {chart_path}")
        return chart_path
    except Exception as e:
        logger.error(f"❌ Failed to generate asset chart: {e}")
        return None


def generate_summary_dashboard(trades: List[Dict], stats: Dict, best_pairs: List[Dict]) -> str:
    """Generate a combined summary dashboard."""
    try:
        fig = plt.figure(figsize=(14, 10))
        gs = fig.add_gridspec(3, 2, hspace=0.3, wspace=0.3)
        
        # 1. Statistics summary (text)
        ax1 = fig.add_subplot(gs[0, :])
        ax1.axis('off')
        
        summary_text = f"""
        📊 TRADING SUMMARY
        
        Total Trades: {stats.get('total_trades', 0)}
        Wins: {stats.get('wins', 0)} | Losses: {stats.get('losses', 0)}
        Win Rate: {stats.get('win_rate', 0):.1f}%
        Total Profit: ${stats.get('total_profit', 0):.2f}
        Avg Profit/Trade: ${stats.get('avg_profit', 0):.2f}
        """
        
        ax1.text(0.5, 0.5, summary_text, ha='center', va='center', 
                fontsize=14, family='monospace', weight='bold')
        
        # 2. P&L Timeline
        ax2 = fig.add_subplot(gs[1, :])
        daily_pnl = {}
        for trade in trades:
            if trade.get('timestamp'):
                date = datetime.fromisoformat(trade['timestamp']).date()
                profit = trade.get('profit', 0) or 0
                daily_pnl[date] = daily_pnl.get(date, 0) + profit
        
        if daily_pnl:
            dates = sorted(daily_pnl.keys())
            profits = [daily_pnl[d] for d in dates]
            cumulative = []
            total = 0
            for p in profits:
                total += p
                cumulative.append(total)
            
            ax2.plot(dates, cumulative, marker='o', linewidth=2, color='#2196F3')
            ax2.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
            ax2.fill_between(dates, cumulative, 0, alpha=0.3, color='#2196F3')
            ax2.set_title('Cumulative Profit', fontweight='bold')
            ax2.grid(True, alpha=0.3)
        
        # 3. Win/Loss Pie Chart
        ax3 = fig.add_subplot(gs[2, 0])
        wins = stats.get('wins', 0)
        losses = stats.get('losses', 0)
        if wins + losses > 0:
            ax3.pie([wins, losses], labels=['Wins', 'Losses'], 
                   colors=['#4CAF50', '#F44336'], autopct='%1.1f%%', startangle=90)
            ax3.set_title('Win/Loss Distribution', fontweight='bold')
        
        # 4. Top Assets
        ax4 = fig.add_subplot(gs[2, 1])
        if best_pairs:
            assets = [pair['asset'][:6] for pair in best_pairs[:5]]
            profits = [pair['total_profit'] for pair in best_pairs[:5]]
            colors = ['#4CAF50' if p >= 0 else '#F44336' for p in profits]
            ax4.barh(assets, profits, color=colors)
            ax4.set_title('Top Assets', fontweight='bold')
            ax4.axvline(x=0, color='gray', linestyle='-', linewidth=0.8)
        
        # Save dashboard
        chart_path = os.path.join(CHARTS_DIR, f'dashboard_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
        plt.savefig(chart_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        logger.info(f"✅ Summary dashboard generated: {chart_path}")
        return chart_path
    except Exception as e:
        logger.error(f"❌ Failed to generate dashboard: {e}")
        return None
