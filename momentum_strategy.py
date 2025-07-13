"""
Momentum Trading Strategy Implementation
Based on Jegadeesh & Titman (1993)

A complete Python implementation of the momentum strategy that buys past winners 
and sells past losers, using real market data.

Author: Edouard Lavalard
GitHub: https://github.com/EdLav59/momentum-trading-strategy.git
License: MIT
"""

import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Set style for better-looking plots
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

class MomentumStrategy:
    """
    Momentum Trading Strategy Implementation
    
    Default Configuration:
    - Formation Period: 12 months
    - Holding Period: 6 months
    - Top/Bottom: 20% (quintile portfolios)
    - Rebalancing: Monthly
    """
    
    def __init__(self):
        # Pre-configured dates (last 5 years of data)
        self.end_date = '2024-12-31'
        self.start_date = '2019-01-01'
        
        # Pre-selected liquid large-cap stocks from different sectors
        self.tickers = [
            # Technology
            'AAPL', 'MSFT', 'GOOGL', 'META', 'NVDA', 'ADBE', 'CRM', 'ORCL',
            # Finance
            'JPM', 'BAC', 'WFC', 'GS', 'MS', 'AXP', 'BLK', 'SPGI',
            # Healthcare
            'JNJ', 'UNH', 'PFE', 'ABBV', 'TMO', 'CVS', 'MDT', 'ABT',
            # Consumer
            'AMZN', 'WMT', 'HD', 'PG', 'KO', 'PEP', 'MCD', 'NKE',
            # Industrial
            'BA', 'CAT', 'HON', 'UPS', 'RTX', 'LMT', 'GE', 'MMM',
            # Energy
            'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'OXY', 'PSX', 'VLO',
            # Others
            'BRK-B', 'V', 'MA', 'DIS', 'NFLX', 'CSCO', 'VZ', 'T'
        ]
        
        # Strategy parameters
        self.formation_months = 12
        self.holding_months = 6
        self.top_percentile = 20  # Top 20% are winners
        self.bottom_percentile = 20  # Bottom 20% are losers
        
        # Data storage
        self.price_data = None
        self.returns_data = None
        self.results = []
        
        print("=" * 60)
        print("MOMENTUM STRATEGY INITIALIZED")
        print("=" * 60)
        print(f"Period: {self.start_date} to {self.end_date}")
        print(f"Universe: {len(self.tickers)} stocks from S&P 500")
        print(f"Formation Period: {self.formation_months} months")
        print(f"Holding Period: {self.holding_months} months")
        print(f"Winner/Loser Percentile: Top/Bottom {self.top_percentile}%")
        print("=" * 60)
    
    def fetch_data(self):
        """Download historical price data from Yahoo Finance"""
        print("\n[1/5] Fetching historical data...")
        # Download all data at once for efficiency
        raw_data = yf.download(
            self.tickers,
            start=self.start_date,
            end=self.end_date,
            progress=True
        )
        # Handle new yfinance structure: MultiIndex columns
        if isinstance(raw_data.columns, pd.MultiIndex):
            if 'Adj Close' in raw_data.columns.get_level_values(0):
                self.price_data = raw_data['Adj Close']
            elif 'Close' in raw_data.columns.get_level_values(0):
                self.price_data = raw_data['Close']
            else:
                print("\nERROR: Neither 'Adj Close' nor 'Close' found in downloaded data.")
                raise RuntimeError("No valid price data available for analysis.")
        else:
            # Single ticker or flat columns
            if 'Adj Close' in raw_data.columns:
                self.price_data = raw_data['Adj Close'].to_frame()
            elif 'Close' in raw_data.columns:
                self.price_data = raw_data['Close'].to_frame()
            else:
                print("\nERROR: Neither 'Adj Close' nor 'Close' found in downloaded data.")
                raise RuntimeError("No valid price data available for analysis.")
        # Remove any stocks with too much missing data
        initial_stocks = self.price_data.shape[1]
        self.price_data = self.price_data.dropna(thresh=len(self.price_data)*0.8, axis=1)
        removed_stocks = initial_stocks - self.price_data.shape[1]
        if removed_stocks > 0:
            print(f"  Removed {removed_stocks} stocks due to insufficient data")
        # Error handling for empty data
        if self.price_data.empty or self.price_data.shape[1] == 0:
            print("\nERROR: No price data could be downloaded. Please check your internet connection, yfinance version, or ticker symbols.")
            raise RuntimeError("No price data available for analysis.")
        print(f"Data downloaded: {self.price_data.shape[1]} stocks, {len(self.price_data)} trading days")
        # Calculate returns
        self.returns_data = self.price_data.pct_change().dropna()
        return self.price_data
    
    def calculate_momentum_scores(self, start_date, end_date):
        """Calculate momentum scores for the formation period"""
        # Get returns for the formation period
        mask = (self.returns_data.index >= start_date) & (self.returns_data.index <= end_date)
        period_returns = self.returns_data[mask]
        
        # Calculate cumulative returns (momentum scores)
        momentum_scores = (1 + period_returns).prod() - 1
        
        # Remove NaN values
        momentum_scores = momentum_scores.dropna()
        
        return momentum_scores
    
    def form_portfolios(self, momentum_scores):
        """Form winner and loser portfolios based on momentum scores"""
        n_stocks = len(momentum_scores)
        n_winners = max(1, int(n_stocks * self.top_percentile / 100))
        n_losers = max(1, int(n_stocks * self.bottom_percentile / 100))
        
        # Sort by momentum scores
        sorted_scores = momentum_scores.sort_values(ascending=False)
        
        # Select winners and losers
        winners = sorted_scores.head(n_winners).index.tolist()
        losers = sorted_scores.tail(n_losers).index.tolist()
        
        return winners, losers
    
    def calculate_portfolio_returns(self, stocks, start_date, end_date):
        """Calculate equal-weighted portfolio returns for holding period"""
        mask = (self.returns_data.index >= start_date) & (self.returns_data.index <= end_date)
        period_returns = self.returns_data[mask][stocks]
        
        # Equal-weighted portfolio return
        portfolio_return = period_returns.mean(axis=1).mean()
        
        return portfolio_return
    
    def run_backtest(self):
        """Execute the momentum strategy backtest"""
        print("\n[2/5] Running momentum strategy backtest...")
        
        # Convert dates to datetime
        returns_dates = self.returns_data.index
        
        # Run strategy month by month
        current_date = returns_dates[252]  # Start after 1 year of data
        end_backtest = returns_dates[-126]  # Stop 6 months before end
        
        month_count = 0
        
        while current_date <= end_backtest:
            # Formation period (12 months back)
            formation_end = current_date
            formation_start = formation_end - pd.DateOffset(months=self.formation_months)
            
            # Holding period (6 months forward)
            holding_start = formation_end
            holding_end = holding_start + pd.DateOffset(months=self.holding_months)
            
            # Check if we have enough data
            if holding_end > returns_dates[-1]:
                break
            
            # Calculate momentum scores
            momentum_scores = self.calculate_momentum_scores(formation_start, formation_end)
            
            if len(momentum_scores) < 10:  # Need minimum stocks
                current_date += pd.DateOffset(months=1)
                continue
            
            # Form portfolios
            winners, losers = self.form_portfolios(momentum_scores)
            
            # Calculate returns
            winner_return = self.calculate_portfolio_returns(winners, holding_start, holding_end)
            loser_return = self.calculate_portfolio_returns(losers, holding_start, holding_end)
            momentum_return = winner_return - loser_return
            
            # Store results
            self.results.append({
                'formation_start': formation_start,
                'formation_end': formation_end,
                'holding_start': holding_start,
                'holding_end': holding_end,
                'winner_return': winner_return,
                'loser_return': loser_return,
                'momentum_return': momentum_return,
                'n_winners': len(winners),
                'n_losers': len(losers)
            })
            
            # Progress update
            month_count += 1
            if month_count % 6 == 0:
                print(f"  Processed {month_count} months...")
            
            # Move to next month
            current_date += pd.DateOffset(months=1)
        
        # Convert results to DataFrame
        self.results_df = pd.DataFrame(self.results)
        print(f"Backtest complete: {len(self.results_df)} monthly periods analyzed")
        
        return self.results_df
    
    def analyze_performance(self):
        """Calculate performance metrics"""
        print("\n[3/5] Analyzing performance metrics...")
        
        # Basic statistics
        avg_return = self.results_df['momentum_return'].mean()
        std_return = self.results_df['momentum_return'].std()
        sharpe_ratio = avg_return / std_return if std_return > 0 else 0
        
        # Statistical significance
        t_stat, p_value = stats.ttest_1samp(self.results_df['momentum_return'], 0)
        
        # Win rate
        win_rate = (self.results_df['momentum_return'] > 0).mean()
        
        # Best and worst
        best_return = self.results_df['momentum_return'].max()
        worst_return = self.results_df['momentum_return'].min()
        
        # Print results
        print("\n" + "=" * 50)
        print("STRATEGY PERFORMANCE SUMMARY")
        print("=" * 50)
        print(f"\nMonthly Returns:")
        print(f"  Average Return:     {avg_return:.4f} ({avg_return*100:.2f}%)")
        print(f"  Standard Deviation: {std_return:.4f} ({std_return*100:.2f}%)")
        print(f"  Sharpe Ratio:       {sharpe_ratio:.4f}")
        print(f"  Win Rate:           {win_rate:.2%}")
        
        print(f"\nAnnualized Metrics:")
        print(f"  Annual Return:      {avg_return*12:.2%}")
        print(f"  Annual Volatility:  {std_return*np.sqrt(12):.2%}")
        print(f"  Annual Sharpe:      {sharpe_ratio*np.sqrt(12):.4f}")
        
        print(f"\nStatistical Significance:")
        print(f"  T-Statistic:        {t_stat:.4f}")
        print(f"  P-Value:            {p_value:.4f}")
        print(f"  Significant (5%):   {'Yes' if p_value < 0.05 else 'No'}")
        
        print(f"\nExtreme Returns:")
        print(f"  Best Month:         {best_return:.2%}")
        print(f"  Worst Month:        {worst_return:.2%}")
        
        return {
            'avg_return': avg_return,
            'std_return': std_return,
            'sharpe_ratio': sharpe_ratio,
            't_stat': t_stat,
            'p_value': p_value,
            'win_rate': win_rate
        }
    
    def create_visualizations(self):
        """Generate comprehensive performance charts"""
        print("\n[4/5] Creating visualizations...")
        
        # Set up the plot style
        fig = plt.figure(figsize=(16, 12))
        
        # 1. Cumulative Returns
        ax1 = plt.subplot(2, 2, 1)
        cumulative_returns = (1 + self.results_df['momentum_return']).cumprod()
        winner_cumulative = (1 + self.results_df['winner_return']).cumprod()
        loser_cumulative = (1 + self.results_df['loser_return']).cumprod()
        
        ax1.plot(cumulative_returns.index, cumulative_returns.values, 
                 label='Momentum (Long-Short)', linewidth=3, color='darkblue')
        ax1.plot(winner_cumulative.index, winner_cumulative.values, 
                 label='Winners Only', linewidth=2, color='green', alpha=0.7)
        ax1.plot(loser_cumulative.index, loser_cumulative.values, 
                 label='Losers Only', linewidth=2, color='red', alpha=0.7)
        
        ax1.set_title('Cumulative Performance', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Period')
        ax1.set_ylabel('Cumulative Return')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Monthly Returns Distribution
        ax2 = plt.subplot(2, 2, 2)
        ax2.hist(self.results_df['momentum_return'], bins=30, 
                 alpha=0.7, color='navy', edgecolor='black')
        ax2.axvline(self.results_df['momentum_return'].mean(), 
                    color='red', linestyle='--', linewidth=2,
                    label=f"Mean: {self.results_df['momentum_return'].mean():.3f}")
        ax2.set_title('Distribution of Monthly Returns', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Monthly Return')
        ax2.set_ylabel('Frequency')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # 3. Rolling Performance
        ax3 = plt.subplot(2, 2, 3)
        rolling_mean = self.results_df['momentum_return'].rolling(12).mean()
        rolling_std = self.results_df['momentum_return'].rolling(12).std()
        
        ax3.plot(rolling_mean.index, rolling_mean.values, 
                 label='12-Month Rolling Mean', linewidth=2)
        ax3.fill_between(rolling_mean.index,
                        rolling_mean - rolling_std,
                        rolling_mean + rolling_std,
                        alpha=0.3, label='±1 Std Dev')
        ax3.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax3.set_title('Rolling 12-Month Performance', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Period')
        ax3.set_ylabel('Return')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # 4. Annual Returns
        ax4 = plt.subplot(2, 2, 4)
        self.results_df['year'] = pd.to_datetime(self.results_df['holding_end']).dt.year
        annual_returns = self.results_df.groupby('year')['momentum_return'].apply(
            lambda x: (1 + x).prod() - 1
        )
        
        bars = ax4.bar(annual_returns.index, annual_returns.values, 
                       color=['green' if x > 0 else 'red' for x in annual_returns.values],
                       alpha=0.7, edgecolor='black')
        ax4.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        ax4.set_title('Annual Returns', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Year')
        ax4.set_ylabel('Annual Return')
        ax4.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.1%}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3 if height > 0 else -15),
                        textcoords="offset points",
                        ha='center', va='bottom' if height > 0 else 'top',
                        fontsize=10)
        
        plt.suptitle('Momentum Strategy Performance Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.savefig('momentum_strategy_performance.png', dpi=300, bbox_inches='tight')
        plt.show()
        
        print("Visualizations created and saved")
    
    def save_results(self):
        """Save results to CSV files"""
        print("\n[5/5] Saving results...")
        
        # Save detailed results
        self.results_df.to_csv('momentum_backtest_results.csv', index=False)
        print("Detailed results saved to 'momentum_backtest_results.csv'")
        
        # Save summary statistics
        summary = {
            'Metric': ['Average Monthly Return', 'Monthly Volatility', 'Sharpe Ratio',
                      'Annual Return', 'Annual Volatility', 'Annual Sharpe',
                      'Win Rate', 'Total Periods', 'Best Month', 'Worst Month'],
            'Value': [
                f"{self.results_df['momentum_return'].mean():.4f}",
                f"{self.results_df['momentum_return'].std():.4f}",
                f"{self.results_df['momentum_return'].mean() / self.results_df['momentum_return'].std():.4f}",
                f"{self.results_df['momentum_return'].mean() * 12:.4f}",
                f"{self.results_df['momentum_return'].std() * np.sqrt(12):.4f}",
                f"{(self.results_df['momentum_return'].mean() / self.results_df['momentum_return'].std()) * np.sqrt(12):.4f}",
                f"{(self.results_df['momentum_return'] > 0).mean():.4f}",
                f"{len(self.results_df)}",
                f"{self.results_df['momentum_return'].max():.4f}",
                f"{self.results_df['momentum_return'].min():.4f}"
            ]
        }
        
        summary_df = pd.DataFrame(summary)
        summary_df.to_csv('momentum_summary_statistics.csv', index=False)
        print("Summary statistics saved to 'momentum_summary_statistics.csv'")
    
    def run_complete_analysis(self):
        """Execute the complete momentum strategy analysis"""
        try:
            # Fetch data
            self.fetch_data()
            
            # Run backtest
            self.run_backtest()
            
            # Analyze performance
            self.analyze_performance()
            
            # Create visualizations
            self.create_visualizations()
            
            # Save results
            self.save_results()
            
            print("\n" + "=" * 60)
            print("MOMENTUM STRATEGY ANALYSIS COMPLETE")
            print("=" * 60)
            print("\nFiles created:")
            print("  - momentum_strategy_performance.png")
            print("  - momentum_backtest_results.csv")
            print("  - momentum_summary_statistics.csv")
            print("\nThank you for using this momentum strategy implementation.")
            
        except Exception as e:
            print(f"\nError during analysis: {str(e)}")
            raise


def main():
    """Main execution function"""
    print("\n" + "=" * 60)
    print("MOMENTUM TRADING STRATEGY")
    print("Based on Jegadeesh & Titman (1993)")
    print("=" * 60)
    
    # Create and run strategy
    strategy = MomentumStrategy()
    strategy.run_complete_analysis()


if __name__ == "__main__":
    main()
