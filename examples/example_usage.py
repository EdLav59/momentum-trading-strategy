"""
Example usage of the momentum strategy
"""

from momentum_strategy import MomentumStrategy

# Run with default settings
strategy = MomentumStrategy()
strategy.run_complete_analysis()

# The strategy will:
# 1. Download 5 years of data for 56 S&P 500 stocks
# 2. Run the momentum backtest
# 3. Display performance metrics
# 4. Generate charts
# 5. Save results to CSV files
