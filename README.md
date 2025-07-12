# momentum-trading-strategy
Implementation of Jegadeesh &amp; Titman (1993) momentum trading strategy
# Momentum Trading Strategy

Python implementation of the Jegadeesh & Titman (1993) momentum trading strategy.

## Overview

This project implements the academic momentum strategy that involves:
- Buying stocks with the highest returns over the past 12 months (winners)
- Selling stocks with the lowest returns over the past 12 months (losers)
- Holding the portfolio for 6 months
- Monthly rebalancing

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/momentum-trading-strategy.git
cd momentum-trading-strategy
pip install -r requirements.txt

Usage
pythonpython momentum_strategy.py
Features

Downloads historical data from Yahoo Finance
Implements the classic 12-6 momentum strategy
Calculates performance metrics (Sharpe ratio, win rate, etc.)
Generates performance visualizations
Exports results to CSV

Results
The strategy analyzes 56 large-cap stocks from the S&P 500 over the period 2019-2024.
Output Files

momentum_strategy_performance.png - Performance charts
momentum_backtest_results.csv - Detailed monthly results
momentum_summary_statistics.csv - Summary statistics

Requirements

Python 3.7+
See requirements.txt for package dependencies

References
Jegadeesh, N., & Titman, S. (1993). Returns to buying winners and selling losers: Implications for stock market efficiency. The Journal of Finance, 48(1), 65-91.
License
MIT License - see LICENSE file for details