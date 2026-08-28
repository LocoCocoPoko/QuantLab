"""
Central configuration for QuantLab.

Change values here rather than hardcoding them throughout the codebase.
"""

# Risk-free rate used in Sharpe Ratio calculations (annualized, as a decimal)
RISK_FREE_RATE = 0.04 

# Trading days per year, used for annualizing daily metrics
TRADING_DAYS_PER_YEAR = 252

# Default tickers shown in the dashboard on first load
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOG"]

# Default max weight per asset for the capped optimizer
DEFAULT_MAX_WEIGHT = 0.4

