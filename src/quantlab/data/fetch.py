# The imports
import pandas as pd
import yfinance as yf

# Function to fetch stock data from Yahoo Finance
def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str,
) -> pd.DataFrame:  
    """
    Fetch adjusted close prices for a list of tickers over a date range.
    
    Args:
        tickers: List of stock ticker symbols. e.g., ['AAPL', 'MSFT'].
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.
        
    Returns:
        A Dataframe indexed by date, with one column per ticker, containing adjusted closed prices.
    """
    
    raw_data = yf.download(tickers,
                           start = start_date,
                           end = end_date,
                           auto_adjust = True)
    prices = raw_data["Close"]
    
    prices = prices.ffill()  # Forward fill missing values. Only fixes gaps after real data has started.
    prices = prices.dropna()  # Drop any remaining rows with NaN values 
    return prices
