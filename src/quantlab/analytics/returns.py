import numpy as np
import pandas as pd

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the log returns of a DataFrame of prices.
    
    Args:
        prices: A DataFrame indexed by date, with one column per ticker, containing adjusted closed prices.
        
    Returns:
        A DataFrame of log returns, indexed by date, with one column per ticker.
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna()  # Drop the first row which will be NaN after the shift
    return log_returns