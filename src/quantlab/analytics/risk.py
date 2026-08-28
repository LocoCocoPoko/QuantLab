import numpy as np
import pandas as pd
from src.quantlab.config import TRADING_DAYS_PER_YEAR, RISK_FREE_RATE


def calculate_annualized_volatility(returns: pd.DataFrame, trading_days: int = TRADING_DAYS_PER_YEAR) -> pd.Series:
    """
    Calculate annualized volatility (standard deviation of returns) per ticker.

    Args:
        returns: DataFrame of daily log returns, one column per ticker
                 (e.g. from calculate_log_returns).
        trading_days: Number of trading days per year used for annualizing.

    Returns:
        A Series with one volatility value per ticker, annualized
        by scaling daily volatility by sqrt(trading_days).
    """
    daily_volatility = returns.std()
    annual_volatility = daily_volatility * np.sqrt(trading_days)
    return annual_volatility


def calculate_annualized_return(returns: pd.DataFrame) -> pd.Series:
    """
    Calculate annualized return per ticker from daily log returns.

    Args:
        returns: DataFrame of daily log returns, one column per ticker.

    Returns:
        A Series with one annualized return value per ticker.
    """
    daily_mean_return = returns.mean()
    annual_return = daily_mean_return * TRADING_DAYS_PER_YEAR
    return annual_return


def calculate_sharpe_ratio(
    returns: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE
) -> pd.Series:
    """
    Calculate the Sharpe Ratio per ticker.

    Args:
        returns: DataFrame of daily log returns, one column per ticker.
        risk_free_rate: Annualized risk-free rate, expressed as a
                        decimal (e.g. 0.04 for 4%).

    Returns:
        A Series with one Sharpe Ratio value per ticker.
    """
    annual_return = calculate_annualized_return(returns)
    annual_volatility = calculate_annualized_volatility(returns)
    sharpe = (annual_return - risk_free_rate) / annual_volatility
    return sharpe


def calculate_correlation_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the correlation matrix between all tickers' returns.

    Args:
        returns: DataFrame of daily log returns, one column per ticker.

    Returns:
        A square DataFrame where both rows and columns are tickers,
        and each cell is the correlation between that pair of tickers.
    """
    return returns.corr()


def calculate_covariance_matrix(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the annualized covariance matrix between all tickers' returns.

    Args:
        returns: DataFrame of daily log returns, one column per ticker.

    Returns:
        A square DataFrame where both rows and columns are tickers,
        and each cell is the annualized covariance between that pair.
    """
    daily_cov = returns.cov()
    annual_cov = daily_cov * TRADING_DAYS_PER_YEAR
    return annual_cov