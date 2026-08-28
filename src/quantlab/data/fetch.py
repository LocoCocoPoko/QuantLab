import pandas as pd
import yfinance as yf

def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    Fetch adjusted close prices for a list of tickers over a date range.

    Missing values are forward-filled (assuming the last known price
    persists), and any remaining leading rows with no data at all
    (e.g. before an IPO) are dropped.

    Args:
        tickers: List of stock ticker symbols, e.g. ["AAPL", "MSFT"].
        start_date: Start date in "YYYY-MM-DD" format.
        end_date: End date in "YYYY-MM-DD" format.

    Returns:
        A DataFrame indexed by date, with one column per ticker,
        containing adjusted close prices.

    Raises:
        ValueError: If no valid data could be fetched for the given
                    tickers and date range.
    """
    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    if raw_data.empty:
        raise ValueError(
            f"No data returned for tickers {tickers} between "
            f"{start_date} and {end_date}. Check that the ticker "
            f"symbols are correct and the date range is valid."
        )

    prices = raw_data["Close"]

    prices = prices.ffill()
    prices = prices.dropna()

    if prices.empty:
        raise ValueError(
            f"After cleaning, no usable data remained for tickers "
            f"{tickers}. This can happen if none of the tickers "
            f"had overlapping trading data in the given date range."
        )

    return prices
