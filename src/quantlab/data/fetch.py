import pandas as pd
import yfinance as yf
import logging

logger = logging.getLogger(__name__)

def fetch_price_data(
    tickers: list[str],
    start_date: str,
    end_date: str
) -> pd.DataFrame:
    """
    ...(docstring unchanged, could add a note about partial failures)...
    """
    logger.info(f"Fetching data for {tickers} from {start_date} to {end_date}")

    raw_data = yf.download(
        tickers,
        start=start_date,
        end=end_date,
        auto_adjust=True
    )

    if raw_data.empty:
        logger.error(f"No data returned for {tickers}")
        raise ValueError(
            f"No data returned for tickers {tickers} between "
            f"{start_date} and {end_date}. Check that the ticker "
            f"symbols are correct and the date range is valid."
        )

    prices = raw_data["Close"]

    # Detect tickers that came back entirely empty (invalid symbol)
    fully_missing = prices.columns[prices.isna().all()].tolist()
    if fully_missing:
        logger.error(f"No data at all for: {fully_missing}")
        raise ValueError(
            f"No data found for the following ticker(s): {fully_missing}. "
            f"Please check that they're spelled correctly."
        )

    prices = prices.ffill()
    prices = prices.dropna()

    if prices.empty:
        logger.error(f"No usable overlapping data for {tickers}")
        raise ValueError(
            f"After cleaning, no usable data remained for tickers "
            f"{tickers}. This can happen if none of the tickers "
            f"had overlapping trading data in the given date range."
        )

    logger.info(f"Successfully fetched {len(prices)} rows for {tickers}")
    return prices
