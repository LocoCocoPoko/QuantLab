from src.quantlab.data.fetch import fetch_price_data
from src.quantlab.analytics.returns import calculate_log_returns

prices = fetch_price_data(
    tickers = ["AAPL", "MSFT", "GOOG"],
    start_date = "2023-01-01",
    end_date = "2023-06-01"
)

returns = calculate_log_returns(prices)
print(returns.head(5)) # Shows first 5 rows
print(returns.shape) # Returns (rows,columns) as a tuple. How many trading days we get and did we get exactly 3 columns(one per ticker - Apple, Google, Microsoft)?
print(prices.shape) # Returns (rows,columns) as a tuple. How many trading days we get and did we get exactly 3 columns(one per ticker - Apple, Google, Microsoft)?