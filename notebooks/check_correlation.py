from src.quantlab.data.fetch import fetch_price_data
from src.quantlab.analytics.returns import calculate_log_returns
from src.quantlab.analytics.risk import (
    calculate_correlation_matrix,
    calculate_covariance_matrix,
)

prices = fetch_price_data(
    tickers = ["AAPL", "MSFT", "GOOG"],
    start_date = "2021-01-01",
    end_date = "2024-01-01"
)

returns = calculate_log_returns(prices)

correlation = calculate_correlation_matrix(returns)
covariance = calculate_covariance_matrix(returns)

print(correlation)
print()
print(covariance)