import numpy as np

from src.quantlab.data.fetch import fetch_price_data
from src.quantlab.analytics.returns import calculate_log_returns
from src.quantlab.analytics.risk import (
    calculate_annualized_return,
    calculate_covariance_matrix,
)
from src.quantlab.optimization.portfolio import (
    calculate_portfolio_return,
    calculate_portfolio_volatility,
)

prices = fetch_price_data(
    tickers = ["AAPL", "MSFT", "GOOG"],
    start_date = "2021-01-01",
    end_date = "2024-01-01"
)


returns = calculate_log_returns(prices)
annual_returns = calculate_annualized_return(returns)
cov_matrix = calculate_covariance_matrix(returns)

weights = np.array([1/3, 1/3, 1/3])  # Equal weights for AAPL, MSFT, and GOOG

port_return = calculate_portfolio_return(weights, annual_returns)
port_volatility = calculate_portfolio_volatility(weights, cov_matrix)

print(f"Portfolio Return: {port_return:.4f}") 
print(f"Portfolio Volatility: {port_volatility:.4f}")


from src.quantlab.optimization.portfolio import generate_random_weights

weights = generate_random_weights(3)
print(weights)
print(weights.sum())

from src.quantlab.optimization.portfolio import run_monte_carlo_simulation

simulation_results = run_monte_carlo_simulation(annual_returns, cov_matrix, num_portfolios = 5000) 
print(simulation_results.head())
print(simulation_results.shape)
print(simulation_results["volatility"].min())
print(simulation_results["return"].max())


import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
plt.scatter(
    simulation_results["volatility"],
    simulation_results["return"],
    c=simulation_results["return"] / simulation_results["volatility"],
    cmap="viridis",
    s=10,
    alpha=0.5
)

plt.colorbar(label="Return / Volatility")
plt.xlabel("Volatility (Risk)")
plt.ylabel("Expected Return")
plt.title("Simulated Portfolios - Efficient Frontier Shape")
plt.savefig("notebooks/simulated_frontier.png")
plt.show()


from src.quantlab.optimization.portfolio import find_minimum_variance_portfolio

min_var_portfolio = find_minimum_variance_portfolio(annual_returns, cov_matrix)

print("Minimum Variance Portfolio:")
print(f" Weights: {min_var_portfolio['weights']}")
print(f" Return: {min_var_portfolio['return']:.4f}")
print(f" Volatility: {min_var_portfolio['volatility']:.4f}")


from src.quantlab.optimization.portfolio import find_max_sharpe_portfolio

max_sharpe_portfolio = find_max_sharpe_portfolio(annual_returns, cov_matrix)

print(f"Maximum Sharpe Ratio Portfolio:")
print(f" Weights: {max_sharpe_portfolio['weights']}")
print(f" Return: {max_sharpe_portfolio['return']:.4f}")
print(f" Volatility: {max_sharpe_portfolio['volatility']:.4f}")

sharpe = (max_sharpe_portfolio['return'] - 0.04) / max_sharpe_portfolio['volatility']
print(f" Sharpe Ratio: {sharpe:.4f}")

print(annual_returns)
print(cov_matrix.columns)

from src.quantlab.optimization.portfolio import find_max_sharpe_portfolio_capped

capped_portfolio = find_max_sharpe_portfolio_capped(annual_returns, cov_matrix, max_weight=0.4)

print("Capped Max Sharpe Portfolio (max 40% per asset):")
print(f"  Weights: {capped_portfolio['weights']}")
print(f"  Return: {capped_portfolio['return']:.4f}")
print(f"  Volatility: {capped_portfolio['volatility']:.4f}")

capped_sharpe = (capped_portfolio['return'] - 0.04) / capped_portfolio['volatility']
print(f"  Sharpe Ratio: {capped_sharpe:.4f}")