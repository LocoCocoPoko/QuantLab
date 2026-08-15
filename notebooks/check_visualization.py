from src.quantlab.data.fetch import fetch_price_data
from src.quantlab.analytics.returns import calculate_log_returns
from src.quantlab.analytics.risk import (
    calculate_correlation_matrix,
    calculate_annualized_return,
    calculate_covariance_matrix,
)
from src.quantlab.optimization.portfolio import (
    run_monte_carlo_simulation,
    find_minimum_variance_portfolio,
    find_max_sharpe_portfolio,
    find_max_sharpe_portfolio_capped,
)
from src.quantlab.visualization.charts import (
    plot_correlation_heatmap,
    plot_efficient_frontier,
    plot_cumulative_returns,
    plot_portfolio_allocation,
)

prices = fetch_price_data(
    tickers=["AAPL", "MSFT", "GOOG"],
    start_date="2021-01-01",
    end_date="2024-01-01"
)

returns = calculate_log_returns(prices)
correlation = calculate_correlation_matrix(returns)

plot_correlation_heatmap(correlation)

annual_returns = calculate_annualized_return(returns)
cov_matrix = calculate_covariance_matrix(returns)

simulation_results = run_monte_carlo_simulation(annual_returns, cov_matrix, num_portfolios=5000)
min_var = find_minimum_variance_portfolio(annual_returns, cov_matrix)
max_sharpe = find_max_sharpe_portfolio(annual_returns, cov_matrix)

plot_efficient_frontier(simulation_results, min_var, max_sharpe)

plot_cumulative_returns(returns)

capped_portfolio = find_max_sharpe_portfolio_capped(annual_returns, cov_matrix, max_weight=0.4)
plot_portfolio_allocation(capped_portfolio["weights"], list(annual_returns.index))