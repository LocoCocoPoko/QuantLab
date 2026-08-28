import numpy as np
import pandas as pd
from scipy.optimize import minimize
from src.quantlab.config import RISK_FREE_RATE, DEFAULT_MAX_WEIGHT


def calculate_portfolio_return(
    weights: np.ndarray,
    annual_returns: pd.Series
) -> float:
    """
    Calculate the expected annual return of a portfolio.

    Args:
        weights: Array of portfolio weights, one per asset, summing to 1.0.
        annual_returns: Series of annualized returns, one per asset
                        (e.g. from calculate_annualized_return).

    Returns:
        The portfolio's expected annual return as a float.
    """
    portfolio_return = np.sum(weights * annual_returns)
    return portfolio_return


def calculate_portfolio_volatility(
    weights: np.ndarray,
    covariance_matrix: pd.DataFrame
) -> float:
    """
    Calculate the expected annual volatility of a portfolio.

    Args:
        weights: Array of portfolio weights, one per asset, summing to 1.0.
        covariance_matrix: Annualized covariance matrix
                            (e.g. from calculate_covariance_matrix).

    Returns:
        The portfolio's expected annual volatility (standard deviation) as a float.
    """
    portfolio_variance = weights.T @ covariance_matrix @ weights
    portfolio_volatility = np.sqrt(portfolio_variance)
    return portfolio_volatility


def generate_random_weights(num_assets: int) -> np.ndarray:
    """
    Generate a random set of portfolio weights that sum to 1.0.

    Args:
        num_assets: The number of assets in the portfolio.

    Returns:
        A NumPy array of random weights that sum to 1.0.
    """
    random_weights = np.random.random(num_assets)
    random_weights = random_weights / np.sum(random_weights)
    return random_weights


def run_monte_carlo_simulation(
    annual_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    num_portfolios: int = 5000
) -> pd.DataFrame:
    """
    Simulate many random portfolios and record their risk/return.

    Args:
        annual_returns: Series of annualized returns, one per asset.
        covariance_matrix: Annualized covariance matrix.
        num_portfolios: The number of random portfolios to simulate.

    Returns:
        A DataFrame with one row per simulated portfolio, containing
        columns 'return', 'volatility', and 'weights'.
    """
    num_assets = len(annual_returns)
    results = []

    for _ in range(num_portfolios):
        weights = generate_random_weights(num_assets)
        port_return = calculate_portfolio_return(weights, annual_returns)
        port_volatility = calculate_portfolio_volatility(weights, covariance_matrix)

        results.append({
            "return": port_return,
            "volatility": port_volatility,
            "weights": weights
        })

    return pd.DataFrame(results)


def find_minimum_variance_portfolio(
    annual_returns: pd.Series,
    covariance_matrix: pd.DataFrame
) -> dict:
    """
    Find the portfolio weights that minimize volatility.

    Args:
        annual_returns: Series of annualized returns, one per asset
                        (used only to know how many assets there are).
        covariance_matrix: Annualized covariance matrix.

    Returns:
        A dictionary with keys 'weights', 'return', and 'volatility'
        for the minimum variance portfolio.
    """
    num_assets = len(annual_returns)

    def objective(weights):
        return calculate_portfolio_volatility(weights, covariance_matrix)

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    return {
        "weights": optimal_weights,
        "return": calculate_portfolio_return(optimal_weights, annual_returns),
        "volatility": calculate_portfolio_volatility(optimal_weights, covariance_matrix)
    }


def find_max_sharpe_portfolio(
    annual_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE
) -> dict:
    """
    Find the portfolio weights that maximize the Sharpe ratio.

    Args:
        annual_returns: Series of annualized returns, one per asset.
        covariance_matrix: Annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate, as a decimal.

    Returns:
        A dictionary with keys 'weights', 'return', and 'volatility'
        for the maximum Sharpe ratio portfolio.
    """
    num_assets = len(annual_returns)

    def objective(weights):
        port_return = calculate_portfolio_return(weights, annual_returns)
        port_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
        sharpe = (port_return - risk_free_rate) / port_volatility
        return -sharpe

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, 1) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    return {
        "weights": optimal_weights,
        "return": calculate_portfolio_return(optimal_weights, annual_returns),
        "volatility": calculate_portfolio_volatility(optimal_weights, covariance_matrix)
    }


def find_max_sharpe_portfolio_capped(
    annual_returns: pd.Series,
    covariance_matrix: pd.DataFrame,
    risk_free_rate: float = RISK_FREE_RATE,
    max_weight: float = DEFAULT_MAX_WEIGHT
) -> dict:
    """
    Find the max Sharpe Ratio portfolio, with a cap on any single
    asset's weight to avoid extreme concentration.

    Args:
        annual_returns: Series of annualized returns, one per asset.
        covariance_matrix: Annualized covariance matrix.
        risk_free_rate: Annualized risk-free rate, as a decimal.
        max_weight: Maximum allowed weight for any single asset (0-1).

    Returns:
        A dictionary with keys 'weights', 'return', and 'volatility'.
    """
    num_assets = len(annual_returns)

    def objective(weights):
        port_return = calculate_portfolio_return(weights, annual_returns)
        port_volatility = calculate_portfolio_volatility(weights, covariance_matrix)
        sharpe = (port_return - risk_free_rate) / port_volatility
        return -sharpe

    constraints = {"type": "eq", "fun": lambda weights: np.sum(weights) - 1}
    bounds = tuple((0, max_weight) for _ in range(num_assets))
    initial_guess = np.array([1 / num_assets] * num_assets)

    result = minimize(
        objective,
        initial_guess,
        method="SLSQP",
        bounds=bounds,
        constraints=constraints
    )

    optimal_weights = result.x
    return {
        "weights": optimal_weights,
        "return": calculate_portfolio_return(optimal_weights, annual_returns),
        "volatility": calculate_portfolio_volatility(optimal_weights, covariance_matrix)
    }