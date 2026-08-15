import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def plot_correlation_heatmap(correlation_matrix: pd.DataFrame) -> None:
    """
    Plot a heatmap of the correlation matrix between assets.

    Args:
        correlation_matrix: Square DataFrame of correlations
                            (e.g. from calculate_correlation_matrix).
    """
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        correlation_matrix,
        annot=True,
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        fmt=".2f"
    )
    plt.title("Asset Correlation Heatmap")
    plt.tight_layout()
    plt.savefig("notebooks/correlation_heatmap.png")
    plt.show()
    
def plot_efficient_frontier(
    simulation_results: pd.DataFrame,
    min_variance_portfolio: dict,
    max_sharpe_portfolio: dict
) -> None:
    """
    Plot the simulated portfolios, highlighting the minimum variance
    and maximum Sharpe Ratio portfolios on the efficient frontier.

    Args:
        simulation_results: DataFrame from run_monte_carlo_simulation,
                            with 'return' and 'volatility' columns.
        min_variance_portfolio: Dict from find_minimum_variance_portfolio,
                                with 'return' and 'volatility' keys.
        max_sharpe_portfolio: Dict from find_max_sharpe_portfolio,
                              with 'return' and 'volatility' keys.
    """
    plt.figure(figsize=(10, 6))

    plt.scatter(
        simulation_results["volatility"],
        simulation_results["return"],
        c=simulation_results["return"] / simulation_results["volatility"],
        cmap="viridis",
        s=10,
        alpha=0.4
    )
    plt.colorbar(label="Return / Volatility")

    plt.scatter(
        min_variance_portfolio["volatility"],
        min_variance_portfolio["return"],
        color="red",
        marker="*",
        s=400,
        label="Minimum Variance"
    )

    plt.scatter(
        max_sharpe_portfolio["volatility"],
        max_sharpe_portfolio["return"],
        color="black",
        marker="*",
        s=400,
        label="Maximum Sharpe Ratio"
    )

    plt.xlabel("Volatility (Risk)")
    plt.ylabel("Expected Return")
    plt.title("Efficient Frontier")
    plt.legend()
    plt.tight_layout()
    plt.savefig("notebooks/efficient_frontier_final.png")
    plt.show()