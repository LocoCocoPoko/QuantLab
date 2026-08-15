import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from datetime import date

from src.quantlab.data.fetch import fetch_price_data
from src.quantlab.analytics.returns import calculate_log_returns
from src.quantlab.analytics.risk import (
    calculate_annualized_return,
    calculate_annualized_volatility,
    calculate_covariance_matrix,
    calculate_correlation_matrix,
)
from src.quantlab.optimization.portfolio import (
    find_minimum_variance_portfolio,
    find_max_sharpe_portfolio,
)

st.title("QuantLab")
st.write("Portfolio optimization and risk analysis.")

st.sidebar.header("Portfolio Settings")

tickers_input = st.sidebar.text_input(
    "Enter tickers (comma-separated)",
    value="AAPL, MSFT, GOOG"
)

start_date = st.sidebar.date_input("Start date", value=date(2021, 1, 1))
end_date = st.sidebar.date_input("End date", value=date(2024, 1, 1))

analyze_button = st.sidebar.button("Analyze Portfolio")

if analyze_button:
    tickers = [ticker.strip().upper() for ticker in tickers_input.split(",")]

    with st.spinner("Fetching data and running analysis..."):
        prices = fetch_price_data(
            tickers=tickers,
            start_date=str(start_date),
            end_date=str(end_date)
        )
        returns = calculate_log_returns(prices)
        annual_returns = calculate_annualized_return(returns)
        annual_volatility = calculate_annualized_volatility(returns)
        cov_matrix = calculate_covariance_matrix(returns)

        min_var = find_minimum_variance_portfolio(annual_returns, cov_matrix)
        max_sharpe = find_max_sharpe_portfolio(annual_returns, cov_matrix)

    actual_tickers = list(annual_returns.index)

    st.success("Analysis complete!")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Annualized Returns")
        st.write(annual_returns)

    with col2:
        st.subheader("Annualized Volatility")
        st.write(annual_volatility)

    st.subheader("Minimum Variance Portfolio")
    st.write(f"Return: {min_var['return']:.2%}")
    st.write(f"Volatility: {min_var['volatility']:.2%}")
    for ticker, weight in zip(actual_tickers, min_var["weights"]):
        st.write(f"{ticker}: {weight:.2%}")

    st.subheader("Maximum Sharpe Ratio Portfolio")
    st.write(f"Return: {max_sharpe['return']:.2%}")
    st.write(f"Volatility: {max_sharpe['volatility']:.2%}")
    for ticker, weight in zip(actual_tickers, max_sharpe["weights"]):
        st.write(f"{ticker}: {weight:.2%}")

    st.subheader("Correlation Heatmap")
    correlation = calculate_correlation_matrix(returns)
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(correlation, annot=True, cmap="coolwarm", vmin=-1, vmax=1, fmt=".2f", ax=ax)
    st.pyplot(fig)

    st.subheader("Cumulative Returns")
    cumulative = np.exp(returns.cumsum())
    fig2, ax2 = plt.subplots(figsize=(10, 6))
    for ticker in cumulative.columns:
        ax2.plot(cumulative.index, cumulative[ticker], label=ticker)
    ax2.legend()
    ax2.set_xlabel("Date")
    ax2.set_ylabel("Growth of $1")
    st.pyplot(fig2)

    st.subheader("Maximum Sharpe Portfolio Allocation")
    fig3, ax3 = plt.subplots(figsize=(8, 5))
    ax3.bar(actual_tickers, max_sharpe["weights"])
    ax3.set_ylabel("Weight")
    ax3.set_title("Maximum Sharpe Portfolio Allocation")
    for i, weight in enumerate(max_sharpe["weights"]):
        ax3.text(i, weight + 0.01, f"{weight:.1%}", ha="center")
    st.pyplot(fig3)