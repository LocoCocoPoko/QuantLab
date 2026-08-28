import numpy as np
import pandas as pd

from src.quantlab.analytics.risk import calculate_annualized_volatility


def test_annualized_volatility_known_values():
    """
    Test annualized volatility against a hand-calculable example.
    """
    # Simple, known daily returns
    returns = pd.DataFrame({
        "TEST": [0.01, -0.02, 0.015, -0.005, 0.02]
    })

    result = calculate_annualized_volatility(returns)

    daily_std = returns["TEST"].std()
    expected = daily_std * np.sqrt(252)

    assert np.isclose(result["TEST"], expected)