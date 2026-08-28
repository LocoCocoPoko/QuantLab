import numpy as np
import pandas as pd

from src.quantlab.analytics.returns import calculate_log_returns


def test_calculate_log_returns_known_values():
    """
    Test that log returns are calculated correctly using known,
    hand-verifiable price data.
    """
    prices = pd.DataFrame({
        "TEST": [100, 103, 101]
    })

    result = calculate_log_returns(prices)

    expected_day2 = np.log(103 / 100)
    expected_day3 = np.log(101 / 103)

    assert np.isclose(result["TEST"].iloc[0], expected_day2)
    assert np.isclose(result["TEST"].iloc[1], expected_day3)