import numpy as np

from src.quantlab.optimization.portfolio import generate_random_weights


def test_random_weights_sum_to_one():
    """
    Test that randomly generated portfolio weights always sum to 1.0,
    regardless of the random values generated.
    """
    weights = generate_random_weights(5)

    assert np.isclose(np.sum(weights), 1.0)


def test_random_weights_correct_length():
    """
    Test that the number of weights matches the number of assets requested.
    """
    weights = generate_random_weights(4)

    assert len(weights) == 4