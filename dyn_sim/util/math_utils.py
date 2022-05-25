import numpy as np


def is_pd(K: np.ndarray) -> bool:
    """Checks whether K is PD."""
    assert np.array_equal(K, K.T)
    try:
        np.linalg.cholesky(K)
        return True
    except np.linalg.linalg.LinAlgError:
        return False
