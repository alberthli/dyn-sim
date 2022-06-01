from functools import wraps

import numpy as np
from jax import jit


def jax_func(func):
    @wraps(func)
    def wrapper(*args, np_out: bool = True, **kwargs):
        result = jit(func, static_argnums=(0,))(*args, **kwargs)
        if np_out:
            return np.array(result)
        else:
            return result

    return wrapper
