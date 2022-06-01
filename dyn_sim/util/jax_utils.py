from functools import wraps
from typing import Any, Callable

import numpy as np
from jax import jit

TFunc = Callable[..., Any]


def jax_func(func: TFunc) -> TFunc:
    """Decorator for internal jax functions for System objects.

    Parameters
    ----------
    func : TFunc
        The function to be wrapped.

    Usage
    -----
    class ExampleSystem(System):

        @jax_func
        def example_func(self, *args, **kwargs) -> TYPE:
            ...
            return result

    # internal usage when AD is needed
    result = example_func(..., np_out=False)

    # external usage when we need a np array (like passing to gurobi)
    result = example_func(...)
    """

    @wraps(func)
    def wrapper(*args, np_out: bool = True, **kwargs) -> Any:
        result = jit(func, static_argnums=(0,))(*args, **kwargs)
        if np_out:
            return np.array(result)
        else:
            return result

    return wrapper
