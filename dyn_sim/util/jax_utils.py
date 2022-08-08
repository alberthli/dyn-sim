from functools import wraps
from typing import Any, Callable

import numpy as np
from jax import jit

TFunc = Callable[..., Any]


def jax_func(func: TFunc) -> TFunc:
    """Decorator for internal jax functions (any func needing auto diff).

    Does two things:
    (1) Assumes that the jax func is an object method so the first argument is self. The function is jit'd but we treat self as a static argument and only autodiff through the other args.
    (2) Adds an np_out flag to the jax function (for convenience). The flag is set to True by default. If True, then the output of the function is cast to a numpy array. If False, then the output is passed as a jnp array. The purpose is so that if the end user calls jax functions outside of a context where they need AD, they won't be confused by jax errors resulting from incompatibility between jnp and np arrays (ex: if you are interfacing with gurobi). The burden to remember to be explicit about the outputs being jnp arrays falls to the designer, who must remember to pass np_out=False.

    Parameters
    ----------
    func : TFunc
        The function to be wrapped.
    obj_method : bool, default=True
        Flag indicating whether the function is an object method.

    Usage
    -----
    class ExampleSystem(System):

        @jax_func
        def example_func(self, *args, **kwargs) -> jnp.ndarray:
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
