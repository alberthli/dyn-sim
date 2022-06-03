from abc import ABC, ABCMeta, abstractmethod
from typing import Optional, Union

import jax.numpy as jnp
import numpy as np
from jax import jacobian
from matplotlib.axes import Axes
from mpl_toolkits.mplot3d.axes3d import Axes3D

from dyn_sim.util.jax_utils import jax_func


class System(ABC):
    """Abstract class for dynamical systems."""

    def __init__(self, n: int, m: int, is3d: bool) -> None:
        """Initialize a system.

        Parameters
        ----------
        n : int
            Dimension of state.
        m : int
            Dimension of control input.
        is_3d : bool
            Flag indicating whether the system is spatially 3D or not.
        """
        super(System, self).__init__()

        assert n > 0
        assert m >= 0

        self._n = n
        self._m = m
        self._is3d = is3d

    @abstractmethod
    @jax_func
    def dyn(self, t: float, x: np.ndarray, u: np.ndarray) -> jnp.ndarray:
        """Dynamics of the system.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray, shape=(n,)
            Current state.
        u : np.ndarray, shape=(m,)
            Current control input.

        Returns
        -------
        dx : jnp.ndarray, shape=(n,)
            Time derivative of current state.
        """

    @jax_func
    def A(self, t: float, x: np.ndarray, u: np.ndarray) -> jnp.ndarray:
        """Linearized autonomous dynamics about (x, u).

        Parameters
        ----------
        t : float
            Time.
        x : np.ndarray, shape=(n,)
            State.
        u : np.ndarray, shape=(m,)
            Control input.

        Returns
        -------
        _A : jnp.ndarray, shape=(n, n)
            Linearized autonomous dynamics about (x, u).
        """
        dyn_jnp = lambda _x, _u: self.dyn(t, _x, _u, np_out=False)
        _A = jacobian(dyn_jnp, argnums=0)(x, u)
        return _A

    @jax_func
    def B(self, t: float, x: np.ndarray, u: np.ndarray) -> jnp.ndarray:
        """Linearized control dynamics about (x, u).

        Parameters
        ----------
        t : float
            Time.
        x : np.ndarray, shape=(n,)
            State.
        u : np.ndarray, shape=(m,)
            Control input.

        Returns
        -------
        _B : jnp.ndarray, shape=(n, m)
            Linearized control dynamics about (x, u).
        """
        dyn_jnp = lambda _x, _u: self.dyn(t, _x, _u, np_out=False)
        _B = jacobian(dyn_jnp, argnums=1)(x, u)
        return _B

    @abstractmethod
    def draw(self, ax: Optional[Union[Axes, Axes3D]], x: np.ndarray) -> None:
        """Draw the current state on the specified Axes object.

        Parameters
        ----------
        ax : Optional[Union[Axes, Axes3D]]
            Axes object on which to draw the system.
        x : np.ndarray, shape=(n,)
            Current state of the system.
        """


class CtrlAffineSystem(System, metaclass=ABCMeta):
    """Abstract class for control affine system."""

    def __init__(self, n: int, m: int, is3d: bool) -> None:
        """Initialize a control affine system.

        The dynamics should have the form:
        dx = f(t, x) + g(t, x)u

        Parameters
        ----------
        n : int
            Dimension of state.
        m : int
            Dimension of control input.
        is_3d : bool
            Flag indicating whether the system is spatially 3D or not.
        """
        super(CtrlAffineSystem, self).__init__(n, m, is3d)

    @abstractmethod
    @jax_func
    def fdyn(self, t: float, x: np.ndarray) -> jnp.ndarray:
        """Drift of the system.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray, shape=(n,)
            Current state.

        Returns
        -------
        f_val : jnp.ndarray, shape=(n,)
            Value of the drift of the system.
        """

    @abstractmethod
    @jax_func
    def gdyn(self, t: float, x: np.ndarray) -> jnp.ndarray:
        """Actuation matrix.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray, shape=(n,)
            Current state.

        Returns
        -------
        g_val : jnp.ndarray, shape=(n, m)
            Value of the actuation matrix of the system.
        """

    @jax_func
    def dyn(self, t: float, x: np.ndarray, u: np.ndarray) -> jnp.ndarray:
        """See parent docstring."""
        assert x.shape == (self._n,)
        assert u.shape == (self._m,)
        return self.fdyn(t, x, np_out=False) + self.gdyn(t, x, np_out=False) @ u
