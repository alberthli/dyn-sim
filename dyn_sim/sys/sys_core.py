from abc import ABC, ABCMeta, abstractmethod

import numpy as np
from matplotlib.axes import Axes


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
    def dyn(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
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
        dx : np.ndarray, shape=(n,)
            Time derivative of current state.
        """

    @abstractmethod
    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearized autonomous dynamics about (x, u).

        Parameters
        ----------
        x : np.ndarray, shape=(n,)
            State.
        u : np.ndarray, shape=(m,)
            Control input.

        Returns
        -------
        _A : np.ndarray, shape=(n, n)
            Linearized autonomous dynamics about (x, u).
        """

    @abstractmethod
    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearized control dynamics about (x, u).

        Parameters
        ----------
        x : np.ndarray, shape=(n,)
            State.
        u : np.ndarray, shape=(m,)
            Control input.

        Returns
        -------
        _B : np.ndarray, shape=(n, m)
            Linearized control dynamics about (x, u).
        """

    @abstractmethod
    def draw(self, ax: Axes, x: np.ndarray) -> None:
        """Draw the current state on the specified Axes object.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
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
    def fdyn(self, t: float, x: np.ndarray) -> np.ndarray:
        """Drift of the system.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray, shape=(n,)
            Current state.

        Returns
        -------
        f_val : np.ndarray, shape=(n,)
            Value of the drift of the system.
        """

    @abstractmethod
    def gdyn(self, t: float, x: np.ndarray) -> np.ndarray:
        """Actuation matrix.

        Parameters
        ----------
        t : float
            Current time.
        x : np.ndarray, shape=(n,)
            Current state.

        Returns
        -------
        g_val : np.ndarray, shape=(n, m)
            Value of the actuation matrix of the system.
        """

    def dyn(self, t: float, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """See parent docstring."""
        assert x.shape == (self._n,)
        assert u.shape == (self._m,)
        return self.fdyn(t, x) + self.gdyn(t, x) * u
