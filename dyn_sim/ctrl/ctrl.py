from abc import ABC, abstractmethod

import numpy as np

from dyn_sim.sys.dyn_sys import System


class Controller(ABC):
    """Abstract class for controllers."""

    def __init__(self, sys: System) -> None:
        """Initialize a controller.

        Parameters
        ----------
        sys : System
            Dynamical system to be controlled.
        """
        super(Controller, self).__init__()

        self._sys = sys
        self._n = sys._n
        self._m = sys._m

    @abstractmethod
    def ctrl(self, t: float, x: np.ndarray) -> np.ndarray:
        """Control law.

        Parameters
        ----------
        t: float
            Time.
        x: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        u: np.ndarray, shape=(m,)
            Control input.
        """

    def reset(self) -> None:
        """Resets controller internals. Used for controllers with memory."""


class BWLC(Controller):
    """Abstract class for bandwidth-limited controllers (BWLCs)."""

    def __init__(self, sys: System, wc: float) -> None:
        """Initialize a BWLC.

        Parameters
        ----------
        sys : System
            Dynamical system to be controlled.
        wc : float
            The control frequency (Hz).
        """
        assert wc > 0.0

        super(BWLC, self).__init__()
        self._wc = wc
        self._dt = 1.0 / wc  # convenience dt for given wc
        self._tmem = None  # memory for last ctrl update time
        self._u = None  # memory for current ZOH ctrl input

    @abstractmethod
    def _ctrl_update(self, t: float, x: np.ndarray) -> np.ndarray:
        """Update the control input.

        This function is called internally by the ctrl(...) method.

        Parameters
        ----------
        t: float
            Time.
        x: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        u: np.ndarray, shape=(m,)
            Control input.
        """

    def ctrl(self, t: float, x: np.ndarray) -> np.ndarray:
        """Bandwidth-limited control law.

        This function ensures that the control updates are only applied at the rate of the controller bandwidth, zero-order holding them otherwise.

        Parameters
        ----------
        See parent function.

        Returns
        -------
        See parent function.
        """
        assert x.shape == (self._n,)

        # memory initialization
        if self._tmem is None:
            self._tmem = t
            self._u = self._ctrl_update(t, x)

        # control update if enough time has elapsed
        elif (t - self._tmem) > self._dt:
            self._tmem = self._tmem + self._dt
            self._u = self._ctrl_update(t, x)

        return self._u

    def reset(self) -> None:
        """Resets the controller memory."""
        self._tmem = None
        self._u = None
