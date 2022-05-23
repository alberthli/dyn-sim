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
        assert wc > 0
        super(BWLC, self).__init__()
        self._wc = wc
