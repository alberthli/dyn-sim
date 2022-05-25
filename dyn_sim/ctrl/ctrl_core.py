from abc import ABC, abstractmethod
from typing import Union

import numpy as np

from dyn_sim.sys.sys_core import System
from dyn_sim.util.ctrl_utils import BWLCMemory, FullMemory, MemoryBank


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
        self._sys: System = sys
        self._n = sys._n
        self._m = sys._m

    @abstractmethod
    def ctrl(self, t: float, x: np.ndarray) -> np.ndarray:
        """Control law.

        Parameters
        ----------
        t : float
            Time.
        x : np.ndarray, shape=(n,)
            State.

        Returns
        -------
        u: np.ndarray, shape=(m,)
            Control input.
        """


class MemoryController(Controller):
    """Abstract class for controllers requiring memory."""

    _mem: MemoryBank = NotImplemented  # required MemoryBank attribute

    def __init__(self, sys: System) -> None:
        """Initialize a memory controller."""
        super(MemoryController, self).__init__(sys)

    def reset(self) -> None:
        """Resets the controller memory."""
        self._mem.reset()


class BWLC(MemoryController):
    """Abstract class for bandwidth-limited controllers (BWLCs)."""

    def __init__(self, sys: System, wc: float, print_t: bool = False) -> None:
        """Initialize a BWLC.

        Parameters
        ----------
        sys : System
            Dynamical system to be controlled.
        wc : float
            The control frequency (Hz).
        print_t : bool
            Flag indicating whether to print time of control computation.
        """
        super(BWLC, self).__init__(sys)
        self._wc = wc
        self._dt = 1.0 / wc  # convenience dt for given wc
        self._mem: BWLCMemory = BWLCMemory()
        self._print_t = print_t

    @property
    def _t_last(self) -> float:
        return self._mem.t_mem[-1]

    @property
    def _t_mem(self) -> Union[float, np.ndarray]:
        """Get time memory.

        Returns
        -------
        t : Union[float, np.ndarray], shape=(1,) OR (T,)
            Time memory.
        """
        if isinstance(self._mem, FullMemory):
            return np.array(self._mem.t_mem)
        elif isinstance(self._mem, BWLCMemory):
            return self._mem.t_mem[-1]
        else:
            raise NotImplementedError

    @property
    def _u_mem(self) -> np.ndarray:
        """Get control memory.

        Returns
        -------
        u : np.ndarray, shape=(m,) OR (T, m)
            Control memory.
        """
        if isinstance(self._mem, FullMemory):
            return np.array(self._mem.u_mem)
        elif isinstance(self._mem, BWLCMemory):
            return self._mem.u_mem[-1]
        else:
            raise NotImplementedError

    @abstractmethod
    def _ctrl_update(self, t: float, x: np.ndarray) -> np.ndarray:
        """Update the control input.

        This function is called internally by the ctrl(...) method.

        Parameters
        ----------
        t : float
            Time.
        x : np.ndarray, shape=(n,)
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
        if not self._mem.initialized:
            self._mem.rem_t(t)
            self._mem.rem_x(x)
            self._mem.rem_u(self._ctrl_update(t, x))
            return self._u_mem[-1]

        # control update if enough time has elapsed
        elif (t - self._t_last) > self._dt:
            self._mem.rem_t(self._t_last + self._dt)
            self._mem.rem_x(x)
            self._mem.rem_u(self._ctrl_update(t, x))
            if self._print_t:
                print(t)

        return self._u_mem[-1]


class FullMemoryBWLC(BWLC):
    """Abstract class for BWLC with full memory.

    In particular, FullMemoryBWLCs store the entire time, state, and control history of the controller for the whole run.
    """

    def __init__(self, sys: System, wc: float, print_t: bool = False) -> None:
        """Initialize a FullMemoryBWLC.

        Parameters
        ----------
        See BWLC.
        """
        super(FullMemoryBWLC, self).__init__(sys, wc, print_t=print_t)
        self._mem: FullMemory = FullMemory()

    @property
    def _x_mem(self) -> np.ndarray:
        """Get state memory.

        Returns
        -------
        x : np.ndarray, shape=(T, n)
            States in memory.
        """
        return np.array(self._mem.x_mem)
