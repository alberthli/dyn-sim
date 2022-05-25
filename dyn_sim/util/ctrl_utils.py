from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import List

import numpy as np


@dataclass
class AbstractDataclass(ABC):
    """Abstract dataclass."""

    def __new__(cls, *args, **kwargs) -> "AbstractDataclass":
        """See: stackoverflow.com/questions/60590442."""
        if cls == AbstractDataclass or cls.__bases__[0] == AbstractDataclass:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)


@dataclass  # type: ignore[misc]
class MemoryBank(AbstractDataclass):
    """Abstract MemoryBank dataclass.

    Subclasses will process the memory components in different ways. Lists are used instead of numpy arrays for fast appending.

    Fields
    ------
    t_mem : List[float]
        List of past times with computations.
    x_mem : List[np.ndarray]
        List of past remembered states of the system.
    u_mem : List[np.ndarray]
        List of past applied control inputs of the system.
    """

    t_mem: List[float] = field(default_factory=list)
    x_mem: List[np.ndarray] = field(default_factory=list)
    u_mem: List[np.ndarray] = field(default_factory=list)

    @abstractmethod
    def reset(self) -> None:
        """Reset the memory bank."""


@dataclass
class BWLCMemory(MemoryBank):
    """Lightweight dataclass for holding memory of a BWLC.

    Fields
    ------
    t_mem : List[float]
        List of most recent computation time.
    x_mem : List[np.ndarray]
        Unused. Inherited for API compliance.
    u_mem : List[np.ndarray]
        List of most recent computed control input.
    """

    @property
    def initialized(self) -> bool:
        """Flag for whether memory is initialized."""
        return len(self.t_mem) > 0 and len(self.u_mem) > 0

    def rem_t(self, t: float) -> None:
        """Remember the remembered time.

        Parameters
        ----------
        t : float
            Time to remember.
        """
        self.t_mem = [t]

    def rem_x(self, x: np.ndarray) -> None:
        """Set the remembered ctrl. Does nothing.

        Parameters
        ----------
        x : np.ndarray, shape=(x,)
            State to remember.
        """

    def rem_u(self, u: np.ndarray) -> None:
        """Remember the remembered ctrl.

        Parameters
        ----------
        u : np.ndarray, shape=(m,)
            Control input to remember.
        """
        self.u_mem = [u]

    def reset(self) -> None:
        """Reset the time and control input."""
        self.t_mem = []
        self.u_mem = []


@dataclass
class FullMemory(BWLCMemory):
    """Dataclass for holding full memory of a BWLC.

    Fields
    ------
    t_mem : List[float]
        List of all past times with computations.
    x_mem : List[np.ndarray]
        List of all past remembered states of the system.
    u_mem : List[np.ndarray]
        List of all past applied control inputs of the system.
    """

    @property
    def initialized(self) -> bool:
        """Flag for whether memory is initialized."""
        lt = len(self.t_mem)
        lx = len(self.x_mem)
        lu = len(self.u_mem)
        return lt > 0 and lx > 0 and lu > 0

    def rem_t(self, t: float) -> None:
        """Add time to memory.

        Parameters
        ----------
        t : float
            Time to remember.
        """
        self.t_mem.append(t)

    def rem_x(self, x: np.ndarray) -> None:
        """Set the remembered ctrl.

        Parameters
        ----------
        x : np.ndarray, shape=(x,)
            State to remember.
        """
        self.x_mem.append(x)

    def rem_u(self, u: np.ndarray) -> None:
        """Set the remembered ctrl.

        Parameters
        ----------
        u : np.ndarray, shape=(m,)
            Control input to remember.
        """
        self.u_mem.append(u)

    def reset(self) -> None:
        """Reset the time, state, and control memory."""
        self.t_mem = []
        self.x_mem = []
        self.u_mem = []
