from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass
class MemoryBank(ABC):
    """Abstract dataclass to prevent instantiation.

    See: stackoverflow.com/questions/60590442.
    """

    def __new__(cls, *args, **kwargs) -> None:
        """Abstract class instantiation prevention function."""
        if cls == MemoryBank or cls.__bases__[0] == MemoryBank:
            raise TypeError("Cannot instantiate abstract class.")
        return super().__new__(cls)

    @abstractmethod
    def reset(self) -> None:
        """Reset the memory bank."""


@dataclass
class BWLCMemory(MemoryBank):
    """Lightweight dataclass for holding memory of a BWLC.

    Fields
    ------
    t_mem : float
        Last time the controller updated.
    u_mem : np.ndarray, shape=(m,)
        Last control input computed.
    """

    t_mem: float = None
    u_mem: np.ndarray = None

    @property
    def initialized(self) -> bool:
        """Flag for whether memory is initialized."""
        return self.t_mem is not None and self.u_mem is not None

    def set_t(self, t: float) -> None:
        """Set the remembered time.

        Parameters
        ----------
        t : float
            Time to remember.
        """
        self.t_mem = t

    def set_u(self, u: np.ndarray) -> None:
        """Set the remembered ctrl.

        Parameters
        ----------
        u : np.ndarray, shape=(m,)
            Control input to remember.
        """
        self.u_mem = u

    def reset(self) -> None:
        """Reset the time and control input."""
        self.t_mem = None
        self.u_mem = None


@dataclass
class FullMemory(MemoryBank):
    """Dataclass for holding full memory of a BWLC.

    Stored as lists for cheap dynamic memory.

    Fields
    ------
    t_mem : List[float]
        List of all past times with computations.
    x_mem : List[np.ndarray]
        List of all past remembered states of the system.
    u_mem : List[np.ndarray]
        List of all past applied control inputs of the system.
    """

    t_mem: List[float] = None
    x_mem: List[np.ndarray] = None
    u_mem: List[np.ndarray] = None

    @property
    def initialized(self) -> bool:
        """Flag for whether memory is initialized."""
        return (
            self.t_mem is not None and self.u_mem is not None and self.x_mem is not None
        )

    def add_t(self, t: float) -> None:
        """Add time to memory.

        Parameters
        ----------
        t : float
            Time to remember.
        """
        if self.t_mem is None:
            self.t_mem = [t]
        else:
            self.t_mem.append(t)

    def add_x(self, x: np.ndarray) -> None:
        """Set the remembered ctrl.

        Parameters
        ----------
        x : np.ndarray, shape=(x,)
            State to remember.
        """
        if self.x_mem is None:
            self.x_mem = [x]
        else:
            self.x_mem.append(x)

    def add_u(self, u: np.ndarray) -> None:
        """Set the remembered ctrl.

        Parameters
        ----------
        u : np.ndarray, shape=(m,)
            Control input to remember.
        """
        if self.u_mem is None:
            self.u_mem = [u]
        else:
            self.u_mem.append(u)

    def reset(self) -> None:
        """Reset the time, state, and control memory."""
        self.t_mem = None
        self.x_mem = None
        self.u_mem = None
