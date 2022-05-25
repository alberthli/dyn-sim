from typing import Callable, Optional, Union

import gurobipy as gp
import numpy as np

from dyn_sim.ctrl.mpc import SLMPC
from dyn_sim.sys.spatial.quad import Quadrotor
from dyn_sim.sys.sys_core import System

# constants
g = 9.80665  # gravitational acceleration


class SLMPCQuadController(SLMPC):
    """A SLMPC controller for the quadrotor system."""

    def __init__(
        self,
        sys: System,
        wc: Optional[float],
        mpc_N: int,
        mpc_P: Optional[np.ndarray],
        mpc_Q: np.ndarray,
        mpc_R: np.ndarray,
        x_ref: Callable[[float, np.ndarray, SLMPC], Union[np.ndarray, gp.MVar]],
        u_ref: Callable[[float, np.ndarray, SLMPC], Union[np.ndarray, gp.MVar]],
        mpc_h: Optional[float] = None,
    ) -> None:
        """Initialize a SLMPCQuadController."""
        super(SLMPCQuadController, self).__init__(
            sys,
            wc,
            mpc_N,
            mpc_P,
            mpc_Q,
            mpc_R,
            x_ref,
            u_ref,
            mpc_h,
        )
        self._sys: Quadrotor

    def _compute_ubar(self, x: np.ndarray) -> np.ndarray:
        """Computes ubar for linearized quad model.

        In this implementation, takes it to just be the input corresponding to gravity-compensating thrust for an upright quad.

        Parameters
        ----------
        x : np.ndarray, shape=(n,)
            The current state. Unused, included for API compliance.

        Returns
        -------
        ubar : np.ndarray, shape=(m,)
            The control input about which to linearize.
        """
        return np.array([self._sys._mass * g, 0, 0, 0])
