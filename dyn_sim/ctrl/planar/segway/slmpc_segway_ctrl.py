from typing import Callable, Optional, Union

import gurobipy as gp
import numpy as np

from dyn_sim.ctrl.mpc import SLMPC
from dyn_sim.sys.planar.segway import Segway
from dyn_sim.sys.sys_core import System

# constants
g = 9.80665  # gravitational acceleration


class SLMPCSegwayController(SLMPC):
    """A SLMPC controller for the segway system.

    Currently implemented constraints:
    [1] max speed constraints
    [2] safety constraints on angular tilt
    """

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
        v_safe: Optional[float] = None,
        ang_safe: Optional[float] = None,
        u_bnd: Optional[float] = None,
        print_t: bool = False,
    ) -> None:
        """Initialize a SLMPCSegwayController.

        Parameters
        ----------
        See SLMPC for full detail.

        v_safe : Optional[float], default=None
            Safe velocity limit.
        ang_safe : Optional[float], default=None
            Safe angle limit.
        u_bnd : Optional[float], default=None
            Bound on control input.
        """
        super(SLMPCSegwayController, self).__init__(
            sys,
            wc,
            mpc_N,
            mpc_P,
            mpc_Q,
            mpc_R,
            x_ref,
            u_ref,
            mpc_h,
            print_t=print_t,
        )
        self._sys: Segway
        self._v_safe = v_safe
        self._ang_safe = ang_safe
        self._u_bnd = u_bnd

    def _compute_ubar(self, x: np.ndarray) -> np.ndarray:
        """Computes ubar for linearized segway model.

        In this implementation, takes it to just be the last input.

        Parameters
        ----------
        x : np.ndarray, shape=(n,)
            The current state. Unused, included for API compliance.

        Returns
        -------
        ubar : np.ndarray, shape=(m,)
            The control input about which to linearize.
        """
        if self._mem.initialized:
            return self._u_mem[-1]
        else:
            return np.array([0.0])

    def _state_constrs(self) -> None:
        """Adds constraint to each state in planning subproblem."""
        x_var = self._gp_xvar
        N = self._N

        # angles
        if self._ang_safe is not None:
            for i in range(N + 1):
                xi = x_var[i, :]
                self._gpmodel.addConstr(xi[1] <= self._ang_safe)
                self._gpmodel.addConstr(xi[1] >= -self._ang_safe)

        # speeds
        if self._v_safe is not None:
            for i in range(N + 1):
                xi = x_var[i, :]
                self._gpmodel.addConstr(xi[2] <= self._v_safe)
                self._gpmodel.addConstr(xi[2] >= -self._v_safe)

    def _input_constrs(self) -> None:
        """Adds constraint to each input in planning subproblem."""
        u_var = self._gp_uvar
        N = self._N

        if self._u_bnd is not None:
            for i in range(N):
                ui = u_var[i, :]
                self._gpmodel.addConstr(ui <= self._u_bnd)
                self._gpmodel.addConstr(ui >= -self._u_bnd)
