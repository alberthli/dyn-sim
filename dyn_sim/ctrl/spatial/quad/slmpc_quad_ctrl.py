from typing import Callable, Optional, Union

import gurobipy as gp
import numpy as np

from dyn_sim.ctrl.mpc import SLMPC
from dyn_sim.sys.spatial.quad import Quadrotor
from dyn_sim.sys.sys_core import System

# constants
g = 9.80665  # gravitational acceleration


class SLMPCQuadController(SLMPC):
    """A SLMPC controller for the quadrotor system.

    It is designed to be able to track a reference (x, u) trajectory and can also enforce state and input constraints (must be implemented in the _state_constrs() and _input_constrs() functions in this file).

    Currently implemented constraints:
    [1] max speed constraints
    [2] safety constraints on angular tilt
    [3] non-negative squared rotor speeds
    [4] non-negative total thrust

    Future:
    [1] Obstacle avoidance constraints
    """

    def __init__(
        self,
        sys: System,
        wc: Optional[float],
        mpc_N: int,
        mpc_P: np.ndarray,
        mpc_Q: np.ndarray,
        mpc_R: np.ndarray,
        x_ref: Callable[[float, np.ndarray, SLMPC], Union[np.ndarray, gp.MVar]],
        u_ref: Callable[[float, np.ndarray, SLMPC], Union[np.ndarray, gp.MVar]],
        mpc_h: Optional[float] = None,
        v_safe: Optional[float] = None,
        ang_safe: Optional[float] = None,
        print_t: bool = False,
    ) -> None:
        """Initialize a SLMPCQuadController.

        Parameters
        ----------
        See SLMPC for full detail.

        v_safe : Optional[float], default=None
            Safe velocity limit.
        ang_safe : Optional[float], default=None
            Safe angle limit.
        """
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
            print_t=print_t,
        )
        self._sys: Quadrotor
        self._v_safe = v_safe
        self._ang_safe = ang_safe

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

    def _state_constrs(self) -> None:
        """Adds constraint to each state in planning subproblem."""
        x_var = self._gp_xvar
        N = self._N

        # angles
        if self._ang_safe is not None:
            for i in range(N + 1):
                xi = x_var[i, :]
                self._gpmodel.addConstr(xi[3] <= self._ang_safe)
                self._gpmodel.addConstr(xi[3] >= -self._ang_safe)
                self._gpmodel.addConstr(xi[4] <= self._ang_safe)
                self._gpmodel.addConstr(xi[4] >= -self._ang_safe)

        # speeds
        if self._v_safe is not None:
            for i in range(N + 1):
                xi = x_var[i, :]
                v_aux = self._gpmodel.addVar()
                self._gpmodel.addConstr(v_aux == gp.norm(xi[6:9], 2))
                self._gpmodel.addConstr(v_aux <= self._v_safe)

    def _input_constrs(self) -> None:
        """Adds constraint to each input in planning subproblem."""
        u_var = self._gp_uvar
        N = self._N

        for i in range(N):
            ui = u_var[i, :]
            self._gpmodel.addConstr(ui[0] >= 0)  # thrust
            self._gpmodel.addConstr(self._sys.invV() @ ui >= 1e-3)  # rotor speed
