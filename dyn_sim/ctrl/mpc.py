from abc import abstractmethod
from typing import Callable, Optional, Union

import gurobipy as gp
import numpy as np
from gurobipy import GRB

from dyn_sim.ctrl.ctrl_core import FullMemoryBWLC
from dyn_sim.sys.sys_core import System
from dyn_sim.util.math_utils import is_pd


class SLMPC(FullMemoryBWLC):
    """Bandwidth-Limited Successive Linearization MPC.

    Some assumptions of the form of this controller:
    (1) quadratic cost using time-invariant P, Q, R matrices;
    (2) fixed time discretization and planning horizon;
    (3) planning dynamics are locally-linearized from the exact model.

    Future TODOs
    ------------
    [1] Maybe abstract out the MPC framework slightly, since the major difference between MPC formulations is the form of the dynamics constraints.
    [2] ^ related to [1], separate out the constraint and cost calculation.
    """

    def __init__(
        self,
        sys: System,
        wc: Optional[float],
        mpc_N: int,
        mpc_P: Optional[np.ndarray],
        mpc_Q: np.ndarray,
        mpc_R: np.ndarray,
        x_ref: Callable[[float, np.ndarray, "SLMPC"], Union[np.ndarray, gp.MVar]],
        u_ref: Callable[[float, np.ndarray, "SLMPC"], Union[np.ndarray, gp.MVar]],
        mpc_h: Optional[float] = None,
        print_t: bool = False,
    ) -> None:
        """Initialize a SLMPC.

        Parameters
        ----------
        sys : System
            Dynamical system to be controlled.
        wc : Optional[float]
            The control frequency (Hz). If None, run in non-bandwidth-limited mode.
        mpc_N : int
            The number of subproblem steps.
        mpc_P : Optional[np.ndarray], shape=(n, n)
            The terminal state cost weighting matrix. If None, internally computed as the solution to the discrete algebraic Riccati equation.
        mpc_Q : np.ndarray, shape=(n, n)
            The stage state cost weighting matrix.
        mpc_R : np.ndarray, shape=(m, m)
            The stage input cost weighting matrix.
        x_ref : Callable[
            [float, np.ndarray, "SLMPC"], Union[np.ndarray, gp.MVar]
        ]
            Reference function for state. Takes in time, current state, and SLMPC so that complicated reference trajectories can be computed. Returns the desired state at input time t. Can also return decision variable objects - useful if you want to enforce smoothness constraints without explicitly computing a trajectory using a method like collocation (e.g. penalize the difference between two consecutive states).
        u_ref : Callable[
            [float, np.ndarray, "SLMPC"], Union[np.ndarray, gp.MVar]
        ]
            See x_ref, same but for control input.
        mpc_h : Optional[float]
            The time-discretization (sec) of the MPC subproblem. By default assumed to be dt (see below).
        print_t : bool
            Flag indicating whether to print time of control computation.
        """
        if wc is None:
            super(SLMPC, self).__init__(sys, 1, print_t=print_t)  # 1=dummy init
            self._wc = 0.0  # dummy value
            self._dt = 0.0
        else:
            assert wc > 0.0
            super(SLMPC, self).__init__(sys, wc, print_t=print_t)
        assert mpc_N > 0
        assert isinstance(mpc_N, int)
        assert mpc_Q.shape == (self._n, self._n)
        assert mpc_R.shape == (self._m, self._m)
        assert np.array_equal(mpc_Q, mpc_Q.T)  # symmetry
        assert np.array_equal(mpc_R, mpc_R.T)
        assert np.all(np.linalg.eigvals(mpc_Q) >= 0.0)  # PSD
        assert is_pd(mpc_R)  # PD check
        if mpc_P is not None:
            assert mpc_P.shape == (self._n, self._n)  # shape
            assert np.array_equal(mpc_P, mpc_P.T)  # symmetry
            assert np.all(np.linalg.eigvals(mpc_P) >= 0.0)  # PSD

        # mpc variables
        self._N = mpc_N
        self._P = mpc_P
        self._Q = mpc_Q
        self._R = mpc_R
        if mpc_h is not None:
            self._h = mpc_h
        else:
            self._h = self._dt

        # reference functions
        self._x_ref = x_ref
        self._u_ref = u_ref

        # gurobi setup #
        # environment
        env = gp.Env(empty=True)
        env.setParam("OutputFlag", 0)  # suppressing output
        env.start()
        self._env = env

        # initializing model and decision variables for subproblem
        # [NOTE] decision variables are indexed like so:
        #   x_var[i, j] gives the ith planning state's jth component.
        #   u_var works similarly. Note that there are (N + 1) x_vars and
        #   only N u_vars, since we have an initial state.
        self._gpmodel = gp.Model("mpc_problem", env=env)
        x_var = self._gpmodel.addMVar(
            (self._N + 1, self._n),
            name="x_var",
            lb=-np.inf,
        )
        u_var = self._gpmodel.addMVar(
            (self._N, self._m),
            name="u_var",
            lb=-np.inf,
        )
        self._gpmodel.update()
        self._gp_xvar = x_var
        self._gp_uvar = u_var

    def ctrl(self, t: float, x: np.ndarray) -> np.ndarray:
        """SLMPC control scheme.

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
        # bandwidth-limited
        if self._wc > 0.0:
            return super(SLMPC, self).ctrl(t, x)
        # non-bandwidth-limited
        else:
            return self._ctrl_update(t, x)

    def _ctrl_update(self, t: float, x: np.ndarray) -> np.ndarray:
        """The internal SLMPC control update.

        Parameters
        ----------
        See ctrl().

        Returns
        -------
        See ctrl().
        """
        # unpacking variables
        n = self._n
        N = self._N
        h = self._h
        P = self._P
        Q = self._Q
        R = self._R

        # continuous-time linearized dynamics
        ubar = self._compute_ubar(x)
        A = self._sys.A(x, ubar)
        B = self._sys.B(x, ubar)
        feq = self._sys.dyn(t, x, ubar)  # offset for Taylor expansion

        # discrete-time linearized dynamics
        Ak = np.eye(n) + h * A
        Bk = h * B
        Ck = -h * (A @ x + B @ ubar - feq)

        # constructing dynamics constraints
        x_var = self._gp_xvar
        u_var = self._gp_uvar
        for i in range(N + 1):
            if i == 0:
                self._gpmodel.addConstr(x_var[i, :] == x)
            else:
                x_next = x_var[i, :]
                x_now = x_var[i - 1, :]
                u_now = u_var[i - 1, :]
                self._gpmodel.addConstr(x_next == Ak @ x_now + Bk @ u_now + Ck)

        # adding problem-specific constraints (must be implemented)
        self._state_constrs()
        self._input_constrs()

        # constructing the cost function
        cost_terms = []
        for i in range(N):
            # current planning state and ctrl
            xi = x_var[i, :]
            ui = u_var[i, :]

            # state and ctrl references
            tr = t + i * h
            xr = self._x_ref(tr, x, self)
            ur = self._u_ref(tr, x, self)

            # [May 24, 2022] gurobi limitations for MVars means must have aux decision variables in order to parse this expression correctly. See the help post: support.gurobi.com/hc/en-us/articles/360038943132.
            x_aux = self._gpmodel.addMVar(self._n, lb=-np.inf)
            u_aux = self._gpmodel.addMVar(self._m, lb=-np.inf)
            self._gpmodel.addConstr(x_aux == xi - xr)
            self._gpmodel.addConstr(u_aux == ui - ur)
            cost_terms.append(x_aux @ Q @ x_aux)
            cost_terms.append(u_aux @ R @ u_aux)

        # terminal cost
        xf = x_var[-1, :]
        tr = t + N * h
        xr = self._x_ref(tr, x, self)
        ur = self._u_ref(tr, x, self)
        x_aux = self._gpmodel.addMVar(self._n, lb=-np.inf)
        self._gpmodel.addConstr(x_aux == xf - xr)
        cost_terms.append(x_aux @ P @ x_aux)

        cost = sum(cost_terms)
        self._gpmodel.setObjective(cost, GRB.MINIMIZE)
        self._gpmodel.update()

        # solving the subproblem
        self._gpmodel.optimize()
        if self._gpmodel.Status in [3, 4, 5]:
            raise RuntimeError("Infeasible or unbounded subproblem!")
        else:
            u = u_var[0, :].X.flatten()
        self._reset_gpmodel()

        return u

    @abstractmethod
    def _state_constrs(self) -> None:
        """Adds constraint to each state in planning subproblem."""

    @abstractmethod
    def _input_constrs(self) -> None:
        """Adds constraint to each input in planning subproblem."""

    @abstractmethod
    def _compute_ubar(self, x: np.ndarray) -> np.ndarray:
        """Computes ubar for linearized model.

        Parameters
        ----------
        x : np.ndarray, shape=(n,)
            The current state.

        Returns
        -------
        ubar : np.ndarray, shape=(m,)
            The control input about which to linearize.
        """

    def _reset_gpmodel(self) -> None:
        """Resets the gurobi model."""
        self._gpmodel = gp.Model("mpc_problem", env=self._env)
        x_var = self._gpmodel.addMVar(
            (self._N + 1, self._n),
            name="x_var",
            lb=-np.inf,
        )
        u_var = self._gpmodel.addMVar(
            (self._N, self._m),
            name="u_var",
            lb=-np.inf,
        )
        self._gpmodel.update()
        self._gp_xvar = x_var
        self._gp_uvar = u_var
