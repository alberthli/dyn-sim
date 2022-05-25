import sys
from typing import Union

import gurobipy as gp
import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from dyn_sim.ctrl.mpc import SLMPC  # noqa: E402
from dyn_sim.ctrl.planar.segway.slmpc_segway_ctrl import (  # noqa: E402
    SLMPCSegwayController,
)
from dyn_sim.sim.simulator import SimulationEnvironment  # noqa: E402
from dyn_sim.sys.planar.segway import Segway  # noqa: E402

# #### #
# NOTE #
# #### #
# MPC alone is pretty crappy for the segway, but it does compute a solution. Since the system is so unstable, we would ideally want to compute better nominal (x, u) trajectories using either some generation method or iLQR.

# CONSTANTS #
g = 9.80665  # gravitational acceleration

# ###### #
# SEGWAY #
# ###### #
segway = Segway()

# ########## #
# CONTROLLER #
# ########## #
wc = 10  # control frequency (Hz)
mpc_N = 10  # planning horizon
mpc_P = np.diag([10, 1, 1, 1])  # LQ cost weights
mpc_Q = mpc_P
mpc_R = np.diag([0.01])
mpc_h = None
v_safe = None  # safe velocity (m/s)
ang_safe = None  # safe angular tilt (rad)
u_bnd = None
print_t = True  # [DEBUG] prints computation times


def x_ref(t: float, x: np.ndarray, slmpc: SLMPC) -> Union[np.ndarray, gp.MVar]:
    """Coarse state reference trajectory: sinusoidal position.

    Parameters
    ----------
    t : float
        Reference time.
    x : np.ndarray, shape=(4,)
        Segway state.
    slmpc : FullMemory
        Memory bank object. Unused for this reference generator.

    Returns
    -------
    _ref : np.ndarray, shape=(4,)
        State reference at time t.
    """
    _ref = np.zeros(4)
    # amp = 1
    # coeff = 0.1
    # _ref[0] = amp * np.sin(coeff * t)
    # _ref[2] = coeff * amp * np.cos(coeff * t)
    _ref[0] = t / 30.0
    _ref[2] = 1 / 30.0
    return _ref


def u_ref(t: float, x: np.ndarray, slmpc: SLMPC) -> Union[np.ndarray, gp.MVar]:
    """Input reference: previous input for smoothness.

    Parameters
    ----------
    See x_ref().

    Returns
    -------
    See x_ref().
    """
    assert isinstance(slmpc._sys, Segway)
    h = slmpc._h
    mem = slmpc._mem
    t0 = slmpc._t_last

    # first interval
    if t0 is None or t < t0 + h:
        if not mem.initialized:
            return np.array([0.0])
        else:
            return slmpc._u_mem[-1, :]

    # not first interval
    else:
        # t in [t_i, t_{i+1}]
        i = 0
        while t >= t0 + (i + 1) * h:
            i += 1
        return slmpc._gp_uvar[i - 1, :]


slmpc = SLMPCSegwayController(
    segway,
    wc,
    mpc_N,
    mpc_P,
    mpc_Q,
    mpc_R,
    x_ref,
    u_ref,
    mpc_h=mpc_h,
    v_safe=v_safe,
    ang_safe=ang_safe,
    u_bnd=u_bnd,
    print_t=print_t,
)

# ###################### #
# SIMULATION ENVIRONMENT #
# ###################### #
simulator = SimulationEnvironment(segway, slmpc)

if __name__ == "__main__":
    # running simulation
    x0 = np.zeros(4)  # initial state
    sim_length = 30.0  # simulation time
    n_frames = int(10 * sim_length + 1)  # number of frames
    tsim = np.linspace(0, sim_length, n_frames)  # query times
    t_sol, x_sol = simulator.simulate(x0, tsim)

    # debug
    plt.plot(t_sol, x_sol[0, :])
    plt.show()

    # animating the solution
    # fps = 20.0  # animation fps
    # xyz_lims = ((-2, 2), (-2, 2), (-2, 2))
    # simulator.animate(t_sol, x_sol, xyz_lims, fps=fps)
