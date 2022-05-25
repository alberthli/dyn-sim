import sys
from typing import Union

import gurobipy as gp
import numpy as np

sys.path.append("..")

from dyn_sim.ctrl.mpc import SLMPC  # noqa: E402
from dyn_sim.ctrl.spatial.quad.slmpc_quad_ctrl import SLMPCQuadController  # noqa: E402
from dyn_sim.sim.simulator import SimulationEnvironment  # noqa: E402
from dyn_sim.sys.spatial.quad import Quadrotor  # noqa: E402

# CONSTANTS #
g = 9.80665  # gravitational acceleration

# ######### #
# QUADROTOR #
# ######### #
m = 1.0  # mass
I = np.array([1.0, 1.0, 1.0])  # principal moments of inertia
kf = 1.0  # thrust factor
km = 1.0  # drag factor
l = 0.1  # rotor arm length

# Optional: total rot moment about prop axes (gyro).
Jtp = 10.0  # [note] essentially induces a nonlinear disturbance

quad = Quadrotor(m, I, kf, km, l, Jtp)

# ########## #
# CONTROLLER #
# ########## #
wc = 5  # control frequency (Hz)
mpc_N = 4  # planning horizon
mpc_P = np.diag([12, 12, 12, 1, 1, 1, 2, 2, 2, 1, 1, 1])  # LQ cost weights
mpc_Q = mpc_P
mpc_R = np.diag([0.01, 0.01, 0.01, 0.01])
v_safe = 0.75  # safe velocity (m/s)
ang_safe = 0.5  # safe angular tilt (rad)
print_t = True  # [DEBUG] prints computation times


def x_ref(t: float, x: np.ndarray, slmpc: SLMPC) -> Union[np.ndarray, gp.MVar]:
    """Coarse state reference trajectory: circle + upright angle.

    Parameters
    ----------
    t : float
        Reference time.
    x : np.ndarray, shape=(12,)
        Quadrotor state.
    slmpc : FullMemory
        Memory bank object. Unused for this reference generator.

    Returns
    -------
    _ref : np.ndarray, shape=(12,)
        State reference at time t.
    """
    _ref = np.zeros(12)
    radius = 0.5
    _ref[0:3] = np.array([np.cos(radius * t), np.sin(radius * t), 0.0])
    _ref[6:9] = quad.Rwb(np.zeros(3)).T @ np.array(
        [-radius * np.sin(radius * t), radius * np.cos(radius * t), 0.0]
    )
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
    assert isinstance(slmpc._sys, Quadrotor)
    h = slmpc._h
    mem = slmpc._mem
    t0 = slmpc._t_last

    # first interval
    if t0 is None or t < t0 + h:
        if not mem.initialized:
            return np.array([slmpc._sys._mass * g, 0, 0, 0])
        else:
            return slmpc._u_mem[-1, :]

    # not first interval
    else:
        # t in [t_i, t_{i+1}]
        i = 0
        while t >= t0 + (i + 1) * h:
            i += 1
        return slmpc._gp_uvar[i - 1, :]


slmpc = SLMPCQuadController(
    quad,
    wc,
    mpc_N,
    mpc_P,
    mpc_Q,
    mpc_R,
    x_ref,
    u_ref,
    v_safe=v_safe,
    ang_safe=ang_safe,
    print_t=print_t,
)

# ###################### #
# SIMULATION ENVIRONMENT #
# ###################### #
simulator = SimulationEnvironment(quad, slmpc)

if __name__ == "__main__":
    # running simulation
    x0 = np.zeros(12)  # initial state
    sim_length = 30.0  # simulation time
    n_frames = int(10 * sim_length + 1)  # number of frames
    tsim = np.linspace(0, sim_length, n_frames)  # query times
    t_sol, x_sol = simulator.simulate(x0, tsim)

    # animating the solution
    fps = 20.0  # animation fps
    xyz_lims = ((-2, 2), (-2, 2), (-2, 2))
    simulator.animate(t_sol, x_sol, xyz_lims, fps=fps)
