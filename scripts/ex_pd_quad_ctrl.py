import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.append("..")

from dyn_sim.ctrl.quad.pd_quad_ctrl import PDQuadController  # noqa: E402
from dyn_sim.sim.simulator import SimulationEnvironment  # noqa: E402
from dyn_sim.sys.spatial.quad import Quadrotor  # noqa: E402

# QUADROTOR #
m = 1.0  # mass
I = np.array([1.0, 1.0, 1.0])  # principal moments of inertia
kf = 1.0  # thrust factor
km = 1.0  # drag factor
l = 0.1  # rotor arm length
Jtp = 0.1  # Optional: total rot moment about prop axes (gyro)

quad = Quadrotor(m, I, kf, km, l, Jtp)

# PD CONTROLLER #
kp_xyz = 0.01  # gains for Cartesian position control
kd_xyz = 0.04
kp_a = 10  # gains for attitude control
kd_a = 5
ref = lambda t: np.array(
    [np.cos(t), np.sin(t), 0, 0]
)  # (x, y, z, yaw), pitch/roll assumed 0

pdc = PDQuadController(quad, kp_xyz, kd_xyz, kp_a, kd_a, ref)

# SIMULATION ENVIRONMENT #
simulator = SimulationEnvironment(quad, pdc)

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

    # STATIC PLOTS #

    # plotting positional results
    o = x_sol[0:3, :]
    alpha = x_sol[3:6, :]

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)

    # positions
    axs[0].plot(t_sol, o[0, :].T, color="orange")
    axs[0].plot(t_sol, o[1, :].T, color="green")
    axs[0].plot(t_sol, o[2, :].T, color="blue")
    axs[0].legend(["x", "y", "z"], loc="lower left", ncol=3)
    axs[0].set_title("Quadrotor Position")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position")
    axs[0].set_ylim([-2, 2])
    axs[0].set_xlim([t_sol[0], t_sol[-1]])

    # angles
    axs[1].plot(t_sol, alpha[0, :].T, color="orange")
    axs[1].plot(t_sol, alpha[1, :].T, color="green")
    axs[1].legend(["roll", "pitch"], loc="lower left", ncol=2)
    axs[1].set_title("Quadrotor Roll and Pitch")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("angle")
    axs[1].set_xlim([t_sol[0], t_sol[-1]])
