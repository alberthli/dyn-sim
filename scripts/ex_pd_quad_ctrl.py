import sys

import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp

sys.path.append("..")

from dyn_sim.ctrl.quad.pd_quad_ctrl import PDQuadController  # noqa: E402
from dyn_sim.sys.spatial.quad import Quadrotor  # noqa: E402

# QUADROTOR #
m = 1.0  # mass
I = np.array([1.0, 1.0, 1.0])  # principal moments of inertia
kf = 1.0  # thrust factor
km = 1.0  # drag factor
l = 0.1  # rotor arm length
Jtp = 0.1  # Optional: total rot moment about prop axes (gyro)

quad = Quadrotor(m, I, kf, km, l, Jtp)


# CONTROLLERS #

# PD controller
kp_xyz = 0.01  # gains for Cartesian position control
kd_xyz = 0.04
kp_a = 10  # gains for attitude control
kd_a = 5
ref = lambda t: np.array([np.cos(t), np.sin(t), 0, 0])  # reference

pdc = PDQuadController(quad, kp_xyz, kd_xyz, kp_a, kd_a, ref)


if __name__ == "__main__":
    # running simulation
    s0 = np.zeros(12)  # initial state
    sim_length = 30.0  # simulation time
    frames = int(10 * sim_length + 1)  # number of frames
    fps = 20.0  # animation fps
    tsim = np.linspace(0, sim_length, frames)  # query times

    ctrl = lambda t, s: pdc.ctrl(t, s)
    dyn = lambda t, s: quad.dyn(t, s, ctrl(t, s))

    sol = solve_ivp(
        dyn, (tsim[0], tsim[-1]), s0, t_eval=tsim, max_step=0.01
    )  # cap framerate of reality
    sim_data = sol.y

    # plotting positional results
    o = sim_data[0:3, :]
    alpha = sim_data[3:6, :]

    fig, axs = plt.subplots(2, 1)
    fig.tight_layout(pad=3.0)

    # positions
    axs[0].plot(tsim, o[0, :].T, color="orange")
    axs[0].plot(tsim, o[1, :].T, color="green")
    axs[0].plot(tsim, o[2, :].T, color="blue")
    axs[0].legend(["x", "y", "z"], loc="lower left", ncol=3)
    axs[0].set_title("Quadrotor Position and Reference Trajectory")
    axs[0].set_xlabel("time")
    axs[0].set_ylabel("position")
    axs[0].set_ylim([-2, 2])
    axs[0].set_xlim([tsim[0], tsim[-1]])

    # angles
    axs[1].plot(tsim, alpha[0, :].T, color="orange")
    axs[1].plot(tsim, alpha[1, :].T, color="green")
    axs[1].legend(["roll", "pitch"], loc="lower left", ncol=2)
    axs[1].set_title("Quadrotor Roll/Pitch and Safety Limits")
    axs[1].set_xlabel("time")
    axs[1].set_ylabel("angle")
    axs[1].set_xlim([tsim[0], tsim[-1]])

    plt.show()  # TODO: fix this also showing the simulator when unwanted
