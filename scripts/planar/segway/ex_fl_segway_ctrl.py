import sys

import numpy as np

sys.path.append("..")

from dyn_sim.ctrl.planar.segway.fdbk_lin_segway_ctrl import (  # noqa: E402
    FLPosRegSegwayController,
)
from dyn_sim.sim.simulator import SimulationEnvironment  # noqa: E402
from dyn_sim.sys.planar.segway import Segway  # noqa: E402

# SEGWAY #
seg = Segway()

# FDBK LIN CONTROLLER #
p_star = 0.0  # desired position
alpha1 = 1.0  # gains for feedback linearization
alpha2 = 1.0
flc = FLPosRegSegwayController(seg, p_star, alpha1, alpha2)

# SIMULATION ENVIRONMENT #
simulator = SimulationEnvironment(seg, flc)

if __name__ == "__main__":
    x0 = np.array([1, 0, 0, 0])
    sim_length = 15.0  # a long horizon here produces a pretty funny animation
    n_frames = int(50 * sim_length + 1)
    horizon = np.linspace(0, sim_length, n_frames)
    t_sol, x_sol = simulator.simulate(x0, horizon)

    # [TODO] fix 2D animations
    # animate
    # fps = 20.0
    # lims = ((-2, 2), (-2, 2), (-1, 1))
    # simulator.animate(t_sol, x_sol, lims, fps=fps)
