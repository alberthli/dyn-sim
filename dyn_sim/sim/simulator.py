from typing import Callable, Optional, Tuple, Union

import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.pyplot import Figure
from mpl_toolkits.mplot3d.axes3d import Axes3D
from scipy.integrate import solve_ivp

from dyn_sim.ctrl.ctrl_core import Controller, MemoryController
from dyn_sim.sys.sys_core import System


class SimulationEnvironment:
    """Simulation environment for the system."""

    def __init__(self, sys: System, ctrler: Controller) -> None:
        """Initialize the simulator.

        Parameters
        ----------
        sys : System
            A dynamical system.
        ctrler: Controller
            A controller object.
        """
        self._sys = sys
        self._ctrler = ctrler

        self._fig: Optional[Figure] = None
        self._ax: Optional[Union[Axes, Axes3D]] = None

    def simulate(
        self,
        x0: np.ndarray,
        tsim: np.ndarray,
        dfunc: Optional[Callable[[float, np.ndarray], np.ndarray]] = None,
        max_step: float = 0.01,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Simulate the system.

        Parameters
        ----------
        x0 : np.ndarray, shape=(n,)
            Initial state.
        tsim : np.ndarray, shape=(T,)
            Simulation query points.
        dfunc : Optional[Callable[np.ndarray, np.ndarray]]
            Disturbance function. Takes in state and time and returns a
            simulated disturbance.
        max_step : float, default=0.01
            Maximum time step in ODE solver.

        Returns
        -------
        t_sol : np.ndarray, shape=(T,)
            Time values associated with state solution.
        x_sol : np.ndarray, shape=(n, T)
            Solution trajectories at the query times.
        """
        assert x0.shape == (self._sys._n,)
        assert tsim.ndim == 1

        sys = self._sys

        # controller
        ctrl = lambda t, x: self._ctrler.ctrl(t, x)

        # disturbance function
        if dfunc is not None:
            raise NotImplementedError  # [TODO] fix the disturbance API
            dyn = lambda t, x: sys.dyn(t, x, ctrl(t, x)) + dfunc(t, x)
        else:
            dyn = lambda t, x: sys.dyn(t, x, ctrl(t, x))

        # simulating dynamics
        sol = solve_ivp(dyn, (tsim[0], tsim[-1]), x0, t_eval=tsim, max_step=max_step)
        t_sol = sol.t
        x_sol = sol.y

        if isinstance(self._ctrler, MemoryController):
            self._ctrler.reset()

        return t_sol, x_sol

    def animate(
        self,
        t_sol: np.ndarray,
        x_sol: np.ndarray,
        lims: Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]],
        fps: float = 10.0,
        anim_name: Optional[str] = None,
    ) -> None:
        """Animate a simulated result.

        Parameters
        ----------
        t_sol : np.ndarray, shape=(T,)
            Time values associated with state solution.
        x_sol : np.ndarray, shape=(n, T)
            Solution trajectories at the query times.
        lims : Tuple[Tuple[float, float], Tuple[float, float], Tuple[float, float]]
            A tuple of the xlim, ylim, zlim for the plot.
        fps : float, default=10.0
            Animation frames per second.
        anim_name : str, default=None
            Name for animation file. If not None, saves an mp4. Else, doesn't save.
        """
        (
            xlim,
            ylim,
            zlim,
        ) = lims  # for 2D animations, still need to pass in a zlim (though it is now unused)

        self._fig = plt.figure()
        if self._sys._is3d:
            self._ax = Axes3D(self._fig)
            self._ax.set_proj_type("ortho")
            self._ax.grid(False)
            self._ax.set_xticks([])
            self._ax.set_yticks([])
            self._ax.set_zticks([])
            self._ax.set_xlim3d(xlim)
            self._ax.set_ylim3d(ylim)
            self._ax.set_zlim3d(zlim)
        else:
            raise NotImplementedError  # [TODO] fix 2D animations
            # self._ax = plt.Axes(
            #     self._fig, [0, 0, self._fig.get_figwidth(), self._fig.get_figheight()]
            # )
            # self._ax = plt.Axes(self._fig, [0, 0, 1, 1])
            # self._ax.grid(False)
            # self._ax.set_xticks([])
            # self._ax.set_yticks([])
            # self._ax.set_xlim(xlim)
            # self._ax.set_ylim(ylim)

        def _clear_frame() -> None:
            """Clear the environment frame."""
            if self._ax is not None:
                for artist in self._ax.lines + self._ax.collections:
                    artist.remove()

        def _anim_sys(i):
            """Animate the system on the frame."""
            _clear_frame()
            self._sys.draw(self._ax, x_sol[:, i])

        anim = animation.FuncAnimation(
            self._fig, _anim_sys, interval=20.0, frames=len(t_sol)
        )

        if anim_name is not None:
            Writer = animation.writers["ffmpeg"]
            writer = Writer(fps=fps, bitrate=1800)
            anim.save("{}.mp4".format(anim_name), writer=writer)

        plt.show()
        _clear_frame()

    # ###################################### #
    # KEEP FOR NOW WHILE REFACTOR IS ONGOING #
    # ###################################### #
    # def simulate(
    #     self,
    #     s0: np.ndarray,
    #     tsim: np.ndarray,
    #     dfunc: Callable[[float, np.ndarray], np.ndarray] = None,
    #     animate: bool = False,
    #     anim_name: str = None,
    #     fps: float = 10.0,
    # ) -> np.ndarray:
    #     """Simulate a quadrotor run.

    #     Parameters
    #     ----------
    #     s0: np.ndarray, shape=(12,)
    #         Initial state.
    #     tsim: np.ndarray, shape=(T,)
    #         Simulation query points.
    #     dfunc: Callable[np.ndarray, np.ndarray]
    #         Disturbance function. Takes in state and time and returns a
    #         simulated disturbance.
    #     animate: bool
    #         Flag for whether an animation of the run should play.
    #     anim_name: str
    #         Name for animation file.
    #     fps: float
    #         Animation frames per second.

    #     Returns
    #     -------
    #     s_sol: np.ndarray, shape=(12, T)
    #         Solution trajectories at the query times.
    #     """
    #     assert s0.shape == (12,)
    #     assert tsim.ndim == 1

    #     quad = self._quad

    #     # controller
    #     if type(self._ctrler) == MultirateQuadController:
    #         ctrl = lambda t, s: self._ctrler.ctrl(t, s, self._obs_list)
    #     else:
    #         ctrl = lambda t, s: self._ctrler.ctrl(t, s)

    #     # disturbance function
    #     if dfunc is not None:
    #         dyn = lambda t, s: quad._dyn(s, ctrl(t, s), dfunc(t, s))
    #     else:
    #         dyn = lambda t, s: quad._dyn(s, ctrl(t, s))

    #     # simulating dynamics
    #     sol = solve_ivp(
    #         dyn, (tsim[0], tsim[-1]), s0, t_eval=tsim, max_step=self._ctrler._sim_dt
    #     )  # cap framerate of reality
    #     s_sol = sol.y

    #     # Get ref traj for plotting
    #     ctrler = self._ctrler
    #     ref = ctrler._ref
    #     ref_traj = np.zeros((12, s_sol.shape[1]))
    #     for i in range(s_sol.shape[1]):
    #         ref_traj[:, i] = ref(tsim[i])

    #     self._ctrler.reset()

    #     # animation
    #     if animate:
    #         self._draw_obs()

    #         def _anim_quad(i):
    #             self._clear_frame()
    #             self._draw_quad(s_sol[:, i])
    #             self._draw_traj(s_sol, ref_traj, i)

    #         anim = animation.FuncAnimation(
    #             self._fig, _anim_quad, interval=20.0, frames=len(tsim)
    #         )

    #         if anim_name is not None:
    #             Writer = animation.writers["ffmpeg"]
    #             writer = Writer(fps=fps, bitrate=1800)
    #             anim.save("{}.mp4".format(anim_name), writer=writer)

    #         plt.show()
    #         self._clear_frame(clear_obs=True)

    #     return s_sol
