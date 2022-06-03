import jax.numpy as jnp
import numpy as np
from matplotlib.axes import Axes

from dyn_sim.sys.sys_core import CtrlAffineSystem
from dyn_sim.util.jax_utils import jax_func
from dyn_sim.util.sim_utils import draw_circle

# constants
g = 9.80665  # gravitational acceleration


class Segway(CtrlAffineSystem):
    """Segway object.

    State vector is in R^4.
        state vector x is in order [p, phi, dp, dphi]
        p : horizontal position, positive right
        phi : rotational position, positive cw, 0 is vertical
        dp : time derivative of horizontal position
        dphi : time derivative of rotational position
    Input vector is in R.
        u : motor voltage

    Segway dynamics retrieved from:
        T. Molnar, A. Kiss, A. Ames, and G. Orosz, “Safety-critical
        control with input delay in dynamic environment,” Dec. 2021,
        unpublished, online at https://arxiv.org/pdf/2112.08445.pdf.
    """

    def __init__(
        self,
        m0: float = 52.71,
        L: float = 0.169,
        J0: float = 5.108,
        Km: float = 2 * 1.262,
        R: float = 0.195,
        bt: float = 2 * 1.225,
        l: float = 0.75,
        mass: float = 44.798,
        phi0: float = 0.138,
    ) -> None:
        """Initialize a segway.

        Parameters
        ----------
        m0 : float, default=52.71
            Lumped mass (kg) of the segway.
        L : float, default=0.169
            Length (m) between the center of rotation and center of gravity.
        J0 : float, default=5.108
            Lumped inertia (kg*m^2) of the segway.
        Km : float, default=(2 * 1.262)
            Motor Torque constant (Nm / V).
        R : float, default=0.195
            Wheel radius (m).
        bt : float, default=(2 * 1.225)
            Motor damping constant (Ns).
        l : float, default=0.75
            Length (m) between center of rotation and tip of segway arm.
        mass : float, default=44.798
            Mass (kg) of upper segway arm (name spelled out to prevent collision).
        phi0 : float, default=0.138
            Equilibrium tilt (rad) of the segway.

        Default parameters retrieved from Molnar (2021), see Segway docstring for link.
        """
        super(Segway, self).__init__(4, 1, False)
        self._m0 = m0
        self._L = L
        self._J0 = J0
        self._Km = Km
        self._R = R
        self._bt = bt
        self._l = l
        self._mass = mass
        self._phi0 = phi0

    @jax_func
    def fdyn(self, t: float, x: np.ndarray) -> jnp.ndarray:
        """Segway autonomous dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        x : np.ndarray, shape=(4,)
            State of segway.

        Returns
        -------
        _fdyn : jnp.ndarray, shape=(4,)
            Time derivatives of states from autonomous dynamics.
        """
        assert x.shape == (4,)

        # unpacking variables
        m0 = self._m0
        mass = self._mass
        L = self._L
        J0 = self._J0
        bt = self._bt
        R = self._R

        # states
        _, phi, dp, dphi = x
        sinphi = jnp.sin(phi)
        cosphi = jnp.cos(phi)

        # inverse inertia matrix
        denom = J0 * m0 - L**2 * mass**2 * cosphi**2
        Dinv = jnp.array([[J0, -mass * L * cosphi], [-mass * L * cosphi, m0]]) / denom

        # coriolis & gravity matrix
        H = jnp.array(
            [
                -mass * L * sinphi * dphi**2 + bt * (dp - R * dphi) / R,
                -mass * g * L * sinphi + bt * (dp - R * dphi),
            ]
        )

        DinvH = Dinv @ H
        _fdyn = jnp.array([dp, dphi, -DinvH[0], -DinvH[1]])
        return _fdyn

    @jax_func
    def gdyn(self, t: float, x: np.ndarray) -> jnp.ndarray:
        """Segway control dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        x : np.ndarray, shape=(4,)
            State of segway.

        Returns
        -------
        _gdyn : jnp.ndarray, shape=(4, 1)
            Matrix representing affine control dynamics.
        """
        assert x.shape == (4,)

        # unpacking variables
        m0 = self._m0
        mass = self._mass
        L = self._L
        J0 = self._J0
        Km = self._Km
        R = self._R

        # states
        _, phi, _, _ = x

        cosphi = jnp.cos(phi)

        # inverse inertia matrix
        denom = J0 * m0 - L**2 * mass**2 * cosphi**2
        Dinv = jnp.array([[J0, -mass * L * cosphi], [-mass * L * cosphi, m0]]) / denom

        # input matrix
        B = jnp.array([Km / R, -Km])

        DinvB = Dinv @ B
        _gdyn = jnp.array([[0, 0, DinvB[0], DinvB[1]]]).T
        return _gdyn

    def draw(self, ax: Axes, x: np.ndarray) -> None:
        """Draws the segway on specified Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the quadrotor.
        x : np.ndarray, shape=(4,)
            Current state of the segway.
        """
        assert x.shape == (4,)
        assert ax.name == "rectilinear" or ax.name == "polar"
        p = x[0]
        phi = x[1]
        sinphi = np.sin(phi + self._phi0)
        cosphi = np.cos(phi + self._phi0)

        pos0 = np.array([p, self._R])
        pos1 = np.array([p + self._l * sinphi, self._R + self._l * cosphi])

        # drawing segway wheel
        draw_circle(ax, pos0, self._R, color="green")

        # drawing segway arm
        ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], "k-")
