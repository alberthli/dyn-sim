import numpy as np
from matplotlib.axes import Axes

from dyn_sim.sys.sys_core import CtrlAffineSystem
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
    ) -> None:
        """Initialize a segway.

        Parameters
        ----------
        m0 : float
            Lumped mass of the segway.
        L : float
            Length between the center of rotation and center of gravity.
        J0 : float
            Lumped inertia of the segway.
        Km : float
            Motor Torque constant.
        R : float
            Wheel radius.
        bt : float
            Motor damping constant.
        l : float
            Length between center of rotation and tip of segway arm.
        mass : float
            Mass of upper segway arm (name spelled out to prevent collision).

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

    def fdyn(self, t: float, x: np.ndarray) -> np.ndarray:
        """Segway autonomous dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        x : np.ndarray, shape=(4,)
            State of segway.

        Returns
        -------
        _fdyn : np.ndarray, shape=(4,)
            Time derivatives of states from autonomous dynamics.
        """
        assert x.shape == (4,)

        # unpacking variables
        m0 = self._m0
        m = self._m
        L = self._L
        J0 = self._J0
        bt = self._bt
        R = self._R

        # states
        _, phi, dp, dphi = x
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        # inverse inertia matrix
        denom = J0 * m0 - L**2 * m**2 * cosphi**2
        Dinv = np.array([[J0, -m * L * cosphi], [-m * L * cosphi, m0]]) / denom

        # coriolis & gravity matrix
        H = np.array(
            [
                -m * L * sinphi * dphi**2 + bt * (dp - R * dphi) / R,
                -m * g * L * sinphi + bt * (dp - R * dphi),
            ]
        )

        DinvH = Dinv @ H
        _fdyn = np.array([dp, dphi, -DinvH[0], -DinvH[1]])
        return _fdyn

    def gdyn(self, t: float, x: np.ndarray) -> np.ndarray:
        """Segway control dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        x : np.ndarray, shape=(4,)
            State of segway.

        Returns
        -------
        _gdyn : np.ndarray, shape=(4,)
            Matrix representing affine control dynamics.
        """
        assert x.shape == (4,)

        # unpacking variables
        m0 = self._m0
        m = self._m
        L = self._L
        J0 = self._J0
        Km = self._Km
        R = self._R

        # states
        _, phi, _, _ = x

        cosphi = np.cos(phi)

        # inverse inertia matrix
        denom = J0 * m0 - L**2 * m**2 * cosphi**2
        Dinv = np.array([[J0, -m * L * cosphi], [-m * L * cosphi, m0]]) / denom

        # input matrix
        B = np.array([Km / R, -Km])

        DinvB = Dinv @ B
        _gdyn = np.array([0, 0, DinvB[0], DinvB[1]])
        return _gdyn

    def A(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearized autonomous dynamics about (x, u).

        Parameters
        ----------
        x : np.ndarray, shape=(4,)
            State.
        u : np.ndarray, shape=(1,)
            Input of the segway.

        Returns
        -------
        _A : np.ndarray, shape=(4, 4)
            Linearized autonomous dynamics about (x, u).
        """
        assert x.shape == (4,)
        assert u.shape == (1,)

        # unpacking variables
        m0 = self._m0
        m = self._m
        L = self._L
        J0 = self._J0
        Km = self._Km
        R = self._R
        bt = self._bt

        p, phi, dp, dphi = x
        sphi = np.sin(phi)
        cphi = np.cos(phi)

        # constructing A
        _A = np.zeros((self._n, self._n))
        TERM1 = J0 * m0 - L**2 * m**2 * cphi**2
        _A[1, 2] = -(
            L
            * m
            * (
                -J0 * cphi * dphi**2
                - R * bt * sphi * dphi
                + Km * u * sphi
                + bt * dp * sphi
                + L * g * m * (2 * cphi**2 - 1)
            )
        ) / TERM1 - (
            2
            * L**2
            * m**2
            * cphi
            * sphi
            * (
                J0 * Km * u
                - J0 * bt * dp
                + J0 * R * bt * dphi
                + L * R * bt * dp * m * cphi
                - L**2 * R * g * m**2 * cphi * sphi
                + J0 * L * R * dphi**2 * m * sphi
                - L * R**2 * bt * dphi * m * cphi
                + Km * L * R * m * u * cphi
            )
        ) / (
            R * TERM1**2
        )
        _A[1, 3] = (
            L
            * m
            * (
                -L * R * m * (2 * cphi**2 - 1) * dphi**2
                + R * bt * sphi * dphi
                + Km * u * sphi
                - bt * dp * sphi
                + R * g * m0 * cphi
            )
        ) / (R * TERM1) + (
            2
            * L**2
            * m**2
            * cphi
            * sphi
            * (
                Km * R * m0 * u
                - R**2 * bt * dphi * m0
                + R * bt * dp * m0
                + Km * L * m * u * cphi
                - L * bt * dp * m * cphi
                + L * R * bt * dphi * m * cphi
                - L * R * g * m * m0 * sphi
                + L**2 * R * dphi**2 * m**2 * cphi * sphi
            )
        ) / (
            R * TERM1**2
        )
        _A[2, 0] = 1
        _A[2, 2] = -(bt * (J0 - L * R * m * cphi)) / (R * TERM1)
        _A[2, 3] = -(bt * (R * m0 - L * m * cphi)) / (R * TERM1)
        _A[3, 1] = 1
        _A[3, 2] = (
            J0 * bt - L * R * bt * m * cphi + 2 * J0 * L * dphi * m * sphi
        ) / TERM1
        _A[3, 3] = (
            -(
                dphi * np.sin(2 * phi) * L**2 * m**2
                + bt * cphi * L * m
                - R * bt * m0
            )
            / TERM1
        )
        return _A

    def B(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearized control dynamics about (x, u).

        Parameters
        ----------
        x : np.ndarray, shape=(4,)
            State.
        u : np.ndarray, shape=(1,)
            Input of the quadrotor.

        Returns
        -------
        _B : np.ndarray, shape=(4,)
            Linearized control dynamics about (x, u).
        """
        assert x.shape == (4,)
        assert u.shape == (1,)

        # unpacking variables
        m0 = self._m0
        m = self._m
        L = self._L
        J0 = self._J0
        Km = self._Km
        R = self._R

        p, phi, dp, dphi = x
        cphi = np.cos(phi)

        # constructing _B
        _B = np.zeros((self._n, self._m))
        denom = R * (J0 * m0 - L**2 * m**2 * cphi**2)
        _B[2] = Km * (J0 + L * R * m * cphi) / denom
        _B[3] = -Km * (R * m0 + L * m * cphi) / denom
        return _B

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
        p = x[0]
        phi = x[1]
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        pos0 = np.array([p, 0])
        pos1 = np.array([p + self._l * sinphi, self._l * cosphi])

        # drawing segway wheel
        draw_circle(ax, pos0, self._R, color="green")

        # drawing segway arm
        ax.plot([pos0[0], pos1[0]], [pos0[1], pos1[1]], "k-")
