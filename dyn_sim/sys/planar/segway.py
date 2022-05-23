import numpy as np
from matplotlib.axes import Axes

from dyn_sim.sys.dyn_sys import CtrlAffineSystem
from dyn_sim.util.sim_utils import draw_circle

# constants
constants = {
    "g": 9.80665,  # gravitational acceleration
}


class Segway(CtrlAffineSystem):
    """Segway object.

    State vector is in R^4.
        state vector s is in order [p, phi, dp, dphi]
        p - horizontal position, positive right
        phi - rotational position, positive cw
        dp - time derivative of horizontal position
        dphi - time derivative of rotational position
    Input vector is in R.
        u - motor voltage
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
    ) -> None:
        """Initialize a segway.

        Parameters
        ----------
        m0 : float
            Lumped mass of the segway
        L : float
            Length between the center of rotation and center of gravity
        J0 : float
            Lumped inertia of the segway
        Km : float
            Motor Torque constant
        R : float
            Wheel radius
        bt : float
            Motor damping constant
        l : float
            Length between center of rotation and tip of segway arm
        """
        super(Segway, self).__init__(4, 1, False)
        self._m0 = m0
        self._L = L
        self._J0 = J0
        self._Km = Km
        self._R = R
        self._bt = bt
        self._l = l

    def fdyn(self, t: float, s: np.ndarray) -> np.ndarray:
        """Segway autonomous dynamics.

        Parameters
        ----------
        t : float,
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(4,)
            State of segway.

        Returns
        -------
        _fdyn : np.ndarray, shape=(4,)
            Time derivatives of states from autonomous dynamics.
        """
        assert s.shape == (4,)

        # states
        # p = s[0] # unused
        phi = s[1]
        dp = s[2]
        dphi = s[3]

        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        # inertia matrix
        D = np.array(
            [
                [self._m0, self._m * self._L * cosphi],
                [self._m * self._L * cosphi, self._J0],
            ]
        )
        Dinv = np.linalg.inv(D)

        # coriolis & gravity matrix
        H = np.array(
            [
                -self._m * self._L * sinphi * dphi**2
                + self._bt * (dp - self._R * dphi) / self._R,
                -self._m * self._g * self._L * sinphi
                + self._bt * (dp - self._R * dphi),
            ]
        )

        DinvH = Dinv @ H
        _fdyn = np.array([dp, dphi, -DinvH[0], -DinvH[1]])
        return _fdyn

    def gdyn(self, t: float, s: np.ndarray) -> np.ndarray:
        """Segway control dynamics.

        Parameters
        ----------
        t : float,
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(4,)
            State of segway.

        Returns
        -------
        _gdyn : np.ndarray, shape=(4,)
            Matrix representing affine control dynamics.
        """
        assert s.shape == (4,)

        # states
        # p = s[0] # unused
        phi = s[1]
        # dp = s[2] # unused
        # dphi = s[3] # unused

        # sinphi = np.sin(phi) # unused
        cosphi = np.cos(phi)

        # inertia matrix
        D = np.array(
            [
                [self._m0, self._m * self._L * cosphi],
                [self._m * self._L * cosphi, self._J0],
            ]
        )
        Dinv = np.linalg.inv(D)

        # input matrix
        B = np.array([self._Km / self._R, -self._Km])

        DinvB = Dinv @ B
        _gdyn = np.array([0, 0, DinvB[0], DinvB[1]])
        return _gdyn

    def A(self, s: np.ndarray, u: float) -> np.ndarray:
        """Linearized autonomous dynamics about (s, u).

        Parameters
        ----------
        s : np.ndarray, shape=(4,)
            State.
        u : float
            Input of the segway. Unused, included for API compliance.

        Returns
        -------
        _A : np.ndarray, shape=(4, 4)
            Linearized autonomous dynamics about s.
        """
        assert s.shape == (4,)
        assert np.isscalar(u)

        raise NotImplementedError  # TODO implement linearized dynamics

    def B(self, s: np.ndarray, u: float) -> np.ndarray:
        """Linearized control dynamics about (s, u).

        Parameters
        ----------
        s : np.ndarray, shape=(4,)
            State.
        u : float
            Input of the quadrotor. Unused, included for API compliance.

        Returns
        -------
        _B : np.ndarray, shape=(4,)
            Linearized control dynamics about s.
        """
        assert s.shape == (4,)
        assert np.isscalar(u)

        raise NotImplementedError  # TODO implement linearized dynamics

    def draw(self, ax: Axes, s: np.ndarray) -> None:
        """Draws the segway on specified Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the quadrotor.
        s : np.ndarray, shape=(4,)
            Current state of the segway.
        """
        assert s.shape == (4,)
        p = s[0]
        phi = s[1]
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        pos0 = np.array([p, 0])
        pos1 = np.array([[cosphi, sinphi], [-sinphi, cosphi]]) @ np.array([p, self._l])

        # drawing segway wheel
        draw_circle(ax, pos0, self._R, color="green")

        # drawing segway arm
        ax.plot(pos0, pos1, "k-")
