import numpy as np

from dyn_sim.ctrl.ctrl import Controller
from dyn_sim.sys.planar.segway import Segway

# constants
g = 9.80665  # gravitational acceleration


class FLPosRegSegwayController(Controller):
    """Feedback linearizing controller for position regulation of a segway."""

    def __init__(
        self,
        seg: Segway,
        p_star: float,
        alpha1: float,
        alpha2: float,
    ) -> None:
        """Initialize a segway feedback linearizing controller.

        Note by the Routh-Hurwitz criterion, for 2D output dynamics,
        a necessary and sufficient condition for (exponential) stability is both
        alpha1 and alpha2 are positive.

        Parameters
        ----------
        seg : Segway
            Segway object to be controlled.
        p_star : float
            Position that segway will be regulated to.
        alpha1 : float
            First gain for output dynamics.
        alpha2 : float
            Second gain for output dynamics.
        """
        assert alpha1 > 0.0
        assert alpha2 > 0.0

        super(FLPosRegSegwayController, self).__init__(seg)

        self._seg = seg
        self._p_star = p_star
        self._alpha1 = alpha1
        self._alpha2 = alpha2

    def ctrl(self, t: float, x: np.ndarray) -> np.ndarray:
        """Feedback linearizing control law.

        Parameters
        ----------
        t : float
            Time.
        x : np.ndarray, shape=(4,)
            State.

        Returns
        -------
        u : np.ndarray, shape=(1,)
            Control input.
        """
        assert x.shape == (self._n,)

        # gains
        alpha1 = self._alpha1
        alpha2 = self._alpha2

        # states
        p, phi, dp, dphi = x
        sinphi = np.sin(phi)
        cosphi = np.cos(phi)

        # segway parameter unpacking
        m0, L, J0, Km, R, bt, _, mass = (
            self._seg._m0,
            self._seg._L,
            self._seg._J0,
            self._seg._Km,
            self._seg._R,
            self._seg._bt,
            self._seg._l,
            self._seg._mass,
        )

        # relative degree condition
        if J0 + mass * R * L * cosphi <= 0:
            raise ArithmeticError("Relative degree condition not satisfied.")

        # Lie derivatives
        h = p - self._p_star
        Lfh = dp
        Lf2h = (
            mass * J0 * L * sinphi * dphi**2
            - J0 * bt / R * (dp - R * dphi)
            - mass**2 * g * L**2 * sinphi * cosphi
            - mass * L * bt * cosphi * (dp - R * dphi)
        ) / (m0 * J0 - mass**2 * L**2 * cosphi**2)
        LgLfh = (Km / R * (J0 + mass * R * L * cosphi)) / (
            m0 * J0 - mass**2 * L**2 * cosphi**2
        )

        k = 1 / LgLfh * (-Lf2h - alpha2 * Lfh - alpha1 * h)
        return np.array([k])
