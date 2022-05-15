import numpy as np
from dyn_sim.dyn_sys import CtrlAffineSystem
from matplotlib.axes import Axes

from dyn_sim.util.sim_utils import draw_circle

# constants
g = 9.80665  # gravitational acceleration


class Quadrotor(CtrlAffineSystem):
    """Quadrotor object.

    Conventions are from:

    'Modelling, Identification and Control of a Quadrotor Helicopter'

    Conventions:
    [1] The default inertial frame is fixed and NWU (north, west, up).
    [2] Forward direction is x, left is y in body frame.
    [3] Forward and rear rotors are numbered 1 and 3 respectively. Left and
        right rotors are numbered 4 and 2 respectively. The even rotors
        rotate CCW and the odd rotors rotate CW. When aligned with the inertial
        frame, this means x is aligned with N, y is aligned with W, and
        z is aligned with U.

                          1 ^ x
                            |
                     y      |
                     4 <----o----> 2
                            |
                            |
                            v 3

    [4] Let the quadrotor state be in R^12. The origin and axes of the body
        frame should coincide with the barycenter of the quadrotor and the
        principal axes. Angle states follow ZYX Euler angle conventions.
        (x, y, z): Cartesian position in inertial frame
        (phi, theta, psi): angular position in inertial frame (RPY respectively)
        (u, v, w): linear velocity in body frame
        (p, q, r): angular velocities in body frame
    [5] Vector conventions
        s: State of the quadrotor in order of (o, alpha, dob, dalphab), where
           o is the position and alpha is the vector of angular positions in
           the INERTIAL frame. dob and dalphab are the linear and angular
           velocities in the BODY frame.
        u: Virtual input in order of (ft, taux, tauy, tauz), where ft is the
           total thrust in the +z direction and tauxyz are the angular torques.
           These are all with respect to the BODY frame.
        d: Disturbances in the BODY frame in order of (fdb, taudb), where fdb
           are forces and taudb are torques. Ordered (x,y,z) each. TODO: make
           it so these are specified in the inertial frame.
    """

    def __init__(
        self,
        mass: float,
        I: np.ndarray,
        kf: float,
        km: float,
        l: float,
        Jtp: float = None,
    ) -> None:
        """Initialize a quadrotor.

        Parameters
        ----------
        mass : float
            Mass of the quadrotor.
        I : np.ndarray, shape=(3,)
            Principal moments of inertia.
        kf : float
            Thrust factor. Internal parameter for rotor speed to body force and
            moment conversion.
        km : float
            Drag factor. Internal parameter for rotor speed to body force and
            moment conversion.
        l : float
            Distance from rotors to center of quadrotor.
        Jtp : Optional[float]
            Total rotational moment of inertia about the propellor axes. Used
            for computing the gyroscopic effect.
        """
        assert mass > 0.0
        assert I.shape == (3,)
        assert np.all(I > 0.0)
        assert kf > 0.0
        assert km > 0.0
        if Jtp is not None:
            assert Jtp > 0.0

        super(Quadrotor, self).__init__(12, 4)
        self._mass = mass
        self._I = I
        self._kf = kf
        self._km = km
        self._l = l
        self._Jtp = Jtp

    @property
    def V(self) -> np.ndarray:
        """Matrix converting squared rotor speeds to virtual forces/moments.

        Controller design occurs using virtual force/torque inputs, but true
        input is rotor speeds, so once the virtual inputs are designed, they
        are converted (in an invertible way) using _V.

        u = V @ wsq
        """
        kf = self._kf
        km = self._km
        l = self._l

        _V = np.array(
            [
                [kf, kf, kf, kf],
                [0.0, -kf * l, 0.0, kf * l],
                [-kf * l, 0.0, kf * l, 0.0],
                [-km, km, -km, km],
            ]
        )

        return _V

    @property
    def invV(self) -> np.ndarray:
        """Matrix converting virtual forces/moments to squared rotor speeds.

        wsq = invV @ u
        """
        kf = self._kf
        km = self._km
        l = self._l

        _invV = (
            np.array(
                [
                    [1.0 / kf, 0.0, -2.0 / (kf * l), -1.0 / km],
                    [1.0 / kf, -2.0 / (kf * l), 0.0, 1 / km],
                    [1.0 / kf, 0.0, 2.0 / (kf * l), -1.0 / km],
                    [1.0 / kf, 2.0 / (kf * l), 0.0, 1.0 / km],
                ]
            )
            / 4.0
        )

        return _invV

    def Rwb(self, alpha: np.ndarray) -> np.ndarray:
        """Rotation matrix from BODY to WORLD frame (ZYX Euler).

        Parameters
        ----------
        alpha : np.ndarray, shape=(3,)
            Roll, pitch, yaw vector.

        Returns
        -------
        R : np.ndarray, shape=(3, 3)
            Rotation matrix from BODY to WORLD frame.
        """
        assert alpha.shape == (3,)

        phi, theta, psi = alpha
        cphi = np.cos(phi)
        cth = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        sth = np.sin(theta)
        spsi = np.sin(psi)

        R = np.array(
            [
                [
                    cth * cpsi,  # row 1
                    sphi * sth * cpsi - cphi * spsi,
                    cphi * sth * cpsi + sphi * spsi,
                ],
                [
                    cth * spsi,  # row 2
                    sphi * sth * spsi + cphi * cpsi,
                    cphi * sth * spsi - sphi * cpsi,
                ],
                [-sth, sphi * cth, cphi * cth],  # row 3
            ]
        )

        return R

    def Twb(self, alpha: np.ndarray) -> np.ndarray:
        """Angular velocity transformation matrix from BODY to WORLD frame (ZYX Euler).

        Parameters
        ----------
        alpha : np.ndarray, shape=(3,)
            Roll, pitch, yaw vector.

        Returns
        -------
        T : np.ndarray, shape=(3, 3)
            Angular velocity transformation matrix from BODY to WORLD frame.
        """
        assert alpha.shape == (3,)

        phi, theta, _ = alpha
        cphi = np.cos(phi)
        cth = np.cos(theta)
        sphi = np.sin(phi)
        tth = np.tan(theta)

        T = np.array(
            [
                [1.0, sphi * tth, cphi * tth],
                [0.0, cphi, -sphi],
                [0.0, sphi / cth, cphi / cth],
            ]
        )

        return T

    def fdyn(self, t: float, s: np.ndarray) -> np.ndarray:
        """Quadrotor autonomous dynamics.

        Parameters
        ----------
        t : float,
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(12,)
            State of quadrotor.

        Returns
        -------
        _fdyn : np.ndarray, shape=(12,)
            Time derivatives of states from autonomous dynamics.
        """
        assert s.shape == (12,)

        # states
        alpha = s[3:6]  # phi, theta, psi
        do_b = s[6:9]  # u, v, w
        dalpha_b = s[9:12]  # p, q, r

        # moments of inertia
        Ix, Iy, Iz = self._I

        # body -> world transformations
        Rwb = self._Rwb(alpha)
        Twb = self._Twb(alpha)

        # velocities
        do = Rwb @ do_b
        dalpha = Twb @ dalpha_b

        # accelerations
        u, v, w = do_b
        p, q, r = dalpha_b
        phi, th, _ = alpha

        ddo_b = np.array(
            [
                r * v - q * w + g * np.sin(th),
                p * w - r * u - g * np.sin(phi) * np.cos(th),
                q * u - p * v - g * np.cos(th) * np.cos(phi),
            ]
        )
        ddalpha_b = np.array(
            [
                ((Iy - Iz) * q * r) / Ix,
                ((Iz - Ix) * p * r) / Iy,
                ((Ix - Iy) * p * q) / Iz,
            ]
        )

        _fdyn = np.hstack((do, dalpha, ddo_b, ddalpha_b))
        return _fdyn

    def gdyn(self, t: float, s: np.ndarray) -> np.ndarray:
        """Quadrotor control dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(12,)
            State of quadrotor.

        Returns
        -------
        _gdyn : np.ndarray, shape=(12, 4)
            Matrix representing affine control dynamics.
        """
        assert s.shape == (12,)

        # mass and inertias
        m = self._mass
        Ix, Iy, Iz = self._I

        # accelerations
        ddo_b = np.zeros((3, 4))
        ddo_b[2, 0] = 1.0 / m

        ddalpha_b = np.zeros((3, 4))
        ddalpha_b[0, 1] = 1.0 / Ix
        ddalpha_b[1, 2] = 1.0 / Iy
        ddalpha_b[2, 3] = 1.0 / Iz

        _gdyn = np.vstack((np.zeros((6, 4)), ddo_b, ddalpha_b))
        return _gdyn

    def wdyn(self, t: float, d: np.ndarray) -> np.ndarray:
        """Quadrotor disturbance dynamics in BODY frame.

        Global disturbances must first be rotated into the body frame!

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        d : np.ndarray, shape=(6,)
            Disturbances to the quadrotor.

        Returns
        -------
        w : np.ndarray, shape=(12,)
            Time derivatives of states from disturbance dynamics
        """
        assert d.shape == (6,)

        # mass and inertias
        m = self._mass
        Ix, Iy, Iz = self._I

        # disturbances
        fdx, fdy, fdz, taudx, taudy, taudz = d

        # accelerations
        ddo_b = np.array([fdx / m, fdy / m, fdz / m])
        ddalpha_b = np.array([taudx / Ix, taudy / Iy, taudy / Iz])

        _wdyn = np.hstack((np.zeros(6), ddo_b, ddalpha_b))
        return _wdyn

    def dyn(
        self,
        t: float,
        s: np.ndarray,
        u: np.ndarray,
        d: np.ndarray = np.zeros(6),
    ) -> np.ndarray:
        """Quadrotor dynamics function.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(12,)
            State of the quadrotor.
        u : np.ndarray, shape=(4,)
            Virtual input of the quadrotor.
        d : np.ndarray, shape=(6,), default=np.zeros(6)
            Disturbances to the quadrotor.

        Returns
        -------
        ds : np.ndarray, shape=(12,)
            Time derivative of the states.
        """
        assert s.shape == (12,)
        assert u.shape == (4,)
        assert d.shape == (6,)

        fdyn = self.fdyn(s)
        gdyn = self.gdyn(s)
        wdyn = self.wdyn(d)
        ds_gyro = np.zeros(12)

        # check whether gyroscopic effects are modeled. Note: this is
        # functionally treated as a disturbance, since if it is directly
        # modeled for control, the system is no longer control affine. However,
        # including this in simulations can help test for robustness.
        if self._Jtp is not None:
            Jtp = self._Jtp
            Ix, Iy, _ = self._I
            p = s[9]
            q = s[10]

            wsq = self.invV @ u
            assert all(wsq >= 0.0)
            w = np.sqrt(wsq)
            w[0] *= -1
            w[2] *= -1
            Omega = np.sum(w)  # net prop speeds

            ds_gyro[9] = -Jtp * q * Omega / Ix
            ds_gyro[10] = Jtp * p * Omega / Iy

        ds = fdyn + gdyn @ u + wdyn + ds_gyro
        return ds

    def A(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearized autonomous dynamics about (s, u).

        Parameters
        ----------
        s : np.ndarray, shape=(12,)
            State.
        u : np.ndarray, shape=(4,)
            Virtual input of the quadrotor. Unused, included for API compliance.

        Returns
        -------
        _A : np.ndarray, shape=(12, 12)
            Linearized autonomous dynamics about s.
        """
        assert s.shape == (12,)

        Ix, Iy, Iz = self._I

        phi, theta, psi = s[3:6]
        u, v, w = s[6:9]
        p, q, r = s[9:12]

        cphi = np.cos(phi)
        cth = np.cos(theta)
        cpsi = np.cos(psi)
        sphi = np.sin(phi)
        sth = np.sin(theta)
        spsi = np.sin(psi)
        tth = np.tan(theta)

        _A = np.zeros((12, 12))

        _A[0, 3:9] = np.array(
            [
                v * (sphi * spsi + cphi * cpsi * sth)
                + w * (cphi * spsi - cpsi * sphi * sth),
                w * cphi * cpsi * cth - u * cpsi * sth + v * cpsi * cth * sphi,
                w * (cpsi * sphi - cphi * spsi * sth)
                - v * (cphi * cpsi + sphi * spsi * sth)
                - u * cth * spsi,
                cpsi * cth,
                cpsi * sphi * sth - cphi * spsi,
                sphi * spsi + cphi * cpsi * sth,
            ]
        )
        _A[1, 3:9] = np.array(
            [
                -v * (cpsi * sphi - cphi * spsi * sth)
                - w * (cphi * cpsi + sphi * spsi * sth),
                w * cphi * cth * spsi - u * spsi * sth + v * cth * sphi * spsi,
                w * (sphi * spsi + cphi * cpsi * sth)
                - v * (cphi * spsi - cpsi * sphi * sth)
                + u * cpsi * cth,
                cth * spsi,
                cphi * cpsi + sphi * spsi * sth,
                cphi * spsi * sth - cpsi * sphi,
            ]
        )
        _A[2, 3:9] = np.array(
            [
                v * cphi * cth - w * cth * sphi,
                -u * cth - w * cphi * sth - v * sphi * sth,
                0.0,
                -sth,
                cth * sphi,
                cphi * cth,
            ]
        )
        _A[3, 3:5] = np.array(
            [
                q * cphi * tth - r * sphi * tth,
                r * cphi * (tth**2.0 + 1.0) + q * sphi * (tth**2.0 + 1.0),
            ]
        )
        _A[3, 9:12] = np.array([1.0, sphi * tth, cphi * tth])
        _A[4, 3] = -r * cphi - q * sphi
        _A[4, 10:12] = np.array([cphi, -sphi])
        _A[5, 3:5] = np.array(
            [
                (q * cphi) / cth - (r * sphi) / cth,
                (r * cphi * sth) / cth**2 + (q * sphi * sth) / cth**2,
            ]
        )
        _A[5, 10:12] = np.array([sphi / cth, cphi / cth])
        _A[6, 4] = g * cth
        _A[6, 7:12] = np.array([r, -q, 0.0, -w, v])
        _A[7, 3:12] = np.array(
            [-g * cphi * cth, g * sphi * sth, 0.0, -r, 0.0, p, w, 0.0, -u]
        )
        _A[8, 3:12] = np.array(
            [g * cth * sphi, g * cphi * sth, 0.0, q, -p, 0.0, -v, u, 0.0]
        )
        _A[9, 10:12] = np.array([(r * (Iy - Iz)) / Ix, (q * (Iy - Iz)) / Ix])
        _A[10, 9:12] = np.array([-(r * (Ix - Iz)) / Iy, 0.0, -(p * (Ix - Iz)) / Iy])
        _A[11, 9:11] = np.array([(q * (Ix - Iy)) / Iz, (p * (Ix - Iy)) / Iz])

        return _A

    def B(self, s: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearized control dynamics about (s, u).

        Parameters
        ----------
        s : np.ndarray, shape=(12,)
            State. Unused, included for API compliance.
        u : np.ndarray, shape=(4,)
            Virtual input of the quadrotor. Unused, included for API compliance.

        Returns
        -------
        _B : np.ndarray, shape=(12, 12)
            Linearized control dynamics about s.
        """
        m = self._mass
        Ix, Iy, Iz = self._I
        _B = np.zeros((12, 4))

        _B[8, 0] = 1.0 / m
        _B[9, 1] = 1.0 / Ix
        _B[10, 2] = 1.0 / Iy
        _B[11, 3] = 1.0 / Iz

        return _B

    def draw(self, ax: Axes, s: np.ndarray) -> None:
        """Draws the quadrotor on specified Axes.

        Parameters
        ----------
        ax : matplotlib.axes.Axes
            Axes object on which to draw the quadrotor.
        s : np.ndarray, shape=(12,)
            Current state of the quadrotor.
        """
        assert ax.name == "3d"
        assert s.shape == (12,)

        # quadrotor plotting params
        l = self._l
        o = s[0:3]  # x, y, z
        alpha = s[3:6]  # phi, theta, psi
        Rwb = self.Rwb(alpha)

        # rotor base locations on frame in inertial frame
        r1 = o + Rwb @ np.array([l, 0.0, 0.0])
        r2 = o + Rwb @ np.array([0.0, -l, 0.0])
        r3 = o + Rwb @ np.array([-l, 0.0, 0.0])
        r4 = o + Rwb @ np.array([0.0, l, 0.0])

        # rotor vertical offsets
        ro = Rwb @ np.array([0.0, 0.0, l / 10.0])
        r1o = r1 + ro
        r2o = r2 + ro
        r3o = r3 + ro
        r4o = r4 + ro

        # drawing quadrotor body and rotors
        ax.plot([r2[0], r4[0]], [r2[1], r4[1]], [r2[2], r4[2]], "k-")
        ax.plot([r1[0], r3[0]], [r1[1], r3[1]], [r1[2], r3[2]], "k-")
        ax.plot([r1[0], r1o[0]], [r1[1], r1o[1]], [r1[2], r1o[2]], "k-")
        ax.plot([r2[0], r2o[0]], [r2[1], r2o[1]], [r2[2], r2o[2]], "k-")
        ax.plot([r3[0], r3o[0]], [r3[1], r3o[1]], [r3[2], r3o[2]], "k-")
        ax.plot([r4[0], r4o[0]], [r4[1], r4o[1]], [r4[2], r4o[2]], "k-")
        draw_circle(ax, r1o, l / 2.0, ro, color="green")
        draw_circle(ax, r2o, l / 2.0, ro)
        draw_circle(ax, r3o, l / 2.0, ro)
        draw_circle(ax, r4o, l / 2.0, ro)
