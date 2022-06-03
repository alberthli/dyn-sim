from typing import Optional

import jax.numpy as jnp
import numpy as np
from matplotlib.axes import Axes

from dyn_sim.sys.sys_core import CtrlAffineSystem
from dyn_sim.util.jax_utils import jax_func
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
        Jtp: Optional[float] = None,
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

        # 12 states, 4 inputs, is a spatially 3D system
        super(Quadrotor, self).__init__(12, 4, True)
        self._mass = mass
        self._I = I
        self._kf = kf
        self._km = km
        self._l = l
        self._Jtp = Jtp

    @jax_func
    def V(self) -> jnp.ndarray:
        """Matrix converting squared rotor speeds to virtual forces/moments.

        Controller design occurs using virtual force/torque inputs, but true
        input is rotor speeds, so once the virtual inputs are designed, they
        are converted (in an invertible way) using _V.

        u = V @ wsq

        Returns
        -------
        _V : jnp.ndarray, shape=(4, 4)
            See above.
        """
        kf = self._kf
        km = self._km
        l = self._l

        _V = jnp.array(
            [
                [kf, kf, kf, kf],
                [0.0, -kf * l, 0.0, kf * l],
                [-kf * l, 0.0, kf * l, 0.0],
                [-km, km, -km, km],
            ]
        )

        return _V

    @jax_func
    def invV(self) -> jnp.ndarray:
        """Matrix converting virtual forces/moments to squared rotor speeds.

        wsq = invV @ u

        Returns
        -------
        _invV : jnp.ndarray, shape=(4, 4)
            Inverse of V. See above.
        """
        kf = self._kf
        km = self._km
        l = self._l

        _invV = (
            jnp.array(
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

    @jax_func
    def Rwb(self, alpha: np.ndarray) -> jnp.ndarray:
        """Rotation matrix from BODY to WORLD frame (ZYX Euler).

        Parameters
        ----------
        alpha : np.ndarray, shape=(3,)
            Roll, pitch, yaw vector.

        Returns
        -------
        R : jnp.ndarray, shape=(3, 3)
            Rotation matrix from BODY to WORLD frame.
        """
        assert alpha.shape == (3,)

        phi, theta, psi = alpha
        cphi = jnp.cos(phi)
        cth = jnp.cos(theta)
        cpsi = jnp.cos(psi)
        sphi = jnp.sin(phi)
        sth = jnp.sin(theta)
        spsi = jnp.sin(psi)

        R = jnp.array(
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

    @jax_func
    def Twb(self, alpha: np.ndarray) -> jnp.ndarray:
        """Angular velocity transformation matrix from BODY to WORLD frame (ZYX Euler).

        Parameters
        ----------
        alpha : np.ndarray, shape=(3,)
            Roll, pitch, yaw vector.

        Returns
        -------
        T : jnp.ndarray, shape=(3, 3)
            Angular velocity transformation matrix from BODY to WORLD frame.
        """
        assert alpha.shape == (3,)

        phi, theta, _ = alpha
        cphi = jnp.cos(phi)
        cth = jnp.cos(theta)
        sphi = jnp.sin(phi)
        tth = jnp.tan(theta)

        T = jnp.array(
            [
                [1.0, sphi * tth, cphi * tth],
                [0.0, cphi, -sphi],
                [0.0, sphi / cth, cphi / cth],
            ]
        )

        return T

    @jax_func
    def fdyn(self, t: float, s: np.ndarray) -> jnp.ndarray:
        """Quadrotor autonomous dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(12,)
            State of quadrotor.

        Returns
        -------
        _fdyn : jnp.ndarray, shape=(12,)
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
        Rwb = self.Rwb(alpha, np_out=False)
        Twb = self.Twb(alpha, np_out=False)

        # velocities
        do = Rwb @ do_b
        dalpha = Twb @ dalpha_b

        # accelerations
        u, v, w = do_b
        p, q, r = dalpha_b
        phi, th, _ = alpha

        ddo_b = jnp.array(
            [
                r * v - q * w + g * jnp.sin(th),
                p * w - r * u - g * jnp.sin(phi) * jnp.cos(th),
                q * u - p * v - g * jnp.cos(th) * jnp.cos(phi),
            ]
        )
        ddalpha_b = jnp.array(
            [
                ((Iy - Iz) * q * r) / Ix,
                ((Iz - Ix) * p * r) / Iy,
                ((Ix - Iy) * p * q) / Iz,
            ]
        )

        _fdyn = jnp.hstack((do, dalpha, ddo_b, ddalpha_b))
        return _fdyn

    @jax_func
    def gdyn(self, t: float, s: np.ndarray) -> jnp.ndarray:
        """Quadrotor control dynamics.

        Parameters
        ----------
        t : float
            Time. Unused, included for API compliance.
        s : np.ndarray, shape=(12,)
            State of quadrotor.

        Returns
        -------
        _gdyn : jnp.ndarray, shape=(12, 4)
            Matrix representing affine control dynamics.
        """
        assert s.shape == (12,)

        # mass and inertias
        m = self._mass
        Ix, Iy, Iz = self._I

        # accelerations
        ddo_b = jnp.zeros((3, 4))
        ddo_b = ddo_b.at[2, 0].set(1.0 / m)

        ddalpha_b = jnp.zeros((3, 4))
        ddalpha_b = ddalpha_b.at[0, 1].set(1.0 / Ix)
        ddalpha_b = ddalpha_b.at[1, 2].set(1.0 / Iy)
        ddalpha_b = ddalpha_b.at[2, 3].set(1.0 / Iz)

        _gdyn = jnp.vstack((jnp.zeros((6, 4)), ddo_b, ddalpha_b))
        return _gdyn

    @jax_func
    def wdyn(self, t: float, d: np.ndarray) -> jnp.ndarray:
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
        w : jnp.ndarray, shape=(12,)
            Time derivatives of states from disturbance dynamics
        """
        assert d.shape == (6,)

        # mass and inertias
        m = self._mass
        Ix, Iy, Iz = self._I

        # disturbances
        fdx, fdy, fdz, taudx, taudy, taudz = d

        # accelerations
        ddo_b = jnp.array([fdx / m, fdy / m, fdz / m])
        ddalpha_b = jnp.array([taudx / Ix, taudy / Iy, taudy / Iz])

        _wdyn = jnp.hstack((jnp.zeros(6), ddo_b, ddalpha_b))
        return _wdyn

    @jax_func
    def dyn(
        self,
        t: float,
        s: np.ndarray,
        u: np.ndarray,
        d: np.ndarray = np.zeros(6),
    ) -> jnp.ndarray:
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
        ds : jnp.ndarray, shape=(12,)
            Time derivative of the states.
        """
        assert s.shape == (12,)
        assert u.shape == (4,)
        assert d.shape == (6,)

        fdyn = self.fdyn(t, s, np_out=False)
        gdyn = self.gdyn(t, s, np_out=False)
        wdyn = self.wdyn(t, d, np_out=False)
        ds_gyro = jnp.zeros(12)

        # check whether gyroscopic effects are modeled. Note: this is
        # functionally treated as a disturbance, since if it is directly
        # modeled for control, the system is no longer control affine. However,
        # including this in simulations can help test for robustness.
        if self._Jtp is not None:
            Jtp = self._Jtp
            Ix, Iy, _ = self._I
            p = s[9]
            q = s[10]

            wsq = self.invV(np_out=False) @ u
            w = jnp.sqrt(wsq)
            w = w.at[0].set(w[0] * -1)
            w = w.at[2].set(w[2] * -1)
            Omega = jnp.sum(w)  # net prop speeds

            ds_gyro = ds_gyro.at[9].set(-Jtp * q * Omega / Ix)
            ds_gyro = ds_gyro.at[10].set(Jtp * p * Omega / Iy)

        ds = fdyn + gdyn @ u + wdyn + ds_gyro
        return ds

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
