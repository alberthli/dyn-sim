from typing import Callable

import numpy as np

from dyn_sim.ctrl.ctrl_core import Controller
from dyn_sim.sys.spatial.quad import Quadrotor

# constants
g = 9.80665  # gravitational acceleration


class PDQuadController(Controller):
    """A simple PD controller for position control of a quadrotor.

    NOTE: it's pretty hard to tune this controller, very finicky.
    """

    def __init__(
        self,
        quad: Quadrotor,
        kp_xyz: float,
        kd_xyz: float,
        kp_a: float,
        kd_a: float,
        ref: Callable[[float], np.ndarray],
    ) -> None:
        """Initialize a quadrotor PD controller.

        Parameters
        ----------
        quad: Quadrotor
            Quadrotor object to be controlled.
        kp: float
            Proportional gain.
        kd: float
            Derivative gain.
        ref: Callable[[float], np.ndarray]
            Reference function. Takes in time, outputs desired (x, y, z, psi).
            Assumes the desired pitch and roll are zero.
        """
        assert kp_xyz >= 0.0
        assert kd_xyz >= 0.0
        assert kp_a >= 0.0
        assert kd_a >= 0.0

        super(PDQuadController, self).__init__(quad)
        self._sys: Quadrotor
        self._kp_xyz = kp_xyz  # xy pd gains
        self._kd_xyz = kd_xyz  # attitude pd gains
        self._kp_a = kp_a
        self._kd_a = kd_a
        self._ref = ref

    def _rebalance(self, wsq_cand: np.ndarray) -> np.ndarray:
        """Rebalance true quadrotor inputs.

        Ensures candidate squared rotor speeds remain non-negative.

        Parameters
        ----------
        wsq_cand: np.ndarray, shape=(4,)
            Candidate squared rotor speeds.

        Returns
        -------
        wsq: np.ndarray, shape=(4,)
            Rebalanced squared rotor speeds.
        """
        assert wsq_cand.shape == (4,)

        if not any(wsq_cand < 0.0):
            return wsq_cand

        else:
            # recovering commanded correction values
            D = np.array(
                [  # cors -> wsq
                    [0.0, -1.0, -1, 1.0],
                    [-1.0, 0.0, 1.0, 1.0],
                    [0.0, 1.0, -1.0, 1.0],
                    [1.0, 0.0, 1.0, 1.0],
                ]
            )
            invD = (
                np.array(
                    [  # wsq -> cors
                        [0.0, -2.0, 0.0, 2.0],
                        [-2.0, 0.0, 2.0, 0.0],
                        [-1.0, 1.0, -1.0, 1],
                        [1.0, 1.0, 1.0, 1.0],
                    ]
                )
                / 4.0
            )
            cors = invD @ wsq_cand  # (phi, theta, psi, z)

            # z_off = (  # gravity offset
            #     self._sys.invV @ np.array([self._sys._mass * g, 0.0, 0.0, 0.0])
            # )[0]  # [TODO] am I supposed to use this?
            z_cor = cors[0]  # z correction
            max_vio = np.max(  # maximum non-negative violation occurs from here
                (np.abs(cors[0]) + np.abs(cors[1])),
                (np.abs(cors[0]) + np.abs(cors[2])),
                (np.abs(cors[1]) + np.abs(cors[2])),
            )

            # rebalance
            vio_ratio = max_vio / z_cor
            cors /= vio_ratio
            cors[0] = z_cor
            wsq = D @ cors

            assert all(wsq >= 0.0)
            return wsq

    def ctrl(self, t: float, s: np.ndarray) -> np.ndarray:
        """PD control law.

        This is an inner-outer loop controller, where the inner
        loop controls the states (z, phi, theta, psi) with PID and the outer
        loop sends desired values by converting errors in (x, y, z).

        Parameters
        ----------
        t: float
            Time.
        s: np.ndarray, shape=(n,)
            State.

        Returns
        -------
        i: np.ndarray, shape=(m,)
            Virtual control input (total thrust and 3 torques).
        """
        assert s.shape == (self._n,)

        # gains
        kp_xyz = self._kp_xyz
        kd_xyz = self._kd_xyz
        kp_a = self._kp_a
        kd_a = self._kd_a

        # reference
        ref = self._ref(t)
        x_d, y_d, z_d, psi_d = ref

        # state extraction
        x, y, z = s[0:3]
        phi, theta, psi = s[3:6]
        u, v, w = s[6:9]
        p, q, r = s[9:12]

        # outer loop: position control
        cpsi = np.cos(psi)
        spsi = np.sin(psi)

        e_x = x - x_d
        e_y = y - y_d
        e_xb = e_x * cpsi + e_y * spsi
        e_yb = -e_x * spsi + e_y * cpsi
        de_xb = u
        de_yb = v

        phi_d = -(-kp_xyz * e_yb - kd_xyz * de_yb)
        theta_d = -kp_xyz * e_xb - kd_xyz * de_xb

        # inner loop: attitude control
        e_phi = phi - phi_d
        e_theta = theta - theta_d
        e_psi = psi - psi_d
        e_z = z - z_d

        z_off = (  # gravity offset
            self._sys.invV @ np.array([self._sys._mass * g, 0.0, 0.0, 0.0])
        )[0]

        phi_cor = -kp_a * e_phi - kd_a * p
        theta_cor = -kp_a * e_theta - kd_a * q
        psi_cor = -kp_xyz * e_psi - kd_xyz * r  # not too aggressive
        z_cor = -kp_xyz * e_z - kd_xyz * w + z_off  # gravity offset
        z_cor = np.maximum(z_cor, 0.1)  # minimum correction to avoid freefall

        # rotor speed mixing law -> real inputs
        wsq = np.zeros(4)
        wsq[0] = z_cor - theta_cor - psi_cor
        wsq[1] = z_cor - phi_cor + psi_cor
        wsq[2] = z_cor + theta_cor - psi_cor
        wsq[3] = z_cor + phi_cor + psi_cor
        wsq = self._rebalance(wsq)

        # conversion to virtual inputs for simulation
        i = self._sys.V @ wsq

        return i
