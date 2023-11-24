from flying_sim.drone import Drone

import numpy as np


class CascadedPD:
    def __init__(self, config, drone):
        self.planar_quad = drone
        self.Kp_x = config.cascaded_PD.Kp_x
        self.Kp_vx = config.cascaded_PD.Kp_vx
        self.Kp_theta = config.cascaded_PD.Kp_theta
        self.Kp_omega = config.cascaded_PD.Kp_omega
        self.Kp_y = config.cascaded_PD.Kp_y
        self.Kp_vy = config.cascaded_PD.Kp_vy

    def policy(self, x_ref: np.array) -> np.ndarray:
        """ Determine the control input based on cascaded PD control
        Args:
            x_ref: current system state reference (x_dim,)

        Returns: control input based on cascaded PD control
        """
        assert x_ref.shape == (self.planar_quad.x_dim,), f'State is of dimension {x.shape}'

        torque = self._torque_command(x_ref[0])
        thrust = self._thrust_command(x_ref[1])

        control_input = self._control_allocation(thrust, torque)

        return control_input

    def _thrust_command(self, y_ref) -> float:
        Vy_ref = self.Kp_y * (y_ref - self.planar_quad.state[1])
        thrust = self.Kp_vy * (Vy_ref - self.planar_quad.state[4]) + self.planar_quad.m * self.planar_quad.g

        return thrust

    def _torque_command(self, x_ref) -> float:
        Vx_ref = self.Kp_x * (x_ref - self.planar_quad.state[0])
        theta_ref = self.Kp_vx * (Vx_ref - self.planar_quad.state[3])
        omega_ref = self.Kp_theta * (theta_ref - self.planar_quad.state[2])
        torque = self.Kp_omega * (omega_ref - self.planar_quad.state[5])

        return torque

    def _control_allocation(self, thrust: float, torque: float) -> np.ndarray:
        l = self.planar_quad.l

        A = np.array([[1, 1], [-l, l]])
        b = np.array([thrust, torque])

        control_input = np.linalg.solve(A, b)
        return control_input
