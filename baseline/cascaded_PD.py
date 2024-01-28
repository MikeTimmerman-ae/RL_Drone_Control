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

        self.l_gains = config.RL_scheduled_config.lower_gains
        self.u_gains = config.RL_scheduled_config.upper_gains

    def policy(self, x_ref: np.array) -> np.ndarray:
        """ Determine the control input based on cascaded PD control
        Args:
            x_ref: current system state reference (x_dim,)

        Returns: control input based on cascaded PD control
        """
        torque = self._torque_command(x_ref[0])
        thrust = self._thrust_command(x_ref[1])

        control_input = self._control_allocation(thrust, torque)

        return control_input

    def configure_gains(self, controller_gains):
        self.Kp_x = self.l_gains[0] + (controller_gains[0] + 1) * (self.u_gains[0] - self.l_gains[0]) / 2
        self.Kp_vx = self.l_gains[1] + (controller_gains[1] + 1) * (self.u_gains[1] - self.l_gains[1]) / 2
        self.Kp_theta = self.l_gains[2] + (controller_gains[2] + 1) * (self.u_gains[2] - self.l_gains[2]) / 2
        self.Kp_omega = self.l_gains[3] + (controller_gains[3] + 1) * (self.u_gains[3] - self.l_gains[3]) / 2
        self.Kp_y = self.l_gains[4] + (controller_gains[4] + 1) * (self.u_gains[4] - self.l_gains[4]) / 2
        self.Kp_vy = self.l_gains[5] + (controller_gains[5] + 1) * (self.u_gains[5] - self.l_gains[5]) / 2
        return np.array([self.Kp_x, self.Kp_vx, self.Kp_theta, self.Kp_omega, self.Kp_y, self.Kp_vy])

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

    def return_errors(self, x_ref, y_ref):
        e_y = (y_ref - self.planar_quad.state[1])
        Vy_ref = self.Kp_y * e_y
        e_vy = (Vy_ref - self.planar_quad.state[4])

        e_x = x_ref - self.planar_quad.state[0]
        Vx_ref = self.Kp_x * e_x
        e_vx = Vx_ref - self.planar_quad.state[3]
        theta_ref = self.Kp_vx * e_vx
        e_theta = theta_ref - self.planar_quad.state[2]
        omega_ref = self.Kp_theta * e_theta
        e_omega = omega_ref - self.planar_quad.state[5]

        return np.array([e_x, e_vx, e_theta, e_omega, e_y, e_vy])
