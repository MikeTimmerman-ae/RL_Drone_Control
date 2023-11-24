from flying_sim.drone import Drone

from scipy.linalg import solve_continuous_are as ricatti_solver
import numpy as np


class GainScheduled:
    def __init__(self, config, drone):
        self.planar_quad = drone
        self.Q = config.gain_scheduled_config.Q
        self.R = config.gain_scheduled_config.R

    def policy(self, x_nom: np.array, u_nom: np.array) -> np.array:
        """ Determine the LQR gain for current linearization point
        Args:
            x_nom: nominal system state (x_dim,)
            u_nom: nominal system input (u_dim,)

        Returns: control input based on LQR gain for linearized system
        """
        assert x_nom.shape == (self.planar_quad.x_dim, ), f'State is of dimension {x_nom.shape}'
        assert u_nom.shape == (self.planar_quad.u_dim, ), f'State is of dimension {u_nom.shape}'

        A, B = self.planar_quad.get_continuous_jacobians(x_nom, u_nom)
        P = np.transpose(ricatti_solver(A, B, self.Q, self.R))
        K = np.linalg.inv(self.R).dot(np.transpose(B)).dot(P)

        control = u_nom - K.dot(self.planar_quad.state - x_nom)
        control = np.clip(control, self.planar_quad.min_thrust_per_prop, self.planar_quad.max_thrust_per_prop)

        assert K.shape == (self.planar_quad.u_dim, self.planar_quad.x_dim), f'Feedback gain matrix is of dimension {K.shape}'
        assert control.shape == (self.planar_quad.u_dim, ), f'Control input is of dimension {control.shape}'

        return control
