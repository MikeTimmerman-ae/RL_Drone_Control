from flying_sim.drone import Drone

from scipy.linalg import solve_continuous_are as ricatti_solver
import numpy as np


class GainScheduled:
    def __init__(self, config):
        self.planar_quad = Drone(config)
        self.Q = config.gain_scheduled_config.Q
        self.R = config.gain_scheduled_config.R

    def policy(self, x: np.array, u: np.array) -> np.array:
        """ Determine the LQR gain for current linearization point
        Args:
            x: current system state (x_dim,)
            u: current system input (u_dim,)

        Returns: LQR gain K for linearized system
        """
        assert x.shape == (self.planar_quad.x_dim, ), f'State is of dimension {x.shape}'
        assert u.shape == (self.planar_quad.u_dim, ), f'State is of dimension {u.shape}'

        A, B = self.planar_quad.get_continuous_jacobians(x, u)
        P = np.transpose(ricatti_solver(A, B, self.Q, self.R))
        K = np.linalg.inv(self.R).dot(np.transpose(B)).dot(P)

        assert K.shape == (self.planar_quad.u_dim, self.planar_quad.x_dim), f'Feedback gain matrix is of dimension {K.shape}'
        return K
