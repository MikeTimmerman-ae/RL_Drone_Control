import numpy as np
from dataclasses import dataclass
from flying_sim.drone import Drone


@dataclass
class Gains:
    k_x: float
    k_vx: float
    k_y: float
    k_vy: float
    k_theta: float
    k_omega: float


def create_gains_from_array(array: np.ndarray) -> Gains:
    if array.shape != (6, ):
        raise ValueError("Gain-Array must have exactly 6 elements!")

    return Gains(*array)


class PDController:
    def __init__(self, d: float) -> None:
        self.allocation_matrix = np.array([[1., 1.], [d, -d]], dtype=float)
        self.allocation_matrix_inv = np.linalg.inv(self.allocation_matrix)

    def compute_thrust(self, drone: Drone, gains: Gains, trajectory: np.ndarray) -> np.ndarray:
        """computes the thrust for both propellers given the current state of the drone, the controller gains and the desired state, given as a trajectory"""

        assert drone.state.shape == trajectory.shape == (6, )
        e_x = trajectory[0] - drone.state[0]
        e_vx = trajectory[1] - drone.state[1]
        e_y = trajectory[2] - drone.state[2]
        e_vy = trajectory[3] - drone.state[3]
        e_theta = trajectory[4] - drone.state[4]
        e_omega = trajectory[5] - drone.state[5]

        F_x = gains.k_x * e_x + gains.k_vx * e_vx
        F_y = gains.k_y * e_y + gains.k_vy * e_vy
        F = F_x * np.sin(drone.state[4]) + F_y * np.cos(drone.state[4])
        tau = gains.k_omega * e_omega + gains.k_theta * e_theta

        thrust = self.allocation_matrix_inv @ np.array(
            [F, tau], dtype=float)

        return thrust
