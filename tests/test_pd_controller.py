import unittest

import numpy as np
from flying_sim.configs.config import DroneConfig

from tests.controller import Gains, PDController
from flying_sim.drone import Drone


class TestPDController(unittest.TestCase):
    def test_compute_thrust(self):
        d = 1.0
        controller = PDController(d)

        drone_state = np.array([1, 0, 1, 0, -np.pi/10, 0], dtype=float)
        trajectory = np.array([1.5, 0, 1.5, 0, 0, 0], dtype=float)

        drone_config = DroneConfig(
            dt=1., tf=10., t0=0., state_0=drone_state, m=10., Inertia=1., length=2 * d, thrust_mag=1.)

        drone = Drone(drone_config)

        gains = Gains(k_x=1.0, k_vx=1.0, k_y=1.0,
                      k_vy=1.0, k_theta=1.0, k_omega=1.0)

        thrust = controller.compute_thrust(drone, gains, trajectory)

        self.assertIsInstance(thrust, np.ndarray)
        self.assertEqual(thrust.shape, (2, 1))


if __name__ == '__main__':
    unittest.main()
