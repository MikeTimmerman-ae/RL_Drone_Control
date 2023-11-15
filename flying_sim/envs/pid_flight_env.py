import numpy as np

import gymnasium as gym
from gymnasium import spaces

from flying_sim.drone import Drone
from flying_sim.config import DroneConfig, DEFAULT_DRONE_CONFIG
from flying_sim.controller import PDController, Gains, create_gains_from_array


def get_dummy_trajectory():
    return np.array([1., 0., 1., 0., 0., 0.], dtype=float)


class PIDFlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: DroneConfig = None, render_mode=None):
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(6, 1), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6, 1), dtype=np.float32)

        self.pd_controller = PDController(d=1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        if config is not None:
            self.configure(config)
        else:
            print("[INFO] Using Default Drone Configurations")
            self.configure(DEFAULT_DRONE_CONFIG)

    def configure(self, config: DroneConfig):
        self.config: DroneConfig = config
        self.drone: Drone = Drone(config)
        # x, vx, y, vy, theta, omega
        self.trajectory = get_dummy_trajectory()
        target_x: float = 10.
        target_y: float = 10.
        self.target = np.array(
            [target_x, target_y], dtype=float)

    def _get_obs(self):
        return {"agent": self.drone.state, "trajectory": self.trajectory}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.drone = Drone(self.config)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        assert action.shape == (6, 1)
        controller_gains: Gains = create_gains_from_array(action)
        thrust = self.pd_controller.compute_thrust(
            self.drone, controller_gains, self.trajectory)
        self.drone.step(thrust, self.config)

        # An episode is done iff the agent has reached the target
        terminated = np.linalg.norm(self.drone.state[:2] - self.target) < 1
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
