import numpy as np
import pygame

import gymnasium as gym
from gymnasium import spaces

from drone import Drone


class FlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config, render_mode=None):
        self.action_space = spaces.Box(low=0.0, high=1.0, shape=(2, 1), dtype=np.float32)

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(12, 1), dtype=np.float32)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.dt = config.dt
        self.configure(config)

    def configure(self, config):
        self.config = config
        self.drone = Drone(config)

    def _get_obs(self):
        return self.drone.x

    def _get_info(self):
        return None

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.drone = Drone(self.config)

        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        self.drone.step(action, self.config)

        # An episode is done iff the agent has reached the target
        terminated = np.linalg.norm(self.drone.x[:2] - self.config.target) < 1
        reward = 1 if terminated else 0  # Binary sparse rewards
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
