import numpy as np

import gymnasium as gym
from gymnasium import spaces

from flying_sim.drone import Drone
from flying_sim.config import Config
from flying_sim.controller import PDController, Gains, create_gains_from_array
from flying_sim.trajectory import Trajectory


class PIDFlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self, config: Config, render_mode=None):
        self.action_space = spaces.Box(
            low=0.0, high=1.0, shape=(6, ), dtype=np.float32)

        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(6, 1), dtype=np.float32)

        self.pd_controller = PDController(d=1)

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        self.window = None
        self.clock = None

        self.configure(config)

    def configure(self, config: Config):
        self.config: Config = config
        print("[INFO] Setting up Drone")
        self.drone: Drone = Drone(config)
        print("[INFO] Setting up Trajectory")
        self.trajectory: Trajectory = Trajectory(config)
        self.final_time, self.traj_f, _ = self.trajectory.interp_trajectory()

        self.target = np.array(
            self.config.trajectory_config.EGO_FINAL_GOAL_POS, dtype=float)

        self.time: list[float] = [config.env_config.t0]
        self.dt = config.env_config.dt
        print("[INFO] Finished setting up Environement")

    def _get_obs(self):
        return {"agent": self.drone.state, "trajectory": self.trajectory}

    def _get_info(self):
        return {}

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)

        self.drone.reset()
        self.time: list[float] = [self.config.env_config.t0]
        observation = self._get_obs()
        info = self._get_info()

        return observation, info

    def step(self, action):
        assert action.shape == (6, )

        controller_gains: Gains = create_gains_from_array(action)

        desired_state: np.array = self.traj_f(self.time[-1])

        thrust = self.pd_controller.compute_thrust(
            self.drone, controller_gains, desired_state)
        thrust = thrust.reshape((self.drone.u_dim,))
        thrust[thrust < 0] = 0

        self.drone.step_RK4(control=thrust, dt=self.dt)

        self.time.append(self.time[-1]+self.dt)

        terminated = np.linalg.norm(
            self.drone.state[:2] - self.target) < 1 or self.time[-1] > self.final_time - self.dt

        if terminated:
            print("Terminated")

        # TODO: Implement reward function
        reward = 1 if terminated else 0
        observation = self._get_obs()
        info = self._get_info()

        return observation, reward, terminated, False, info
