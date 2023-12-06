import numpy as np
import random
import torch

import gymnasium as gym
from gymnasium import spaces

from flying_sim.drone import Drone
from flying_sim.configs.config import Config
from baseline.cascaded_PD import CascadedPD
from flying_sim.trajectory import Trajectory


class PIDFlightEnv(gym.Env):
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 4}

    def __init__(self):
        self.config: Config = Config()
        self.num_envs = self.config.training.num_processes
        self.n_steps = self.config.ppo.num_steps

        self.action_space = spaces.Box(low=-1, high=1, shape=(6,), dtype=np.float32)    # Normalized PD gains

        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(6,), dtype=np.float64)

        self.prev_deviation = 0
        self.error = 0
        self.reach_count = 0
        self.deviation_count = 0
        self.timeout_count = 0
        self.is_success = False

        # Log environment variables
        self.time = []
        self.reference = np.zeros((self.n_steps, 6))
        self.states = np.zeros((self.n_steps, 6))

        self.configure(self.config)

    def configure(self, config: Config):
        print("[INFO] Setting up Drone")
        self.drone: Drone = Drone(config)
        print("[INFO] Setting up Controller")
        self.pd_controller = CascadedPD(self.config, self.drone)
        print("[INFO] Setting up Trajectory")
        self.trajectory: Trajectory = Trajectory(config)
        self.final_time, self.traj_f, _ = self.trajectory.interp_trajectory(load_file=config.trajectory_config.training_files[1])

        self.target = np.array(
            self.config.trajectory_config.EGO_FINAL_GOAL_POS, dtype=float)

        self.time.append(config.env_config.t0)
        self.reference[0, :] = self.traj_f(self.time[0])
        self.states[0, :] = self.drone.state
        self.dt = config.env_config.dt
        print("[INFO] Finished setting up Environement")

    def set_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def _get_obs(self):
        ref = self.traj_f(self.time[-1])
        errors = self.pd_controller.return_errors(ref[0], ref[1])
        return errors

    def _get_info(self):
        return {'cur_state': self.drone.state,
                'cur_time': self.time[-1],
                'cur_reference': self.traj_f(self.time[-1]),
                'reach_count': self.reach_count,
                'deviation_count': self.deviation_count,
                'timeout_count': self.timeout_count,
                'is_success': self.is_success,
                'states': self.states[:len(self.time), :],
                'reference': self.reference[:len(self.time), :],
                'time': self.time}

    def reset(self, seed=None, options=None):

        observation = self._get_obs()

        info = self._get_info()

        # Reset environment
        super().reset(seed=seed)        # seed self.np_random

        self.drone.reset()
        self.time = [self.time[0]]
        self.reference = np.zeros((self.n_steps, 6))
        self.reference[0, :] = self.traj_f(self.time[0])
        self.states = np.zeros((self.n_steps, 6))
        self.states[0, :] = self.drone.state
        self.error = 0
        self.is_success = False

        return observation, info

    def step(self, action):
        assert action.shape == (6,)

        # Retrieve control input from controller
        self.pd_controller.configure_gains(action)
        desired_state: np.array = self.traj_f(self.time[-1])
        control_input = self.pd_controller.policy(desired_state)

        # Update drone dynamics according to control input
        self.drone.step_RK4(control=control_input, dt=self.dt)

        # Log progress
        self.time.append(self.time[-1] + self.dt)
        self.states[len(self.time)-1, :] = self.drone.state
        self.reference[len(self.time)-1, :] = desired_state

        # Check for terminal state
        reached = np.linalg.norm(self.drone.state[:2] - self.target) < 0.05
        deviation = np.linalg.norm(self.drone.state[:2] - desired_state[:2])
        deviated = deviation > 10.
        terminated = reached or self.time[-1] > self.final_time * 1.5 or deviated
        self.prev_deviation = deviation
        self.error += deviation

        if reached:
            # Drone reached its goal
            reward = 10 * len(self.time)/self.error
            self.reach_count += 1
            self.is_success = True
            print("Goal reached with reward: {}".format(reward))
        elif deviated:
            # Drone became unstable and deviated from path
            reward = -5
            self.deviation_count += 1
            print("Drone deviated with: {}".format(np.linalg.norm(self.drone.state[:2] - desired_state[:2])))
        elif terminated:
            # Drone did not reach goal in time
            reward = -1
            self.timeout_count += 1
            print("Simulation terminated!")
        else:
            # Anything else
            reward = (self.prev_deviation - deviation) / self.prev_deviation if deviation != 0 else 1

        observation = self._get_obs()

        info = self._get_info()

        return observation, reward, terminated, False, info
