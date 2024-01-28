""" File contains evaluation metrics to compare the performanc of different controller """

import numpy as np


def ISE(state_traj: np.ndarray, ref_traj: np.ndarray, dt: float) -> float:
    ISE = np.sum(np.linalg.norm(state_traj[:, :2] - ref_traj[:, :2])) * dt
    return ISE


def ITSE(state_traj: np.ndarray, ref_traj: np.ndarray, dt: float) -> float:
    ITSE = np.sum(np.array(
        [i*dt * np.linalg.norm(state[:2] - ref[:2]) for i, state, ref in zip(list(range(len(state_traj))), state_traj, ref_traj)]
    )) * dt
    return ITSE
