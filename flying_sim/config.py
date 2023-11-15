from dataclasses import asdict, dataclass
import numpy as np


@dataclass
class DroneConfig:
    dt: float
    tf: float
    t0: float
    state_0: np.ndarray
    m: float
    Inertia: float
    length: float
    thrust_mag: float

    def dict(self):
        """ Turns the current dataclass into a python dictionary """
        return {str(k): v for k, v in asdict(self).items()}


DEFAULT_DRONE_CONFIG = DroneConfig(
    dt=1., tf=10., t0=0., state_0=np.zeros(6), m=10., Inertia=1., length=1., thrust_mag=1.)
