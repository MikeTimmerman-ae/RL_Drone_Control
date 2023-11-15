from dataclasses import dataclass
import numpy


@dataclass
class DroneConfig:
    dt: float
    tf: float
    t0: float
    x0: float
    m: float
    Inertia: float
    length: float
    thrust_mag: float


DEFAULT_DRONE_CONFIG = DroneConfig(
    dt=1., tf=10., t0=0., x0=0., m=10., Inertia=1., length=1., thrust_mag=1.)
