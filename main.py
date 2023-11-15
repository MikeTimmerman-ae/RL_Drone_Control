from baseline.gain_scheduled import GainScheduled
from flying_sim.trajectory import Trajectory
from flying_sim.drone import Drone
from flying_sim.config import Config

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def main():
    config = Config()
    # Configure reference trajectory
    trajectory = Trajectory(config)
    tf, f_sref, f_uref = trajectory.interp_trajectory()

    # Configure drone
    planar_quad = Drone(config)

    # Configure controllers
    gain_scheduled = GainScheduled(config)

    states = [f_sref(0)]
    dt = 0.01
    t = [0]
    while t[-1] < tf-dt:
        x_nom, u_nom = f_sref(t[-1]), f_uref(t[-1])
        K = gain_scheduled.policy(x_nom, u_nom)
        control = u_nom - K.dot(states[-1] - x_nom)
        control = np.clip(control, planar_quad.min_thrust_per_prop, planar_quad.max_thrust_per_prop)

        next_state = planar_quad.step_RK4(states[-1], control, dt)
        states.append(next_state)

        t.append(t[-1]+dt)

    t = np.array(t)
    states = np.array(states)
    x_ref = f_sref(t)

    plt.figure()
    plt.plot(x_ref[:, 0], x_ref[:, 1], 'r-', states[:, 0], states[:, 1], 'g+')
    plt.show()

    return 0


if __name__ == '__main__':
    main()
