from baseline.gain_scheduled import GainScheduled
from baseline.cascaded_PD import CascadedPD
from baseline.eval_performance import ISE, ITSE
from flying_sim.trajectory import Trajectory
from flying_sim.drone import Drone
from flying_sim.configs.config import Config
from stable_baselines3.ppo.ppo import PPO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('TkAgg')


def main():
    config = Config()
    # Configure reference trajectory
    trajectory = Trajectory(config)
    tf, f_sref, f_uref = trajectory.interp_trajectory(load_file=config.trajectory_config.training_files[2])

    # Configure drone
    planar_quad_1 = Drone(config)
    planar_quad_2 = Drone(config)
    planar_quad_3 = Drone(config)

    # Configure controllers
    gain_scheduled = GainScheduled(config, planar_quad_1)
    PD_cascaded = CascadedPD(config, planar_quad_2)
    PD_cascaded_RL = CascadedPD(config, planar_quad_3)
    RL_control = PPO.load(config.training.output_dir + 'policy')

    states_1 = []
    states_2 = []
    states_3 = []
    dt = 0.01
    t = [0]
    while t[-1] < 10:
        # Time step for gain schedule controlled drone
        x_nom, u_nom = f_sref(t[-1]), f_uref(t[-1])
        control = gain_scheduled.policy(x_nom, u_nom)
        planar_quad_1.step_RK4(control, dt)

        # Time step for PD cascade controlled drone
        x_ref = f_sref(t[-1])
        control = PD_cascaded.policy(x_ref)
        planar_quad_2.step_RK4(control, dt)

        # Time step for RL controlled drone
        obs = PD_cascaded_RL.return_errors(f_sref(t[-1])[0], f_sref(t[-1])[1])
        action = RL_control.predict(obs, deterministic=True)
        PD_cascaded_RL.configure_gains(action[0])
        control = PD_cascaded_RL.policy(x_ref)
        planar_quad_3.step_RK4(control, dt)

        # Log states and time
        states_1.append(planar_quad_1.state.copy())
        states_2.append(planar_quad_2.state.copy())
        states_3.append(planar_quad_3.state.copy())
        t.append(t[-1]+dt)

    t = np.array(t)
    states_1 = np.array(states_1)
    states_2 = np.array(states_2)
    states_3 = np.array(states_3)
    x_ref = f_sref(t)

    # Evaluate and plot tracking trajectories
    ISE_1 = ISE(states_1, x_ref, dt)
    ISE_2 = ISE(states_2, x_ref, dt)
    ISE_3 = ISE(states_3, x_ref, dt)
    ITSE_1 = ITSE(states_1, x_ref, dt)
    ITSE_2 = ITSE(states_2, x_ref, dt)
    ITSE_3 = ITSE(states_3, x_ref, dt)


    print('========================= Evaluation Metric =========================')
    print(f'The gain scheduled controller has ISE of {ISE_1}')
    print(f'The cascaded PD controller has ISE of {ISE_2}')
    print(f'The RL controller has ISE of {ISE_3}')
    print(f'The gain scheduled controller has ITSE of {ITSE_1}')
    print(f'The cascaded PD controller has ITSE of {ITSE_2}')
    print(f'The RL controller has ITSE of {ITSE_3}')

    plt.figure()
    plt.plot(x_ref[:, 0], x_ref[:, 1], 'r-', label='Reference Trajectory')
    plt.plot(states_1[:, 0], states_1[:, 1], 'g+', label='LQR Gain Scheduled Drone')
    plt.plot(states_2[:, 0], states_2[:, 1], 'b*', label='PD Cascaded Drone')
    plt.plot(states_3[:, 0], states_3[:, 1], 'y-', label='RL Drone')
    plt.legend()
    plt.show()

    return 0


if __name__ == '__main__':
    main()
