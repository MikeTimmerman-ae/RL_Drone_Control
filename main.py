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
    tf, f_sref, f_uref = trajectory.interp_trajectory(load_file=config.trajectory_config.training_files[0])

    # Configure drone
    planar_quad_1 = Drone(config)
    planar_quad_2 = Drone(config)
    planar_quad_3 = Drone(config)

    # Configure controllers
    gain_scheduled = GainScheduled(config, planar_quad_1)
    PD_cascaded = CascadedPD(config, planar_quad_2)
    PD_cascaded_RL = CascadedPD(config, planar_quad_3)
    RL_control = PPO.load(config.training.model_dir + 'model-v0')

    states_1 = []
    states_2 = []
    states_3 = []
    control_1 = []
    control_2 = []
    control_3 = []
    actions = []
    dt = 0.01
    t = [0]
    while t[-1] < 12:
        # Time step for gain schedule controlled drone
        x_nom, u_nom = f_sref(t[-1]), f_uref(t[-1])
        control = gain_scheduled.policy(x_nom, u_nom)
        planar_quad_1.step_RK4(control, dt)
        control_1.append(planar_quad_1.clip_control(control, dt))

        # Time step for PD cascade controlled drone
        x_ref = f_sref(t[-1])
        control = PD_cascaded.policy(x_ref)
        planar_quad_2.step_RK4(control, dt)
        control_2.append(planar_quad_2.clip_control(control, dt))

        # Time step for RL controlled drone
        obs = PD_cascaded_RL.return_errors(f_sref(t[-1])[0], f_sref(t[-1])[1])
        action = RL_control.predict(obs, deterministic=True)
        gains = PD_cascaded_RL.configure_gains(action[0])
        control = PD_cascaded_RL.policy(x_ref)
        planar_quad_3.step_RK4(control, dt)
        control_3.append(planar_quad_3.clip_control(control, dt))
        actions.append(gains)

        # Log states and time
        states_1.append(planar_quad_1.state.copy())
        states_2.append(planar_quad_2.state.copy())
        states_3.append(planar_quad_3.state.copy())
        t.append(t[-1]+dt)

    # Plot logged results
    t = np.array(t[:-1])
    states_1 = np.array(states_1)
    states_2 = np.array(states_2)
    states_3 = np.array(states_3)
    control_1 = np.array(control_1)
    control_2 = np.array(control_2)
    control_3 = np.array(control_3)
    actions = np.array(actions)
    x_ref = f_sref(t)
    u_ref = f_uref(t)

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

    # State Trajectories
    plt.figure('State Trajectories')
    plt.plot(x_ref[:, 0], x_ref[:, 1], 'r-', label='Reference Trajectory')
    # plt.plot(states_1[:, 0], states_1[:, 1], 'g+', label='LQR Gain Scheduled Drone')
    plt.plot(states_2[:, 0], states_2[:, 1], 'b-', label='Baseline Controller')
    plt.plot(states_3[:, 0], states_3[:, 1], 'y-', label='RL Controller')
    plt.xlabel('x-position [m]')
    plt.ylabel('y-position [m]')
    plt.grid()
    plt.legend()

    # State Time Trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, x_ref[:, 0], 'r-', label='Reference Trajectory')
    # ax1.plot(t, states_1[:, 0], 'g-', label='LQR Gain Scheduled Drone')
    ax1.plot(t, states_2[:, 0], 'b-', label='Baseline Controller')
    ax1.plot(t, states_3[:, 0], 'y-', label='RL Controller')
    ax1.set_ylabel('x-position [m]')
    ax1.grid()
    ax1.legend()

    ax2.plot(t, x_ref[:, 1], 'r-')
    # ax2.plot(t, states_1[:, 1], 'g-')
    ax2.plot(t, states_2[:, 1], 'b-')
    ax2.plot(t, states_3[:, 1], 'y-')
    ax2.set_ylabel('y-position [m]')
    ax2.set_xlabel('time [s]')
    ax2.grid()

    # Input Time Trajectories
    fig, (ax1, ax2) = plt.subplots(2, 1)
    ax1.plot(t, u_ref[:, 0], 'r-', label='Reference Input')
    # ax1.plot(t, control_1[:, 0], 'g-', label='LQR Gain Scheduled Drone')
    ax1.plot(t, control_2[:, 0], 'b-', label='Baseline Controller')
    ax1.plot(t, control_3[:, 0], 'y-', label='RL Controller')
    ax1.legend()

    ax2.plot(t, u_ref[:, 1], 'r-')
    # ax2.plot(t, control_1[:, 1], 'g-')
    ax2.plot(t, control_2[:, 1], 'b-')
    ax2.plot(t, control_3[:, 1], 'y-')

    # Gain Time Trajectories
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3)
    ax1.plot(t, config.RL_scheduled_config.lower_gains[0] * np.ones(len(t)), 'r-')
    ax1.text(0.5, config.RL_scheduled_config.lower_gains[0], 'Lower Limit', verticalalignment='bottom')
    ax1.plot(t, config.RL_scheduled_config.upper_gains[0] * np.ones(len(t)), 'r-')
    ax1.text(0.5, config.RL_scheduled_config.upper_gains[0], 'Upper Limit', verticalalignment='top')
    ax1.plot(t, config.cascaded_PD.Kp_x * np.ones(len(t)), 'b-', label='Manual gain')
    ax1.plot(t, actions[:, 0], 'g-', label='RL gain')
    ax1.set_title(r'$Kp_x$')
    ax1.set_ylabel('Value [-]')
    ax1.set_xlim([0, 12])
    ax1.grid()

    ax2.plot(t, config.RL_scheduled_config.lower_gains[1] * np.ones(len(t)), 'r-', label='Lower Limit')
    ax2.text(0.5, config.RL_scheduled_config.lower_gains[1], 'Lower Limit', verticalalignment='bottom')
    ax2.plot(t, config.RL_scheduled_config.upper_gains[1] * np.ones(len(t)), 'r-', label='Upper Limit')
    ax2.text(0.5, config.RL_scheduled_config.upper_gains[1], 'Upper Limit', verticalalignment='top')
    ax2.plot(t, config.cascaded_PD.Kp_vx * np.ones(len(t)), 'b-', label='Manual gain')
    ax2.plot(t, actions[:, 1], 'g-', label='RL gain')
    ax2.set_title(r'$Kp_{V_x}$')
    ax2.set_xlim([0, 12])
    ax2.grid()

    ax3.plot(t, config.RL_scheduled_config.lower_gains[2] * np.ones(len(t)), 'r-')
    ax3.text(0.5, config.RL_scheduled_config.lower_gains[2], 'Lower Limit', verticalalignment='bottom')
    ax3.plot(t, config.RL_scheduled_config.upper_gains[2] * np.ones(len(t)), 'r-')
    ax3.text(0.5, config.RL_scheduled_config.upper_gains[2], 'Upper Limit', verticalalignment='top')
    ax3.plot(t, config.cascaded_PD.Kp_theta * np.ones(len(t)), 'b-', label='Manual gain')
    ax3.plot(t, actions[:, 2], 'g-', label='RL gain')
    ax3.set_title(r'$Kp_{\theta}$')
    ax3.legend(loc='upper right', fontsize=8)
    ax3.set_xlim([0, 12])
    ax3.grid()

    ax4.plot(t, config.RL_scheduled_config.lower_gains[3] * np.ones(len(t)), 'r-', label='Lower Limit')
    ax4.text(0.5, config.RL_scheduled_config.lower_gains[3], 'Lower Limit', verticalalignment='bottom')
    ax4.plot(t, config.RL_scheduled_config.upper_gains[3] * np.ones(len(t)), 'r-', label='Upper Limit')
    ax4.text(0.5, config.RL_scheduled_config.upper_gains[3], 'Upper Limit', verticalalignment='top')
    ax4.plot(t, config.cascaded_PD.Kp_omega * np.ones(len(t)), 'b-', label='Manual gain')
    ax4.plot(t, actions[:, 3], 'g-', label='RL gain')
    ax4.set_title(r'$Kp_{\omega}$')
    ax4.set_xlabel('Time [s]')
    ax4.set_ylabel('Value [-]')
    ax4.set_xlim([0, 12])
    ax4.grid()

    ax5.plot(t, config.RL_scheduled_config.lower_gains[4] * np.ones(len(t)), 'r-', label='Lower Limit')
    ax5.text(0.5, config.RL_scheduled_config.lower_gains[4], 'Lower Limit', verticalalignment='bottom')
    ax5.plot(t, config.RL_scheduled_config.upper_gains[4] * np.ones(len(t)), 'r-', label='Upper Limit')
    ax5.text(0.5, config.RL_scheduled_config.upper_gains[4], 'Upper Limit', verticalalignment='top')
    ax5.plot(t, config.cascaded_PD.Kp_y * np.ones(len(t)), 'b-', label='Manual gain')
    ax5.plot(t, actions[:, 4], 'g-', label='RL gain')
    ax5.set_title(r'$Kp_y$')
    ax5.set_xlabel('Time [s]')
    ax5.set_xlim([0, 12])
    ax5.grid()

    ax6.plot(t, config.RL_scheduled_config.lower_gains[5] * np.ones(len(t)), 'r-', label='Lower Limit')
    ax6.text(0.5, config.RL_scheduled_config.lower_gains[5], 'Lower Limit', verticalalignment='bottom')
    ax6.plot(t, config.RL_scheduled_config.upper_gains[5] * np.ones(len(t)), 'r-', label='Upper Limit')
    ax6.text(0.5, config.RL_scheduled_config.upper_gains[5], 'Upper Limit', verticalalignment='top')
    ax6.plot(t, config.cascaded_PD.Kp_vy * np.ones(len(t)), 'b-', label='Manual gain')
    ax6.plot(t, actions[:, 5], 'g-', label='RL gain')
    ax6.set_title(r'$Kp_{V_y}$')
    ax6.set_xlabel('Time [s]')
    ax6.set_xlim([0, 12])
    ax6.grid()

    plt.tight_layout()
    plt.show()

    return 0


if __name__ == '__main__':
    main()
