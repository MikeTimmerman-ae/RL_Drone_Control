clc;
close all;
clear;

run("loopshaping\tuning.m")

%% Condigure drone parameters
g = 9.807;          % gravity (m / s**2)
m = 2.5;            % mass (kg)
l = 1.0;            % half-length (m)
I = 1.0;            % moment of inertia about the out-of-plane axis (kg * m**2)
Cd_v = 0;%.25;        % translational drag coefficient
Cd_phi = 0;%.02255;   % rotational drag coefficient

%% Import trajectory

trajectory = csvread('trajectory.csv');
t = trajectory(:, 1);
x_ref = timeseries(trajectory(:, 2),t);
y_ref = timeseries(trajectory(:, 3),t);

%% Run simulation

x0 = [0, 5, 0, 0, 0, 0];
sim_out = sim('drone_PID.slx');

figure
hold on
plot(sim_out.sim_pos.Data(:,1), sim_out.sim_pos.Data(:,2))
plot(sim_out.sim_ref.Data(:,1), sim_out.sim_ref.Data(:,2))
hold off
grid on
ylabel('y-position')
xlabel('x-position')
title('Drone Trajectory Tracking')
legend('Drone position', 'Reference position')


figure
subplot(2,1,1)
hold on
plot(sim_out.sim_pos.Time, sim_out.sim_pos.Data(:,1))
plot(sim_out.sim_ref.Time, sim_out.sim_ref.Data(:,1))
hold off
grid on
title('Drone Trajectory Tracking')
legend('Drone position', 'Reference position')
ylabel('x-position')

subplot(2,1,2)
hold on
plot(sim_out.sim_pos.Time, sim_out.sim_pos.Data(:,2))
plot(sim_out.sim_ref.Time, sim_out.sim_ref.Data(:,2))
hold off
grid on
title('Drone Trajectory Tracking')
ylabel('y-position')