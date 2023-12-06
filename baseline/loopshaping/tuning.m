
%% Condigure drone parameters
g = 9.807;          % gravity (m / s**2)
m = 2.5;            % mass (kg)
l = 1.0;            % half-length (m)
I = 1.0;            % moment of inertia about the out-of-plane axis (kg * m**2)
Cd_v = 0;%.25;        % translational drag coefficient
Cd_phi = 0;%.02255;   % rotational drag coefficient

%% Linear feedback control angular verlocity (inner loop)

sim("drone_level_1.slx",[0,10]);

% Continuous-time model
A = drone_level_1_Timed_Based_Linearization.a;
B = drone_level_1_Timed_Based_Linearization.b;
C = drone_level_1_Timed_Based_Linearization.c;
D = drone_level_1_Timed_Based_Linearization.d;

sys_level_1 = ss(A,B,C,D);

% Feedback control
Kp_omega = 10;

sys_omega = tf(Kp_omega*sys_level_1(1,1));
sys_omega_cl = feedback(sys_omega, 1);

%% Linear feedback control attitude (inner loop)

sim("drone_level_2.slx",[0,10]);

% Continuous-time model
A = drone_level_2_Timed_Based_Linearization.a;
B = drone_level_2_Timed_Based_Linearization.b;
C = drone_level_2_Timed_Based_Linearization.c;
D = drone_level_2_Timed_Based_Linearization.d;

sys_level_2 = ss(A,B,C,D);

% Feedback control
Kp_theta = 5.6;

sys_theta = tf(Kp_theta*sys_level_2(1,1));
sys_theta_cl = feedback(sys_theta, 1);

%% Linear feedback control velocity (outer loop)

sim("drone_level_3.slx",[0,10]);

% Continuous-time model
A = drone_level_3_Timed_Based_Linearization.a;
B = drone_level_3_Timed_Based_Linearization.b;
C = drone_level_3_Timed_Based_Linearization.c;
D = drone_level_3_Timed_Based_Linearization.d;

sys_level_3 = ss(A,B,C,D);

% Feedback control
Kp_vx = -0.2;
Kp_vy = 7.5;

sys_vx = tf(Kp_vx*sys_level_3(1,1));
sys_vx_cl = feedback(sys_vx, 1);

sys_vy = tf(Kp_vy*sys_level_3(2,2));
sys_vy_cl = feedback(sys_vy, 1);

%% Linear feedback control position (outer loop)

sim("drone_level_4.slx",[0,10]);

% Continuous-time model
A = drone_level_4_Timed_Based_Linearization.a;
B = drone_level_4_Timed_Based_Linearization.b;
C = drone_level_4_Timed_Based_Linearization.c;
D = drone_level_4_Timed_Based_Linearization.d;

sys_level_4 = ss(A,B,C,D);

% Feedback control
Kp_x = 1.04;
Kp_y = 1.05;

sys_x = tf(Kp_x*sys_level_4(1,1));
sys_x_cl = feedback(sys_x, 1);

sys_y = tf(Kp_y*sys_level_4(2,2));
sys_y_cl = feedback(sys_y, 1);


subplot(2,1,1)
[x,tOut] = step(sys_x_cl, 5);
plot(tOut, x)
grid on
title('Position Tracking of Planar Drone')
ylabel('x-position')

subplot(2,1,2)
[y,tOut] = step(sys_y_cl, 5);
plot(tOut, y)
grid on
ylabel('y-position')
xlabel('time')