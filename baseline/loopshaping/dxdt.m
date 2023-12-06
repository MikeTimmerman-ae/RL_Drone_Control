function dx = dxdt(u)
% Planar drone dynamics
g = 9.807;          % gravity (m / s**2)
m = 2.5;            % mass (kg)
l = 1.0;            % half-length (m)
I = 1.0;            % moment of inertia about the out-of-plane axis (kg * m**2)
Cd_v = 0;%.25;        % translational drag coefficient
Cd_phi = 0;%.02255;   % rotational drag coefficient

x = u(3); y = u(4); theta = u(5); v_x = u(6); v_y = u(7); omega = u(8);
T = u(1); tau = u(2);

A = [1 1;
    -l +l];
b = [T;
    tau];
u_c = linsolve(A, b);

T_1 = u_c(1); T_2 = u_c(2);

dx = [v_x;
    v_y;
    omega;
    (-(T_1 + T_2) * sin(theta) - Cd_v * v_x) / m;
    ((T_1 + T_2) * cos(theta) - Cd_v * v_y) / m - g;
    ((T_2 - T_1) * l - Cd_phi * omega) / I
    ];
end

