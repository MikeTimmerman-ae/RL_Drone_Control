import numpy as np
from scipy.linalg import solve_continuous_are as ricatti_solver

import numpy as np
import scipy.interpolate


class PlanarQuadrotor:

    def __init__(self):
        # Dynamics constants
        self.x_dim = 6  # state dimension (see dynamics below)
        self.u_dim = 2  # control dimension (see dynamics below)
        self.g = 9.807  # gravity (m / s**2)
        self.m = 2.5  # mass (kg)
        self.l = 1.0  # half-length (m)
        self.Iyy = 1.0  # moment of inertia about the out-of-plane axis (kg * m**2)
        self.Cd_v = 0.25  # translational drag coefficient
        self.Cd_phi = 0.02255  # rotational drag coefficient

        # Control constraints
        self.max_thrust_per_prop = 0.75 * self.m * self.g  # total thrust-to-weight ratio = 1.5
        self.min_thrust_per_prop = 0  # at least until variable-pitch quadrotors become mainstream :D

    def ode(self, state, control):
        """Continuous-time dynamics of a planar quadrotor expressed as an ODE."""
        x, v_x, y, v_y, phi, omega = state
        T_1, T_2 = control
        return np.array([
            v_x,
            (-(T_1 + T_2) * np.sin(phi) - self.Cd_v * v_x) / self.m,
            v_y,
            ((T_1 + T_2) * np.cos(phi) - self.Cd_v * v_y) / self.m - self.g,
            omega,
            ((T_2 - T_1) * self.l - self.Cd_phi * omega) / self.Iyy,
        ])

    def discrete_step(self, state, control, dt):
        """Discrete-time dynamics (Euler-integrated) of a planar quadrotor."""
        # RK4 would be more accurate, but this runs more quickly in a homework problem;
        # in this notebook we use Euler integration for both control and simulation for
        # illustrative purposes (i.e., so that planning and simulation match exactly).
        # Often simulation may use higher fidelity models than those used for planning/
        # control, e.g., using `scipy.integrate.odeint` here for much more accurate
        # (and expensive) integration.
        return state + dt * self.ode(state, control)

    def get_continuous_jacobians(self, state_nominal, control_nominal):
        """Continuous-time Jacobians of planar quadrotor, written as a function of input state and control"""
        x, v_x, y, v_y, phi, omega = state_nominal
        T_1, T_2 = control_nominal
        A = np.array([[0., 1., 0., 0., 0., 0.],
                      [0., -self.Cd_v / self.m, 0., 0., -(T_1 + T_2) * np.cos(phi) / self.m, 0.],
                      [0., 0., 0., 1., 0., 0.],
                      [0., 0., 0., -self.Cd_v / self.m, -(T_1 + T_2) * np.sin(phi) / self.m, 0.],
                      [0., 0., 0., 0., 0., 1.],
                      [0., 0., 0., 0., 0., -self.Cd_phi / self.Iyy]])
        B = np.array([[0., 0.],
                      [-np.sin(phi) / self.m, -np.sin(phi) / self.m],
                      [0., 0.],
                      [np.cos(phi) / self.m, np.cos(phi) / self.m],
                      [0., 0.],
                      [-self.l / self.Iyy, self.l / self.Iyy]])
        return A, B


class GainScheduled:
    def __init__(self, trajectory=None):
        self.trajectory = trajectory
        self.planar_quad = PlanarQuadrotor()

    def find_closest_nominal_state(self, current_state):
        dist = []
        for x_t in self.trajectory:
            dist_i = np.linalg.norm(current_state - x_t)
            dist.append(dist_i)
        min_val = min(dist)
        closest_state_idx = dist.index(min_val)
        return closest_state_idx

    def policy(self, x, u):
        Q = 100 * np.diag([1., 0.1, 1., 0.1, 0.1, 0.1])
        R = 1e0 * np.diag([1., 1.])

        A, B = self.planar_quad.get_continuous_jacobians(x, u)
        P = np.transpose(ricatti_solver(A, B, Q, R))
        K = np.linalg.inv(R).dot(np.transpose(B)).dot(P)
        return K
