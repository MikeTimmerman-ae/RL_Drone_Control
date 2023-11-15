import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
from scipy.interpolate import interp1d
import typing as T
from baseline.gain_scheduled import PlanarQuadrotor


class Trajectory:
    def __init__(self, config=None):

        self.EGO_START_POS, self.EGO_FINAL_GOAL_POS = (0.0, 5.0), (10.0, 7.0)
        self.EGO_RADIUS = 0.1

        self.s_0 = np.array([self.EGO_START_POS[0], 0., self.EGO_START_POS[1], 0., 0., 0.])
        self.s_f = np.array([self.EGO_FINAL_GOAL_POS[0], 0., self.EGO_FINAL_GOAL_POS[1], 0., 0., 0.])

        self.N = 50  # Number of time discretization nodes (0, 1, ... N).
        self.s_dim = 6  # State dimension; 3 for (x, y, vx, vy, theta, omega).
        self.u_dim = 2  # Control dimension; 2 for (T1, T2).
        self.T_max = 1  # Maximum thrust.

        self.g = 9.81
        self.l = 1
        self.I = 1
        self.m = 1

        self.planar_quad = PlanarQuadrotor()

        self.equilibrium_thrust = 0.5 * self.planar_quad.m * self.planar_quad.g
        self.x_dim = self.planar_quad.x_dim
        self.u_dim = self.planar_quad.u_dim

    def render_scene(self, traj=None, print_alpha=None):
        fig, ax = plt.subplots()
        ego_circle_start = plt.Circle(self.EGO_START_POS, radius=self.EGO_RADIUS, color='lime')
        ego_circle_end   = plt.Circle(self.EGO_FINAL_GOAL_POS, radius=self.EGO_RADIUS, color='red')
        if traj is not None:
            for i in range(traj.shape[0]):
                x, _, y, _, theta, _ = traj[i]
                ego_circle_current = plt.Circle((x, y), radius=self.EGO_RADIUS, color='cyan')
                ax.add_patch(ego_circle_current)
                ego_arrow_current = plt.arrow(x, y, dx=np.sin(theta)/2, dy=np.cos(theta)/2, head_width=0.1)
                ax.add_patch(ego_arrow_current)
        ax.add_patch(ego_circle_start)
        ax.add_patch(ego_circle_end)
        ax.set_xlim((-1.0, 11.0))
        ax.set_ylim((4.0, 10.0))
        ax.set_aspect('equal')
        if print_alpha is not None:
            plt.title("Alpha: {:.2f}".format(print_alpha))
        return plt

    def pack_decision_variables(self, final_time, states, controls):
        """Packs decision variables (final_time, states, controls) into a 1D vector.

        Args:
            final_time: scalar.
            states: array of shape (N + 1, x_dim).
            controls: array of shape (N, u_dim).
        Returns:
            An array `z` of shape (1 + (N + 1) * x_dim + N * u_dim,).
        """
        return np.concatenate([[final_time], states.ravel(), controls.ravel()])

    def unpack_decision_variables(self, z):
        """Unpacks a 1D vector into decision variables (final_time, states, controls).

        Args:
            z: array of shape (1 + (N + 1) * x_dim + N * u_dim,).
        Returns:
            final_time: scalar.
            states: array of shape (N + 1, x_dim).
            controls: array of shape (N, u_dim).
        """
        final_time = z[0]
        states = z[1:1 + (self.N + 1) * self.x_dim].reshape(self.N + 1, self.x_dim)
        controls = z[-self.N * self.u_dim:].reshape(self.N, self.u_dim)
        return final_time, states, controls

    def optimize_trajectory(self, N=50, verbose=False):
        planar_quad = PlanarQuadrotor()

        equilibrium_thrust = 0.5 * planar_quad.m * planar_quad.g
        x_dim = planar_quad.x_dim
        u_dim = planar_quad.u_dim

        def cost(z):
            final_time, states, controls = self.unpack_decision_variables(z)
            dt = final_time / N
            return final_time + dt * np.sum(np.square(controls - equilibrium_thrust))

        z_guess = self.pack_decision_variables(10, self.s_0 + np.linspace(0, 1, N + 1)[:, np.newaxis] * (self.s_f - self.s_0),
                                          equilibrium_thrust * np.ones((N, u_dim)))

        bounds = Bounds(
            self.pack_decision_variables(0., -np.inf * np.ones((N + 1, x_dim)),
                                    planar_quad.min_thrust_per_prop * np.ones((N, u_dim))),
            self.pack_decision_variables(np.inf, np.inf * np.ones((N + 1, x_dim)),
                                    planar_quad.max_thrust_per_prop * np.ones((N, u_dim))))

        def equality_constraints(z):
            final_time, states, controls = self.unpack_decision_variables(z)
            dt = final_time / N
            constraint_list = [states[i + 1] - planar_quad.discrete_step(states[i], controls[i], dt) for i in range(N)]
            constraint_list.append(states[0] - self.s_0)
            constraint_list.append(states[-1] - self.s_f)
            return np.concatenate(constraint_list)

        def inequality_constraints(z):
            final_time, states, controls = self.unpack_decision_variables(z)
            # Collision avoidance
            return np.sum(np.square(states[:, [0, 2]] - np.array([5, 5])), -1) - 3 ** 2

        result = minimize(cost,
                         z_guess,
                         bounds=bounds,
                         constraints=[{
                             "type": "eq",
                             "fun": equality_constraints
                         }, {
                             "type": "ineq",
                             "fun": inequality_constraints
                         }])
        if verbose:
            print(result.message)
        return self.unpack_decision_variables(result.x)

    def interp_trajectory(self):
        tf, s, u = self.optimize_trajectory(verbose=True)
        self.render_scene(s)

        t = np.linspace(0, tf, self.N)

        f_sref = interp1d(t, s[:-1], axis=0)
        f_uref = interp1d(t, u, axis=0)
        return tf, f_sref, f_uref
