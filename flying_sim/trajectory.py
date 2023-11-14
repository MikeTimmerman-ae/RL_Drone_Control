import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import minimize, Bounds
import typing as T

EGO_START_POS, EGO_FINAL_GOAL_POS = (0.0, 0.0), (5.0, 5.0)
EGO_RADIUS = 0.1

s_0 = np.array([EGO_START_POS[0], EGO_START_POS[1], 0, 0, np.pi/2, 0])  # Initial state.
s_f = np.array([EGO_FINAL_GOAL_POS[0], EGO_FINAL_GOAL_POS[1], 0, 0, np.pi/2, 0])  # Final state.

N = 50  # Number of time discretization nodes (0, 1, ... N).
s_dim = 6  # State dimension; 3 for (x, y, vx, vy, theta, omega).
u_dim = 2  # Control dimension; 2 for (T1, T2).
T_max = 1  # Maximum thrust.

g = 9.81
l = 1
I = 1
m = 1


def render_scene(traj=None, print_alpha=None):
    fig, ax = plt.subplots()
    ego_circle_start = plt.Circle(EGO_START_POS, radius=EGO_RADIUS, color='lime')
    ego_circle_end   = plt.Circle(EGO_FINAL_GOAL_POS, radius=EGO_RADIUS, color='red')
    if traj is not None:
        for i in range(traj.shape[0]):
            x, y, _, _, theta, _ = traj[i]
            ego_circle_current = plt.Circle((x, y), radius=EGO_RADIUS, color='cyan')
            ax.add_patch(ego_circle_current)
            ego_arrow_current = plt.arrow(x, y, dx=np.cos(theta)/2, dy=np.sin(theta)/2, head_width=0.1)
            ax.add_patch(ego_arrow_current)
    ax.add_patch(ego_circle_start)
    ax.add_patch(ego_circle_end)
    ax.set_xlim((-1.0, 6.0))
    ax.set_ylim((-1.0, 6.0))
    ax.set_aspect('equal')
    if print_alpha is not None:
        plt.title("Alpha: {:.2f}".format(print_alpha))
    return plt


def pack_decision_variables(t_f: float, s: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Packs decision variables (final time, states, controls) into a 1D vector.

    Args:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).

    Returns:
        An array `z` of shape (1 + (N + 1) * s_dim + N * u_dim,).
    """
    return np.concatenate([[t_f], s.ravel(), u.ravel()])


def unpack_decision_variables(z: np.ndarray) -> T.Tuple[float, np.ndarray, np.ndarray]:
    """Unpacks a 1D vector into decision variables (final time, states, controls).

    Args:
        z: An array of shape (1 + (N + 1) * s_dim + N * u_dim,).

    Returns:
        t_f: Final time, a scalar.
        s: States, an array of shape (N + 1, s_dim).
        u: Controls, an array of shape (N, u_dim).
    """
    t_f = float(z[0])
    s = z[1:1 + (N + 1) * s_dim].reshape(N + 1, s_dim)
    u = z[-N * u_dim:].reshape(N, u_dim)
    return t_f, s, u


def optimize_trajectory(
        time_weight: float = 1.0,
        verbose: bool = True
):
    """Computes the optimal trajectory as a function of `time_weight`.

    Args:
        time_weight: \alpha in the HW writeup.

    Returns:
        t_f_opt: Final time, a scalar.
        s_opt: States, an array of shape (N + 1, s_dim).
        u_opt: Controls, an array of shape (N, u_dim).
    """

    def cost(z) -> float:
        ############################## Code starts here ##############################
        # TODO: Define a cost function here
        # HINT: you may find `unpack_decision_variables` useful here. z is the packed 1D representation of t,s and u. Return the value of the cost.
        t_f, s, u = unpack_decision_variables(z)
        dt = t_f / N

        cost_value = 0.0
        for i in range(N):
            cost_value += (time_weight + np.linalg.norm(u[i])) * dt

        return cost_value
        ############################## Code ends here ##############################

    # Initialize the trajectory with a straight line
    z_guess = pack_decision_variables(
        20, s_0 + np.linspace(0, 1, N + 1)[:, np.newaxis] * (s_f - s_0),
        np.ones(N * u_dim))

    # Minimum and Maximum bounds on states and controls
    # This is because we would want to include safety limits
    # for omega (steering) and velocity (speed limit)
    bounds = Bounds(
        pack_decision_variables(0., -np.inf * np.ones((N + 1, s_dim)), np.array([-T_max, -T_max]) * np.ones((N, u_dim))),
        pack_decision_variables(np.inf, np.inf * np.ones((N + 1, s_dim)), np.array([T_max, T_max]) * np.ones((N, u_dim)))
    )

    # Define the equality constraints
    def eq_constraints(z):
        t_f, s, u = unpack_decision_variables(z)
        dt = t_f / N
        constraint_list = []
        for i in range(N):
            T1, T2 = u[i]
            x, y, vx, vy, theta, omega = s[i]
            x_n, y_n, vx_n, vy_n, theta_n, omega_n = s[i + 1]
            ############################## Code starts here ##############################
            # TODO: Append to `constraint_list` with dynanics constraints
            dvx = (T1 + T2) * np.sin(theta) / m
            dvy = (T1 + T2) * np.cos(theta) / m - g
            domega = (T2 - T1) * l / I

            constraint_list.append(np.array(
                [(x + dt * vx) - x_n,
                 (y + dt * vy) - y_n,
                 (vx + dt * dvx) - vx_n,
                 (vy + dt * dvy) - vx_n,
                 (theta + dt * omega) - theta_n,
                 (omega + dt * domega) - omega_n]
            ))
            ############################## Code ends here ##############################

        ############################## Code starts here ##############################
        # TODO: Append to `constraint_list` with initial and final state constraints
        constraint_list.append(np.array([s[0][0] - s_0[0], s[N][0] - s_f[0],
                                         s[0][1] - s_0[1], s[N][1] - s_f[1],
                                         s[0][2] - s_0[2], s[N][2] - s_f[2],
                                         s[0][3] - s_0[3], s[N][3] - s_f[3],
                                         s[0][4] - s_0[4], s[N][4] - s_f[4],
                                         s[0][5] - s_0[5], s[N][5] - s_f[5]]))
        ############################## Code ends here ##############################
        return np.concatenate(constraint_list)

    result = minimize(cost,
                      z_guess,
                      bounds=bounds,
                      constraints={'type': 'eq', 'fun': eq_constraints})
    if verbose:
        print(result.message)

    return unpack_decision_variables(result.x)


t, s, u = optimize_trajectory(time_weight=1, verbose=True)
plt = render_scene(s)
plt.show()
