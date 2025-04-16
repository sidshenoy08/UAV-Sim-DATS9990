import numpy as np

from .graph_search import graph_search

from numpy.core.fromnumeric import cumsum

import matplotlib.pyplot as plt
from scipy.linalg import lstsq

# from traj_generation import full_continuous_trajectory, full_boundary_trajectory

## Plotting
def plot_dynamics(t, c_traj):
    fig, axes = plt.subplots(4, 1, figsize=(10, 8), sharex=True)
    labels = ["Position", "Velocity", "Acceleration", "Jerk"]

    num_of_segments = len(t) - 1

    for i in range(num_of_segments):
        t1 = t[i]
        t2 = t[i + 1]
        c_i = c_traj[i * 6: i * 6 + 6]

        t_vals = np.linspace(t1, t2, 20)

        data = [
            np.polyval(c_i, t_vals),
            np.polyval(np.polyder(c_i, 1), t_vals),
            np.polyval(np.polyder(c_i, 2), t_vals),
            np.polyval(np.polyder(c_i, 3), t_vals),
        ]

        for j, ax in enumerate(axes):
            ax.plot(t_vals, data[j], label=f"Segment {i + 1}")
            ax.set_ylabel(labels[j])
            ax.grid(True)

    axes[-1].set_xlabel("Time")
    fig.suptitle("Polynomial Trajectory: Position, Velocity, Acceleration, and Jerk")
    plt.show()

## Min Jerk QP Constraints
def position_constraint(t1, p1):

    A = np.array([
        [t1**5, t1**4, t1**3, t1**2, t1, 1]
    ])

    b = np.array([p1])

    return A, b

def boundary_constraint(t1):

    A = np.array([
        [5*t1**4, 4*t1**3, 3*t1**2, 2*t1, 1, 0],
        [20*t1**3, 12*t1**2, 6*t1, 2, 0, 0]
    ])

    return A

def continuity_constraint(t1):

    A = np.array([
        [5 * t1 ** 4, 4 * t1 ** 3, 3 * t1 ** 2, 2 * t1, 1, 0, -5 * t1 ** 4, -4 * t1 ** 3, -3 * t1 ** 2, -2 * t1, -1, 0],
        [20 * t1 ** 3, 12 * t1 ** 2, 6 * t1, 2, 0, 0, -20 * t1 ** 3, -12 * t1 ** 2, -6 * t1, -2, 0, 0]
    ])

    return A

def qp_equality_constraints(t, p):

    num_of_segments = len(t)-1

    A = np.zeros((num_of_segments*6, num_of_segments*6))
    b = np.zeros(num_of_segments*6)

    constraint_counter = 0

    # position constraints
    for i in range(num_of_segments):

        A_i, b_i = position_constraint(t[i], p[i])

        A[constraint_counter, i*6: i*6+6] = A_i
        b[constraint_counter] = b_i

        constraint_counter += 1

        A_i_plus_1, b_i_plus_1 = position_constraint(t[i+1], p[i+1])
        A[constraint_counter, i*6 : i*6+6] = A_i_plus_1
        b[constraint_counter] = b_i_plus_1

        constraint_counter += 1

    position_counter = constraint_counter

    # continuity constraints
    for i in range(num_of_segments-1):

        A_i = continuity_constraint(t[i+1])

        A[constraint_counter: constraint_counter+2, i*6: i*6+12] = A_i

        constraint_counter += 2

    continuity_counter = constraint_counter

    A_0 = boundary_constraint(t[0])
    A[constraint_counter : constraint_counter+2, 0: 6] = A_0
    constraint_counter += 2

    A_m = boundary_constraint(t[-1])
    A[constraint_counter : constraint_counter+2, -6: ] = A_m
    constraint_counter += 2

    return A, b

## Min Jerk QP Objective
def cost_matrix(t1, t2):

    H = np.zeros((6, 6))
    H[0:3, 0:3] = np.array([
        [720*(t2**5-t1**5), 360*(t2**4-t1**4), 120*(t2**3-t1**3)],
        [360*(t2**4-t1**4), 192*(t2**3-t1**3), 72*(t2**2-t1**2)],
        [120*(t2**3-t1**3), 72*(t2**2-t1**2), 36*(t2-t1)]
    ])

    return H

def qp_objective_matrix(t):

    num_of_segments = len(t) - 1

    H = np.zeros((num_of_segments * 6, num_of_segments * 6))

    for i in range(num_of_segments):

        H[i*6: i*6+6, i*6: i*6+6] = cost_matrix(t[i], t[i+1])

    return H

## Min Jerk QP Solver
def solve_qp(Q, A, b):
    # Form the augmented matrix for the system:
    # [ Q  A^T ]
    # [ A  0   ]
    augmented_matrix = np.block([
        [Q, A.T],
        [A, np.zeros((A.shape[0], A.shape[0]))]
    ])

    # Right-hand side of the system
    rhs = np.concatenate([np.zeros(Q.shape[0]), b])

    # Solve the system of equations using least squares
    x, residuals, rank, s = lstsq(augmented_matrix, rhs)

    # Extract the solution for x
    x = x[:Q.shape[0]]

    return x

def min_jerk_trajectory(timestamps, points):

    points_x = [point[0] for point in points]
    points_y = [point[1] for point in points]
    points_z = [point[2] for point in points]

    H = qp_objective_matrix(timestamps)

    A_x, b_x = qp_equality_constraints(timestamps, points_x)

    c_x = solve_qp(H, A_x, b_x)

    A_y, b_y = qp_equality_constraints(timestamps, points_y)
    c_y = solve_qp(H, A_y, b_y)

    A_z, b_z = qp_equality_constraints(timestamps, points_z)
    c_z = solve_qp(H, A_z, b_z)

    # plot_dynamics(timestamps, c_x)
    # plot_dynamics(timestamps, c_y)
    # plot_dynamics(timestamps, c_z)

    return c_x, c_y, c_z

def timestamp(speed, points):
    direction = np.diff(points, axis=0)
    norm = np.linalg.norm(direction, axis=1)
    norm = norm[:, np.newaxis]

    duration = norm / speed
    t = np.concatenate(([0], cumsum(duration)))

    return t

class WorldTraj(object):
    """

    """
    def __init__(self, world, start, goal):
        """
        This is the constructor for the trajectory object. A fresh trajectory
        object will be constructed before each mission. For a world trajectory,
        the input arguments are start and end positions and a world object. You
        are free to choose the path taken in any way you like.

        You should initialize parameters and pre-compute values such as
        polynomial coefficients here.

        Parameters:
            world, World object representing the environment obstacles
            start, xyz position in meters, shape=(3,)
            goal,  xyz position in meters, shape=(3,)

        """

        # You must choose resolution and margin parameters to use for path
        # planning. In the previous project these were provided to you; now you
        # must chose them for yourself. Your may try these default values, but
        # you should experiment with them!
        self.resolution = np.array([0.25, 0.25, 0.25])
        self.margin = 0.6

        # You must store the dense path returned from your Dijkstra or AStar
        # graph search algorithm as an object member. You will need it for
        # debugging, it will be used when plotting results.
        self.path, _ = graph_search(world, self.resolution, self.margin, start, goal, astar=True)

        # You must generate a sparse set of waypoints to fly between. Your
        # original Dijkstra or AStar path probably has too many points that are
        # too close together. Store these waypoints as a class member; you will
        # need it for debugging and it will be used when plotting results.
        # self.points = np.zeros((1,3)) # shape=(n_pts,3)
        self.points = self.path  # shape=(n_pts,3)

        # Finally, you must compute a trajectory through the waypoints similar
        # to your task in the first project. One possibility is to use the
        # WaypointTraj object you already wrote in the first project. However,
        # you probably need to improve it using techniques we have learned this
        # semester.

        # STUDENT CODE HERE

        self.speed = 2.5
        # self.speed = 2.9
        self.timestamps = timestamp(self.speed, self.points)

        self.c_x, self.c_y, self.c_z = min_jerk_trajectory(self.timestamps, self.points)
        # self.c_x, self.c_y, self.c_z = full_continuous_trajectory(self.timestamps, self.points)
        # self.c_x, self.c_y, self.c_z = full_boundary_trajectory(self.timestamps, self.points)

    def update(self, t):
        """
        Given the present time, return the desired flat output and derivatives.

        Inputs
            t, time, s
        Outputs
            flat_output, a dict describing the present desired flat outputs with keys
                x,        position, m
                x_dot,    velocity, m/s
                x_ddot,   acceleration, m/s**2
                x_dddot,  jerk, m/s**3
                x_ddddot, snap, m/s**4
                yaw,      yaw angle, rad
                yaw_dot,  yaw rate, rad/s
        """
        x        = np.zeros((3,))
        x_dot    = np.zeros((3,))
        x_ddot   = np.zeros((3,))
        x_dddot  = np.zeros((3,))
        x_ddddot = np.zeros((3,))
        yaw = 0
        yaw_dot = 0

        # STUDENT CODE HERE

        if len(self.points) == 1:
            flat_output = {'x': x, 'x_dot': x_dot, 'x_ddot': x_ddot, 'x_dddot': x_dddot, 'x_ddddot': x_ddddot,
                           'yaw': yaw, 'yaw_dot': yaw_dot}
            return flat_output

        if t > self.timestamps[-1]:
            x = self.points[-1]
            x_dot = np.zeros(3)
        else:
            i = np.searchsorted(self.timestamps, t, side="right")

            l_i, u_i = 6 * i - 6, 6 * i

            c_x = self.c_x[l_i:u_i]
            c_y = self.c_y[l_i:u_i]
            c_z = self.c_z[l_i:u_i]

            c_x_dot = np.polyder(c_x)  # velocity
            c_y_dot = np.polyder(c_y)
            c_z_dot = np.polyder(c_z)

            x = np.array([np.polyval(c_x, t), np.polyval(c_y, t), np.polyval(c_z, t)])
            x_dot = np.array([np.polyval(c_x_dot, t), np.polyval(c_y_dot, t), np.polyval(c_z_dot, t)])

        flat_output = { 'x':x, 'x_dot':x_dot, 'x_ddot':x_ddot, 'x_dddot':x_dddot, 'x_ddddot':x_ddddot,
                        'yaw':yaw, 'yaw_dot':yaw_dot}
        return flat_output
