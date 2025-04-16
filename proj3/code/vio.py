#%% Imports

import numpy as np
from numpy.linalg import inv
from numpy.linalg import norm
from scipy.spatial.transform import Rotation


#%% Functions
def skew_symmetric(vector):
    a_x, a_y, a_z = vector
    return np.array([[0, -a_z, a_y],
                     [a_z, 0, -a_x],
                     [-a_y, a_x, 0]])

def nominal_state_update(nominal_state, w_m, a_m, dt):
    """
    function to perform the nominal state update

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                    all elements are 3x1 vectors except for q which is a Rotation object
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :return: new tuple containing the updated state
    """
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    # new_p = np.zeros((3, 1))
    # new_v = np.zeros((3, 1))
    # new_q = Rotation.identity()

    R = q.as_matrix()

    new_p = p + v * dt + 0.5 * (R @ (a_m-a_b) + g) * dt**2
    new_v = v + (R @ (a_m-a_b) + g) * dt
    new_q = q * Rotation.from_rotvec((w_m-w_b).reshape(3,) * dt)

    return new_p, new_v, new_q, a_b, w_b, g


def error_covariance_update(nominal_state, error_state_covariance, w_m, a_m, dt,
                            accelerometer_noise_density, gyroscope_noise_density,
                            accelerometer_random_walk, gyroscope_random_walk):
    """
    Function to update the error state covariance matrix

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param w_m: 3x1 vector - measured angular velocity in radians per second
    :param a_m: 3x1 vector - measured linear acceleration in meters per second squared
    :param dt: duration of time interval since last update in seconds
    :param accelerometer_noise_density: standard deviation of accelerometer noise
    :param gyroscope_noise_density: standard deviation of gyro noise
    :param accelerometer_random_walk: accelerometer random walk rate
    :param gyroscope_random_walk: gyro random walk rate
    :return:
    """

    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE
    R = q.as_matrix()

    F_x = np.identity(18)
    F_x[0:3, 3:6] = np.identity(3) * dt
    F_x[3:6, 6:9] = -R @ skew_symmetric((a_m-a_b).reshape(3,)) * dt
    F_x[3:6, 9:12] = -R * dt
    F_x[3:6, 15:18] = np.identity(3) * dt
    F_x[6:9, 6:9] = Rotation.from_rotvec((w_m-w_b).reshape(3,)*dt).as_matrix().T
    F_x[6:9, 12:15] = -np.identity(3) * dt

    F_i = np.zeros((18, 12))
    F_i[3:6, 0:3] = np.identity(3)
    F_i[6:9, 3:6] = np.identity(3)
    F_i[9:12, 6:9] = np.identity(3)
    F_i[12:15, 9:12] = np.identity(3)

    Q_i = np.zeros((12, 12))
    Q_i[0:3, 0:3] = accelerometer_noise_density**2 * dt**2 * np.identity(3)
    Q_i[3:6, 3:6] = gyroscope_noise_density ** 2 * dt ** 2 * np.identity(3)
    Q_i[6:9, 6:9] = accelerometer_random_walk ** 2 * dt * np.identity(3)
    Q_i[9:12, 9:12] = gyroscope_random_walk ** 2 * dt * np.identity(3)

    P = F_x @ error_state_covariance @ F_x.T + F_i @ Q_i @ F_i.T

    # return an 18x18 covariance matrix
    return P


def measurement_update_step(nominal_state, error_state_covariance, uv, Pw, error_threshold, Q):
    """
    Function to update the nominal state and the error state covariance matrix based on a single
    observed image measurement uv, which is a projection of Pw.

    :param nominal_state: State tuple (p, v, q, a_b, w_b, g)
                        all elements are 3x1 vectors except for q which is a Rotation object
    :param error_state_covariance: 18x18 initial error state covariance matrix
    :param uv: 2x1 vector of image measurements
    :param Pw: 3x1 vector world coordinate
    :param error_threshold: inlier threshold
    :param Q: 2x2 image covariance matrix
    :return: new_state_tuple, new error state covariance matrix
    """
    
    # Unpack nominal_state tuple
    p, v, q, a_b, w_b, g = nominal_state

    # YOUR CODE HERE - compute the innovation next state, next error_state covariance
    R0 = q.as_matrix()
    u_c, v_c = uv.reshape(2,)

    P_c0 = R0.T @ (Pw - p)
    x_c, y_c, z_c = P_c0.reshape(3,)

    innovation = uv - np.array([[x_c/z_c], [y_c/z_c]])

    if np.linalg.norm(innovation) > error_threshold:
        return (p, v, q, a_b, w_b, g), error_state_covariance, innovation

    # d_zt_d_Pc = np.array([[1, 0, -u_c],
    #                       [0, 1, -v_c]]) / z_c
    d_zt_d_Pc = np.array([[1, 0, -x_c/z_c],
                          [0, 1, -y_c/z_c]]) / z_c

    d_Pc_d_delta_theta = skew_symmetric(P_c0.reshape(3,))
    d_Pc_d_delta_p = -R0.T

    d_zt_d_delta_theta = d_zt_d_Pc @ d_Pc_d_delta_theta
    d_zt_d_delta_p = d_zt_d_Pc @ d_Pc_d_delta_p

    H_t = np.zeros((2, 18))
    H_t[:, 0:3] = d_zt_d_delta_p
    H_t[:, 6:9] = d_zt_d_delta_theta

    K_t = error_state_covariance @ H_t.T @ np.linalg.inv(H_t@error_state_covariance@H_t.T + Q)
    error_state_covariance = (np.identity(18)-K_t@H_t) @ error_state_covariance @ (np.identity(18)-K_t@H_t).T + K_t@Q@K_t.T

    prediction = np.array([[x_c/z_c], [y_c/z_c]])

    delta_x = K_t @ (uv - prediction)

    delta_p = delta_x[0:3]
    delta_v = delta_x[3:6]
    delta_theta = delta_x[6:9]
    delta_a_b = delta_x[9:12]
    delta_w_b = delta_x[12:15]
    delta_g = delta_x[15:18]

    p += delta_p
    v += delta_v
    q = q * Rotation.from_rotvec(delta_theta.reshape(3,))
    a_b += delta_a_b
    w_b += delta_w_b
    g += delta_g

    return (p, v, q, a_b, w_b, g), error_state_covariance, innovation
