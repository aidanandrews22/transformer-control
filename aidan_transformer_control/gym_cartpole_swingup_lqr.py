import os
import numpy as np
import math
from scipy.linalg import solve_continuous_are

def find_nearest_upright(theta):
    return np.pi * 2 * round(theta / (2 * np.pi))

def LQR_controller(state, cartmass, polemass, polelength):
    """
    state: [x, x_dot, theta, theta_dot]
    """
    x, x_dot, theta, theta_dot = state
    g = 9.81

    eq_theta = find_nearest_upright(theta)
    eq_pt = np.array([0.0, 0.0, eq_theta, 0.0])

    total_mass = cartmass + polemass
    alpha = polelength * (4 * polemass + cartmass) / (3 * total_mass)

    a23 = - (g * polemass * polelength) / (total_mass * alpha)
    a43 = g / alpha

    b2 = (1/total_mass) + (polemass * polelength) / (total_mass**2 * alpha)
    b4 = - 1 / (total_mass * alpha)

    A = np.array([
        [0, 1, 0, 0],
        [0, 0, a23, 0],
        [0, 0, 0, 1],
        [0, 0, a43, 0]
    ])

    B = np.array([
        [0],
        [b2],
        [0],
        [b4]
    ])

    Q = np.diag([1, 1, 1, 1])
    R = np.array([[.01]])


    # Solve the continuous-time algebraic Riccati equation
    P = solve_continuous_are(A, B, Q, R)
    # Compute the LQR gain
    K = np.linalg.inv(R) @ B.T @ P
    K = K.flatten()  # Ensure K is a 1D array

    return K



def swingup_lqr_controller(state, switched, cartmass, polemass, polelength):
# def swingup_lqr_controller(state, cartmass, polemass, polelength):
    """
    state: [x, x_dot, theta, theta_dot]
    switched: boolean indicating if the controller has switched to LQR (bool)
    """
    x, x_dot, theta, theta_dot = state
    g = 9.81

    eq_theta = find_nearest_upright(theta)
    eq_pt = np.array([0.0, 0.0, eq_theta, 0.0])
    # eq_pt = np.array([0.0, 0.0, 0.0, 0.0])  # Upright position
    # eq_theta = 0.0  # Upright position
    

    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)

    # Switching Conditions for LQR
    if switched or (np.abs(theta- eq_theta) < 0.3 and np.abs(theta_dot) < 0.5):
        # LQR fcn here later
        switched = True
        K_lqr = LQR_controller(state, cartmass, polemass, polelength)
        f = - K_lqr @ (state - eq_pt)
    else:
        # Energy-based controller
        K_e = 0.3
        K_p = 1
        K_d = 1

        Etilde = 0.5 * polemass * polelength**2 * theta_dot**2 + polemass * g * polelength * (1 - cos_theta) - polemass * g * polelength 
        # xpp_desired = -K_e * cos_theta * theta_dot * Etilde - K_p * x - K_d * x_dot
        # theta_pp = - cos_theta/polelength * xpp_desired - (g/polelength) * sin_theta
        # f = (cartmass + polemass) * xpp_desired + cos_theta * polemass * polelength* theta_pp - sin_theta * polemass * polelength * theta_dot**2
        f = -K_e * cos_theta * theta_dot * Etilde - K_p * x - K_d * x_dot

    return f, switched


