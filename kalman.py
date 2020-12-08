import numpy as np
import matplotlib.pyplot as plt
import random
from filterpy.kalman import KalmanFilter


def kalman(beta, row, s_m, s_r):
    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = np.array([20, 0, 0]).T
    # Assumed parameters
    B = beta
    p = row
    s_m = s_m
    s_r = s_r

    kf.F = np.array([[1, 0, 0],
                     [0, p, 0],
                     [0, 0, 1]])
    kf.H = np.array([[B, 1, 1],
                     [1, 0, 0]])
    # No measurement Noise
    kf.R = 0
    W = np.array([1, s_m**2, s_r**2])
    kf.Q = np.array([[W[0], 0, 0],
                     [0, W[1], 0],
                     [0, 0, W[2]]])
    kf.P *= 100
    return kf