import numpy as np
from filterpy.kalman import KalmanFilter


def kalman(beta, p, s_m, s_r):
    """
    This builds a Kalman filter which approximates the pair with preset parameters,
    using maximum likelihood we can then find the parameters which best match the pair,
    hence determining the underlying parameters of the model

    see data.py for detailed definitions of these parameters
    :param beta: The guessed parameter for Beta
    :param p: The guessed parameter for p
    :param s_m: The guessed parameter for s_m
    :param s_r: The guessed parameter for s_r
    :return: the kalman filter object given by fitlerpy (see https://filterpy.readthedocs.io/en/latest/kalman/KalmanFilter.html)
    """
    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = np.array([20, 0, 0]).T
    # Assumed parameters
    B = beta
    p = p
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