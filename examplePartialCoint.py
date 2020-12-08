import numpy as np
import matplotlib.pyplot as plt
import random
from filterpy.kalman import KalmanFilter
from kalmanFilter import KalmanPete
from sklearn.preprocessing import MinMaxScaler
import time
from filterpy.common import Saver

def kalman(beta, row, s_m, s_r):
    kf = KalmanFilter(dim_x=3, dim_z=2)
    kf.x = np.array([30, 0, 0]).T
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

def pete_kalman(beta, row, s_m, s_r):
    kf = KalmanPete(dim_x=3, dim_z=2)
    kf.x = np.array([30, 0, 0]).T
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
    W = np.array([1, s_m**2, s_r**2])
    kf.Q = np.array([[W[0], 0, 0],
                     [0, W[1], 0],
                     [0, 0, W[2]]])
    kf.P *= 100
    return kf

def lh_for_params(params, X):
    beta = params[0]
    row = params[1]
    s_m = params[2]
    s_r = params[3]
    filter = kalman(beta=beta, row=row, s_m=s_m, s_r=s_r)
    len_X = len(X)
    pred = np.array([20, 0, 0])
    log_lh = []
    for i in range(1, len_X):
        filter.predict()
        filter.update(X[i, :].T)
        if i == 1:
            pred = filter.x
            log_lh.append(filter.log_likelihood)
        else:
            pred = np.vstack((pred, filter.x))
            log_lh.append(filter.log_likelihood)
    # return negative of the log_lh as then we can minimize it.
    return -sum(log_lh)


def create_r(r0, s_r, n=1000):
    R = [r0]
    for i in range(1, n):
        R.append(R[i-1] + np.random.normal(scale=s_r))
    return np.array(R)

def create_m(m0, s_m, p, n=1000):
    m = [m0]
    for i in range(1, n):
        m.append(p*m[i - 1] + np.random.normal(scale=s_m))
    return np.array(m)


def generatePair(beta, row, s_r, s_m, n=1000, seed=True):
    if seed:
        np.random.seed(73)
    r0 = 0
    s_x = 0.0236
    p = row
    B = beta
    r = create_r(r0=r0, s_r=s_r, n=n)
    m = create_m(m0=r0, s_m=s_m, p=p, n=n)
    w = r + m
    X_1 = create_r(r0=20, s_r=s_x, n=n)
    X_2 = B*X_1 + w
    X = np.array([X_2,
                  X_1])
    return X, m, r
    # fig, axs = plt.subplots(2)
    # axs[0].plot(X_1)
    # axs[0].plot(X_2)
    # axs[1].plot(X_2 - 2*X_1)
    # plt.show()
    # r_mr = np.var(np.diff(av_m))/np.var(np.diff(av_w))

if __name__=='__main__':

    (X, m, r) = generatePair(1, 0.9, 1, 1)


    X = X.T


    tracker = kalman(1, 0.9, 1, 1)
    tracker_pete = pete_kalman(1, 0.9, 1, 1)

    len_X = len(X)
    upd = np.array([30, 0, 0])
    pred = np.array([20, 0, 0])
    log_lh = []
    tic = time.time()
    for i in range(1, len_X):
        tracker.predict()
        pred = np.vstack((pred, tracker.x))
        tracker.update(X[i, :].T)
        upd = np.vstack((upd, tracker.x))
        #log_lh.append(tracker.log_likelihood)
    print("Iterative Method took {}s".format(time.time()-tic))
    _ = 1

