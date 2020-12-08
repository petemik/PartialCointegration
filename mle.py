import numpy as np
import matplotlib.pyplot as plt
import random
from filterpy.kalman import KalmanFilter
from scipy.optimize import minimize
from kalman import kalman
from examplePartialCoint import generatePair
from sklearn import preprocessing
import time
import pandas as pd
import os
import sys

class MaxLikelihood:
    def __init__(self, _Z):
        # This is the data, in form (n_samples, 2 columns (price of S2, price of S1))
        self.Z = _Z
        # Initial prediction, take the first element of S_1 and then guess 0 for both stds
        self.x0 = np.array([self.Z[0, 1], 0, 0]).T
        # Predetermine order of parameters Beta, row, s_m, s_r

    def get_likelihood(self, params):
        beta = params[0]
        row = params[1]
        s_m = params[2]
        s_r = params[3]
        filter = kalman(beta=beta, row=row, s_m=s_m, s_r=s_r)
        # Edit a couple of values of the filter to the ones specific for this dataset
        filter.x = self.x0
        filter.Q[0, 0] = np.std(np.diff(self.Z[:, 1]))
        len_Z = len(self.Z)
        pred = self.x0
        log_lh = []
        for i in range(1, len_Z):
            filter.predict()
            filter.update(self.Z[i, :].T)
            if i == 1:
                pred = filter.x
                log_lh.append(filter.log_likelihood)
            else:
                pred = np.vstack((pred, filter.x))
                log_lh.append(filter.log_likelihood)
        # return negative of the log_lh as then we can minimize it.
        return -sum(log_lh)


    def run_minimize(self):
        guesses = np.array([1, 0.5, 0.3, 0.3])
        results = minimize(self.get_likelihood, x0=guesses, options={'disp': False}, method='Nelder-Mead')
        if results.success==0:
            print("Optimizer failed {} values prediction: ".format(results.x))
        mle_estimate = results.x
        return mle_estimate


def compare_minimize():
    # Range of parameters to test.

    # Fixed parameters
    B = 1
    p = 0.9
    s_r = 1
    # Parameters we vary
    # s_m = np.arange(0, 2.1, 0.5)
    # n = [100, 1000, 10000]
    s_m = [0.5, 1, 1.5, 2]
    # Length of the data
    n = [1000, 1000]
    # 100 replications, varying s_m from 0->2 in steps of 0.5 and varying
    df = pd.DataFrame(columns=['n',
                               'B_hat', 'B_hat_std', 'B',
                               'p_hat', 'p_hat_std', 'p',
                               's_m_hat', 's_m_hat_std', 's_m',
                               's_r_hat', 's_r_hat_std', 's_r'])
    tic = time.time()
    # Table 2 uses Nelder-Mead whereas table 1 uses L-BFGS
    path_to_file = os.getcwd() + '\\results\\table2.csv'
    fail_count = 0
    for i in range(0, len(s_m)):
        for j in range(0, len(n)):
            estimate = np.array([0, 0, 0, 0])
            for k in range(0, 100):
                (X, m, r) = generatePair(beta=B, row=p, s_r=s_r, s_m=s_m[i], n=n[j], seed=False)
                X = X.T
                try:
                    tic = time.time()
                    ml = MaxLikelihood(X)
                    estimate_result = ml.run_minimize()
                    print("Minimizing took {} seconds for a pair of length {}".format(time.time()-tic, n[j]))
                except:
                    fail_count += 1
                    continue
                if k == 0:
                    estimate = estimate_result
                else:
                    estimate = np.vstack((estimate, estimate_result))
            estimate_avg = np.mean(estimate, axis=0)
            estimate_std = np.std(estimate, axis=0)
            row = {'n': n[j], 'B_hat': estimate_avg[0], 'p_hat': estimate_avg[1], 's_m_hat': estimate_avg[2], 's_r_hat': estimate_avg[3],
                   'B_hat_std': estimate_std[0], 'p_hat_std': estimate_std[1], 's_m_hat_std': estimate_std[2], 's_r_hat_std': estimate_std[3],
                   'B': B, 'p': p, 's_m': s_m[i], 's_r': s_r}
            try:
                # Need to delete the file between runs.
                df = pd.read_csv(path_to_file, index_col=0)
            except FileNotFoundError:
                df = pd.DataFrame(columns=['n',
                                           'B_hat', 'B_hat_std', 'B',
                                           'p_hat', 'p_hat_std', 'p',
                                           's_m_hat', 's_m_hat_std', 's_m',
                                           's_r_hat', 's_r_hat_std', 's_r'])
            df = df.append(row, ignore_index=True)
            df.to_csv(path_to_file)
            print("Saved values for s_m = {}, n={}".format(s_m[i], n[j]))

    #path_to_dir = os.getcwd() + '\\results\\'
    #df.to_csv(path_to_dir + "table1.csv")
    toc = time.time()
    print("process failed {} times".format(fail_count))
    print("Process took {} seconds".format(toc-tic))




# B = 2
# p = 0.9
# s_x = 0.0236
# s_m = 0.5
# s_r = 1

# (X, m, r) = generatePair(beta=B, row=p, s_r=s_r, s_m=s_m, n=1000)
# X = X.T
compare_minimize()
# plt.plot(X[:, 0])
# plt.plot(X[:, 1])
# plt.show()
# exit()
# ml = MaxLikelihood(X)
# ml.run_minimize()
