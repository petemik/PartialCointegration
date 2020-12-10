from data import generatePair
import os
import time
import pandas as pd
import numpy as np
from minimize_likelihood import run_minimize


def compare_minimize():
    # Range of parameters to test.

    # Fixed parameters
    B = 1
    s_r = 1

    # Parameters we vary
    # p = [0.6, 0.7, 0.8, 0.9, 1]
    p = [0.6, 0.7, 0.8, 0.9, 1]
    s_m = 1
    # Length of the data
    n = [100, 1000]
    # 100 replications, varying s_m from 0->2 in steps of 0.5 and varying
    df = pd.DataFrame(columns=['n',
                               'B_hat', 'B_hat_std', 'B',
                               'p_hat', 'p_hat_std', 'p',
                               's_m_hat', 's_m_hat_std', 's_m',
                               's_r_hat', 's_r_hat_std', 's_r'])
    tic = time.time()
    # Table 2 uses Nelder-Mead whereas table 1 uses L-BFGS
    path_to_file = os.getcwd() + '\\results\\table4.csv'
    fail_count = 0
    for i in range(0, len(p)):
        for j in range(0, len(n)):
            estimate = np.array([0, 0, 0, 0])
            for k in range(0, 3):
                (X, m, r) = generatePair(beta=B, p=p[i], s_r=s_r, s_m=s_m, n=n[j], seed=False)
                try:
                    tic = time.time()
                    estimate_result = run_minimize(X)
                    print("Minimizing took {} seconds for a pair of length {}, iteration {}/99".format(time.time()-tic, n[j], k))
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
                   'B': B, 'p': p[i], 's_m': s_m, 's_r': s_r}
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
            print("Saved values for p={}, n={}".format(p[i], n[j]))

    #path_to_dir = os.getcwd() + '\\results\\'
    #df.to_csv(path_to_dir + "table1.csv")
    toc = time.time()
    print("process failed {} times".format(fail_count))
    print("Process took {} seconds".format(toc-tic))

compare_minimize()