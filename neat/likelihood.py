from kalman import kalman
import numpy as np


def get_likelihood(params, Z):
    """
    This function returns the log_likelihood the paramaters you've guessed are correct for the underlying parameters.
    Minimizing this function should gives approximates for (Beta, p, s_m, s_r)

    :param params: The four parameters to calculate the likelihood for
    :param Z: The underlying data they are trying to match
    :return: The log_likelihood for the parameters
    """
    beta = params[0]
    row = params[1]
    s_m = params[2]
    s_r = params[3]
    filter = kalman(beta=beta, p=row, s_m=s_m, s_r=s_r)
    # Edit a couple of values of the filter to the ones specific for this dataset
    x0 = np.array([Z[0, 1], 0, 0]).T
    filter.x = x0
    filter.Q[0, 0] = np.std(np.diff(Z[:, 1]))
    len_Z = len(Z)
    log_lh = []
    for i in range(1, len_Z):
        filter.predict()
        filter.update(Z[i, :].T)
        if i == 1:
            log_lh.append(filter.log_likelihood)
        else:
            log_lh.append(filter.log_likelihood)
    # return negative of the log_lh as then we can minimize it.
    return -sum(log_lh)