import numpy as np
from scipy.optimize import minimize
from likelihood import get_likelihood



def run_minimize(Z):
    """
    Using the scipy minimize function to minimize the likelihood hence approximating the true parameters
    :return: the estimated parameters according to the minimizer
    """
    guesses = np.array([1, 0.5, 0.3, 0.3])
    results = minimize(get_likelihood, x0=guesses, args=(Z,), options={'disp': False}, method='Nelder-Mead')
    if results.success == 0:
        print("Optimizer failed {} values prediction: ".format(results.x))
    mle_estimate = results.x
    success = results.success
    return mle_estimate