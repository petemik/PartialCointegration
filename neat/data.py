import numpy as np


def _create_m(m0, s_m, p, n=1000):
    """
    Creates a autoregressize model, set p=1 to generate a random walk.
    :param m0: Initial value of the PAR model.
    :param s_m: Standard deviation within with the error for every step is placed
    :param p: The AR(1) coefficient
    :param n: Number of datapoints to generate
    :return: The AR(1) model
    """
    m = [m0]
    for i in range(1, n):
        m.append(p*m[i - 1] + np.random.normal(scale=s_m))
    return np.array(m)


def generatePair(beta, p, s_m, s_r, n=1000, seed=True, s_x=0.0236, r0=0, m0=0, x0=20):
    """
    Generates a pair with a partially autoregressive spread (sum of a random walk and an autoregressive sequence)

    X_2_t = B*X_1_t + W_t
    W_t = M_t + R_t
    M_t = p*M_(t-1) + e_m
    R_t = 1*R_(t-1) + e_r

    Where e_m drawn from a N(0, s_m^2), and e_r drawn from a N(0, s_r^2) representing normal distributions.

    (see https://www.econstor.eu/bitstream/10419/140632/1/858609614.pdf for more details)

    These are the 4 parameters we are trying to find in the model.
    :param beta: Linear Relationship between the pair
    :param p: The AR coefficient to use for
    :param s_m: See above
    :param s_r: see above

    :param n: Length of the pair
    :param seed: Whether or not to set seed, setting seed yields a consistent pair back otherwise it will be random.
    :param s_x: see_ above
    :param r0: The starting point of the random walk
    :param m0: The starting point of the autoregressive sequence
    :param x0: Starting point of stock 1
    :return: A pair which are partially cointegrated and the M and R sequences
    """

    if seed:
        np.random.seed(73)
    r0 = r0
    m0 = m0
    s_x = s_x
    p = p
    B = beta
    r = _create_m(m0=r0, s_m=s_r, p=1, n=n)
    m = _create_m(m0=m0, s_m=s_m, p=p, n=n)
    w = r + m
    X_1 = _create_m(m0=x0, s_m=s_x, p=1, n=n)
    X_2 = B*X_1 + w
    X = np.array([X_2,
                  X_1])
    return X.T, m, r


if __name__=='__main__':
    (X, m, r) = generatePair(1, 0.9, 1, 1)

