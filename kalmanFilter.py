import numpy as np
from numpy import dot

class KalmanPete:
    def __init__(self, dim_z, dim_x):
        # In our case dim_z is 2 and dim_x is 3
        self.dim_z = dim_z
        self.dim_x = dim_x
        # How to get to the next time step in x: expected dimension is (dim_x, dim_x)
        self.F = None
        # Initial hidden state guess: expected dim  is (dim_x, 1)
        self.x0 = None
        # How the hidden state relates to the observables: expected dim is (dim_z, dim_x)
        self.H = None
        # The noise in the process when x_t ->x_(t+1): expected dim is (dim_x, dim_x)
        self.Q = None
        # the current value of x: expected dim is (dim_x, 1)
        self.x = self.x0
        # The initial guess for how wrong we are: expected dim is (dim_x, dim_x)
        self.P = np.eye(dim_x)

    def predict(self):
        # Predict next X value
        # next_x = dot(self.F, self.x)
        # next_P = dot(self.F, dot(self.P, self.F.T)) + self.Q
        self.x = dot(self.F, self.x)
        self.P = dot(self.F, dot(self.P, self.F.T)) + self.Q

    def update(self, z):
        # Calculate Kalman Gain: expected dim is (dim_x, dim_z)
        K = dot(dot(self.P, self.H.T), np.linalg.inv(dot(dot(self.H, self.P), self.H.T)))
        self.x += dot(K, z - dot(self.H, self.x))
        self.P = dot((np.eye(self.dim_x) - dot(K, self.H)), self.P)

    def update_batch(self, z):
        # Here z should have dimensions (n, dim_z), where n is the number of valyes for z available
        update = None
        for i in range(0, np.shape(z)[0]):
            self.predict()
            self.update(z[i, :].T)
            if i == 0:
                update = self.x
            else:
                update = np.vstack((update, self.x))
        return update






