import numpy as np
from dopri5 import dopri5
import time


def burger1ddf(Y, N, hx, t=None):
    """
    Parameters
    ----------
    Y : ndarray
        Current solution vector of length N.
    hx : float
        Spatial step size.
    N : int
        Number of spatial grid points.

    Returns
    -------
    F : ndarray
        Right-hand side vector of length N.
    """

    a = 0.01
    d = 0.3
    F = np.zeros_like(Y)
    for i in range(N):
        if i == 0:
            F[i] = (d / hx**2) * (Y[N-1] - 2*Y[i] + Y[i+1]) \
                   + Y[i] * (a / (2*hx)) * (Y[N-1] - Y[i+1]) \
                   + np.sin(Y[i]**2)
        elif i == N-1:
            F[i] = (d / hx**2) * (Y[i-1] - 2*Y[i] + Y[0]) \
                   + Y[i] * (a / (2*hx)) * (Y[i-1] - Y[0]) \
                   + np.sin(Y[i]**2)
        else:
            F[i] = (d / hx**2) * (Y[i-1] - 2*Y[i] + Y[i+1]) \
                   + Y[i] * (a / (2*hx)) * (Y[i-1] - Y[i+1]) \
                   + np.sin(Y[i]**2)
    return F



t_end = 0.5
d = 0.01
N = 150
hx = 1 / N
x = hx * np.arange(1, N + 1)
y0 = 1 + np.sin(2 * np.pi * x)

yz = y0.copy()
t = 0
h = t_end * 2 ** (-18)

start_time = time.time()

num_steps = int(t_end / h)
for k in range(num_steps):
    yz = dopri5(yz, N, hx, h, t, burger1ddf)


elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

np.savetxt('burger1dyz.txt', yz, fmt='%.15e')
