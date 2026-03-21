import numpy as np
import time
from dopri5 import dopri5
from rkce import rkce
t_end = 1
d = 0.01
N = 799
hx = 2 / (N + 1)
x = np.zeros(N)
for j in range(N):
    x[j] = -1 + hx * (j + 1)

y0 = np.zeros(N)
for j in range(N):
    xx = x[j]
    y0[j] = 0.53 * xx + 0.47 * np.sin(-1.5 * np.pi * xx)



def ac1ddf(Y, N, hx,t=None):
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
    d = 0.01
    F = np.zeros_like(Y)
    for i in range(N):
        if i == 0:
            F[i] = (d / hx ** 2) * (-1 - 2 * Y[i] + Y[i + 1]) + Y[i] - Y[i] ** 3
        elif i == N - 1:
            F[i] = (d / hx ** 2) * (Y[i - 1] - 2 * Y[i] + 1) + Y[i] - Y[i] ** 3
        else:
            F[i] = (d / hx ** 2) * (Y[i - 1] - 2 * Y[i] + Y[i + 1]) + Y[i] - Y[i] ** 3
    return F





np.set_printoptions(precision=15)



yz = y0.copy()
t = 0
h=2**(-18)

start_time = time.time()


num_steps = int(t_end / h)
for k in range(num_steps):
    yz = dopri5(yz, N, hx, h, t, ac1ddf)

# 结束计时
elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")


np.savetxt('ac1dyz.txt', yz, fmt='%.15e')  # 15位科学计数法

