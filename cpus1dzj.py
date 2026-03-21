import numpy as np
import time
from dopri5 import dopri5


# ------------------------------
# 右端项函数 (与 MATLAB 的 df 一致)
# ------------------------------
def cpus1ddf(Y, N, hx , t=None):
    """
    Compute the right-hand side of the ODE system.
    Parameters
    ----------
    Y : ndarray of shape (3*N)
        Current solution vector (concatenated: u, v, w).
    N : int
        Number of spatial points per variable.
    hx : float
        Spatial step size.
    Returns
    -------
    F : ndarray of shape (3*N)
        Right-hand side vector.
    """
    a = 1e-4
    b = 1.0
    F = np.zeros(3 * N)

    for j in range(N):
        idx_u = j
        idx_v = N + j
        idx_w = 2 * N + j

        if j == 0:
            diff_u = (Y[N - 1] - 2 * Y[idx_u] + Y[idx_u + 1])
        elif j == N - 1:
            diff_u = (Y[idx_u - 1] - 2 * Y[idx_u] + Y[0])
        else:
            diff_u = (Y[idx_u - 1] - 2 * Y[idx_u] + Y[idx_u + 1])

        F[idx_u] = (b / hx ** 2) * diff_u - (1.0 / a) * (Y[idx_u] ** 3 + Y[idx_v] * Y[idx_u] + Y[idx_w])

        if j == 0:
            diff_v = (Y[2 * N - 1] - 2 * Y[idx_v] + Y[idx_v + 1])
        elif j == N - 1:
            diff_v = (Y[idx_v - 1] - 2 * Y[idx_v] + Y[N])
        else:
            diff_v = (Y[idx_v - 1] - 2 * Y[idx_v] + Y[idx_v + 1])

        term2 = 0.07 * (Y[idx_u] - 0.7) * (Y[idx_u] - 1.3) / ((Y[idx_u] - 0.7) * (Y[idx_u] - 1.3) + 0.1)
        F[idx_v] = (b / hx ** 2) * diff_v + Y[idx_w] + term2

        if j == 0:
            diff_w = (Y[3 * N - 1] - 2 * Y[idx_w] + Y[idx_w + 1])
        elif j == N - 1:
            diff_w = (Y[idx_w - 1] - 2 * Y[idx_w] + Y[2 * N])
        else:
            diff_w = (Y[idx_w - 1] - 2 * Y[idx_w] + Y[idx_w + 1])

        term3 = 0.035 * (Y[idx_u] - 0.7) * (Y[idx_u] - 1.3) / ((Y[idx_u] - 0.7) * (Y[idx_u] - 1.3) + 0.1)
        F[idx_w] = (b / hx ** 2) * diff_w + (1 - Y[idx_v] ** 2) * Y[idx_w] - Y[idx_v] - 0.4 * Y[idx_u] + term3

    return F


N = 32
hx = 1.0 / N
t_end = 3.1
N1 = 3 * N

y0 = np.zeros(3 * N)

x = hx * np.arange(1, N + 1)

for j in range(N):
    y0[j] = 0.0
    y0[N + j] = -2.0 * np.cos(2 * np.pi * x[j])
    y0[2 * N + j] = 2.0 * np.sin(2 * np.pi * x[j])

h = t_end * 2.0 ** (-18)
t = 0

yz = y0.copy()

start_time = time.time()

num_steps = int(t_end / h)
for k in range(num_steps):
    yz = dopri5(yz, N, hx, h, t, cpus1ddf)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

np.savetxt('cpus1dyz.txt', yz, fmt='%.15e')
