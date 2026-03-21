import numpy as np
import time
from dopri5 import dopri5
# ------------------------------
# 右端项函数 df
# ------------------------------
def diff1ddf(Y, N, hx, t=None):
    """
    Compute the right-hand side for the 3-component system.

    Parameters
    ----------
    Y : ndarray of shape (3*N)
        Current solution vector (concatenated: u, v, w).
    N : int
        Number of spatial points per component.
    hx : float
        Spatial step size.
    t : float, optional
        Current time (unused, but kept for compatibility).

    Returns
    -------
    F : ndarray of shape (3*N)
        Right-hand side vector.
    """
    # Constants
    a = 10.0 / (25.0 * np.pi ** 2)
    b = 1.0 / (25.0 * np.pi ** 2)
    r = 1.0 / (4.0 * np.pi ** 2)

    F = np.zeros(3 * N)

    for j in range(N):
        u_idx = j
        v_idx = N + j
        w_idx = 2 * N + j


        if j == 0:

            diff_u = (0 - 2 * Y[u_idx] + Y[u_idx + 1])
            F[u_idx] = -0.04 * Y[u_idx] + 1e4 * Y[v_idx] * Y[w_idx] + a / hx ** 2 * diff_u


            diff_v = (0 - 2 * Y[v_idx] + Y[v_idx + 1])
            F[v_idx] = 0.04 * Y[u_idx] - 3e7 * Y[v_idx] ** 2 - 1e4 * Y[v_idx] * Y[w_idx] + b / hx ** 2 * diff_v


            diff_w = (1 - 2 * Y[w_idx] + Y[w_idx + 1])
            F[w_idx] = 3e7 * Y[v_idx] ** 2 + r / hx ** 2 * diff_w

        elif j == N - 1:

            diff_u = (Y[u_idx - 1] - 2 * Y[u_idx] + 0)
            F[u_idx] = -0.04 * Y[u_idx] + 1e4 * Y[v_idx] * Y[w_idx] + a / hx ** 2 * diff_u


            diff_v = (Y[v_idx - 1] - 2 * Y[v_idx] + 0)
            F[v_idx] = 0.04 * Y[u_idx] - 3e7 * Y[v_idx] ** 2 - 1e4 * Y[v_idx] * Y[w_idx] + b / hx ** 2 * diff_v


            diff_w = (Y[w_idx - 1] - 2 * Y[w_idx] + 1)
            F[w_idx] = 3e7 * Y[v_idx] ** 2 + r / hx ** 2 * diff_w

        else:
            diff_u = (Y[u_idx - 1] - 2 * Y[u_idx] + Y[u_idx + 1])
            F[u_idx] = -0.04 * Y[u_idx] + 1e4 * Y[v_idx] * Y[w_idx] + a / hx ** 2 * diff_u


            diff_v = (Y[v_idx - 1] - 2 * Y[v_idx] + Y[v_idx + 1])
            F[v_idx] = 0.04 * Y[u_idx] - 3e7 * Y[v_idx] ** 2 - 1e4 * Y[v_idx] * Y[w_idx] + b / hx ** 2 * diff_v


            diff_w = (Y[w_idx - 1] - 2 * Y[w_idx] + Y[w_idx + 1])
            F[w_idx] = 3e7 * Y[v_idx] ** 2 + r / hx ** 2 * diff_w

    return F


t_end = 1.0
N = 499
hx = 1.0 / (N + 1)

# 生成网格点 x
x = np.zeros(N)
y0 = np.zeros(3 * N)

for j in range(N):
    x[j] = hx * (j + 1)        # MATLAB: j*hx, j=1..N
    y0[j] = np.sin(np.pi * x[j])
    y0[N + j] = 0.0
    y0[2 * N + j] = 1.0 - np.sin(np.pi * x[j])



h = t_end * 2.0 ** (-18)
t = 0

yz = y0.copy()

start_time = time.time()

num_steps = int(t_end / h)
for k in range(num_steps):
    yz = dopri5(yz, N, hx, h, t, diff1ddf)

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

np.savetxt('diff1dyz.txt', yz, fmt='%.15e')