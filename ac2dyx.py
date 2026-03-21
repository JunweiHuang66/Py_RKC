import numpy as np
from rkce import rkce

# ------------------------------
# 右端项函数 df
# ------------------------------
def ac2ddf(Y, N, hx, t=None):
    """
    Compute the right-hand side for the 2D Allen-Cahn type equation.

    Parameters
    ----------
    Y : ndarray of shape (N*N)
        Current solution vector (flattened 2D grid).
    N : int
        Number of grid points in each spatial dimension.
    hx : float
        Spatial step size (same in both dimensions).
    t : float, optional
        Current time (unused, kept for compatibility).

    Returns
    -------
    F : ndarray of shape (N*N)
        Right-hand side vector.
    """
    epslion = 0.015
    d = 0.2
    F = np.zeros(N * N)

    for j in range(N):
        for i in range(N):
            idx = j * N + i

            if i == 0:
                diff_x = Y[j * N + (N - 1)] - 2 * Y[idx] + Y[j * N + (i + 1)]
            elif i == N - 1:
                diff_x = Y[j * N + (i - 1)] - 2 * Y[idx] + Y[j * N + 0]
            else:
                diff_x = Y[j * N + (i - 1)] - 2 * Y[idx] + Y[j * N + (i + 1)]

            j_prev = (j - 1 + N) % N
            j_next = (j + 1) % N
            diff_y = Y[j_prev * N + i] - 2 * Y[idx] + Y[j_next * N + i]

            reaction = (1.0 / epslion ** 2) * (Y[idx] - Y[idx] ** 3)

            F[idx] = (d / hx ** 2) * diff_x + (1.0 / hx ** 2) * diff_y + reaction

    return F


np.set_printoptions(precision=15)

N = 150
hx = 1.0 / N
d = 0.2

epsilion = 0.015
t_end = (np.sqrt(2) * epsilion) / (4 * np.sqrt(d))

x = np.zeros(N)
for j in range(N):
    x[j] = -0.5 + hx * (j + 1)

y0 = np.zeros(N * N)

for j in range(N):
    for i in range(N):
        xx = x[i]
        y0[j * N + i] = 0.5 * (1.0 - np.tanh((np.sqrt(d) * xx) / (2 * np.sqrt(2) * epsilion)))

t = 0
d = 0.2
j1 = 3
tol = 10**(-3)
linear = 0
yz = np.loadtxt('ac2dyz.txt')



nfe, err1, smaxz, nac, nre = rkce(y0, N, hx, j1, d, 0, t_end, tol, tol, yz, linear, 2,df=ac2ddf)


print("nfe =", nfe)
print("err1 =", err1)
print("smaxz =", smaxz)
print("nac =", nac)
print("nre =", nre)
