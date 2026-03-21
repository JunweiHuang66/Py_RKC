import numpy as np
from rkce import rkce



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
d = 0.3
N = 150
hx = 1 / N
x = hx * np.arange(1, N + 1)
y0 = 1 + np.sin(2 * np.pi * x)
j1 = 1
tol = 1e-5
linear = 1
yz = np.loadtxt('burger1dyz.txt')



nfe, err1, smaxz, nac, nre = rkce(y0, N, hx, j1, d, 0, t_end, tol, tol, yz, linear,1, df=burger1ddf)


print("nfe =", nfe)
print("err1 =", err1)
print("smaxz =", smaxz)
print("nac =", nac)
print("nre =", nre)





