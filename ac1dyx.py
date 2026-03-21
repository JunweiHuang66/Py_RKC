import numpy as np

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


tol = 1e-4
j1 = 3
linear = 1
def ac1ddf(Y, N, hx,t=None):
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


yz = np.loadtxt('ac1dyz.txt')



nfe, err1, smaxz, nac, nre = rkce(y0, N, hx, j1, d, 0, t_end, tol, tol, yz, linear, 1, df=ac1ddf)


print("nfe =", nfe)
print("err1 =", err1)
print("smaxz =", smaxz)
print("nac =", nac)
print("nre =", nre)