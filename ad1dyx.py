import numpy as np
from rkce import rkce
N = 200
t_end = 0.5
hx = 1/N
x = np.linspace(hx, 1, N)

a = 0.01
d = 0.1


U0x = np.sin(2*np.pi*x)


lamda = np.zeros(N, dtype=complex)
phi = np.zeros((N, N), dtype=complex)
for k in range(N):

    k_float = k+1
    lamda[k] = ((2*d)/hx**2)*(np.cos(2*np.pi*k_float*hx)-1) - ((1j*a)/hx)*np.sin(2*np.pi*k_float*hx)
    phi[:, k] = np.exp(2*np.pi*1j*k_float*x)


zk = (1/N) * np.conj(phi).T @ U0x


yz = phi @ (zk * np.exp(t_end * lamda))


y0 = phi @ zk


tol = 1e-2
j1 = 1
linear = 1


def ad1ddf(Y, N, hx, t=None):
    """
    Right-hand side of the ODE system.
    Discretization of advection-diffusion equation:
        u_t = d * u_xx + a * u_x
    with periodic boundary conditions.
    """
    a = 0.01
    d = 0.1
    F = np.zeros_like(Y)

    for i in range(N):
        if i == 0:
            F[i] = (d / hx ** 2) * (Y[N - 1] - 2 * Y[i] + Y[i + 1]) + (a / (2 * hx)) * (Y[N - 1] - Y[i + 1])
        elif i == N - 1:
            F[i] = (d / hx ** 2) * (Y[i - 1] - 2 * Y[i] + Y[0]) + (a / (2 * hx)) * (Y[i - 1] - Y[0])
        else:
            F[i] = (d / hx ** 2) * (Y[i - 1] - 2 * Y[i] + Y[i + 1]) + (a / (2 * hx)) * (Y[i - 1] - Y[i + 1])
    return F

# 调用自适应RKC
nfe, err1, smaxz, nac, nre = rkce(y0, N, hx, j1, d, 0, t_end, tol, tol, yz, linear,1, df=ad1ddf)

print("nfe =", nfe)
print("err1 =", err1)
print("smaxz =", smaxz)
print("nac =", nac)
print("nre =", nre)