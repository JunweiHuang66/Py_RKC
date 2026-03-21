import numpy as np

from rkce import rkce


# ------------------------------

# ------------------------------
def cpus1ddf(Y, N, hx , t=None):
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
    a = 1e-4
    d = 1.0
    F = np.zeros(3 * N)

    for j in range(N):
        # 当前点的索引
        u_idx = j
        v_idx = N + j
        w_idx = 2 * N + j


        if j == 0:
            diff_u = (Y[N-1] - 2*Y[u_idx] + Y[1])
        elif j == N-1:
            diff_u = (Y[N-2] - 2*Y[u_idx] + Y[0])
        else:
            diff_u = (Y[u_idx-1] - 2*Y[u_idx] + Y[u_idx+1])

        F[u_idx] = (d / hx ** 2) * diff_u - (1.0 / a) * (Y[u_idx] ** 3 + Y[v_idx] * Y[u_idx] + Y[w_idx])


        if j == 0:
            diff_v = (Y[2*N-1] - 2*Y[v_idx] + Y[v_idx+1])
        elif j == N-1:
            diff_v = (Y[v_idx-1] - 2*Y[v_idx] + Y[N])
        else:
            diff_v = (Y[v_idx-1] - 2*Y[v_idx] + Y[v_idx+1])

        term2 = 0.07 * (Y[u_idx] - 0.7) * (Y[u_idx] - 1.3) / ((Y[u_idx] - 0.7)*(Y[u_idx] - 1.3) + 0.1)
        F[v_idx] = (d / hx ** 2) * diff_v + Y[w_idx] + term2


        if j == 0:
            diff_w = (Y[3*N-1] - 2*Y[w_idx] + Y[w_idx+1])
        elif j == N-1:
            diff_w = (Y[w_idx-1] - 2*Y[w_idx] + Y[2*N])
        else:
            diff_w = (Y[w_idx-1] - 2*Y[w_idx] + Y[w_idx+1])

        term3 = 0.035 * (Y[u_idx] - 0.7) * (Y[u_idx] - 1.3) / ((Y[u_idx] - 0.7)*(Y[u_idx] - 1.3) + 0.1)
        F[w_idx] = (d / hx ** 2) * diff_w + (1 - Y[v_idx] ** 2) * Y[w_idx] - Y[v_idx] - 0.4 * Y[u_idx] + term3

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


t = 0
d = 1
j1 = 1
tol = 1e-4
linear = 0
yz = np.loadtxt('cpus1dyz.txt')



nfe, err1, smaxz, nac, nre = rkce(y0, N, hx, j1, d, 0, t_end, tol, tol, yz, linear,1, df=cpus1ddf)


print("nfe =", nfe)
print("err1 =", err1)
print("smaxz =", smaxz)
print("nac =", nac)
print("nre =", nre)
