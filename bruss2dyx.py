import numpy as np
from rkce import rkce


def g(tn, N, hx):
    Nsq = N * N
    G = np.zeros(Nsq)
    if tn >= 1.1:
        for j in range(N):
            for j1 in range(1, N + 1):
                x = j1 * hx
                y = (j + 1) * hx
                if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.01:
                    idx = j * N + (j1 - 1)
                    G[idx] = 5.0
    return G


def bruss2ddf(Y, N, hx, tn):
    """
    Right-hand side for a 2-component reaction-diffusion system with periodic boundaries.

    Parameters
    ----------
    Y : ndarray of shape (2*N^2)
        Current solution vector: first N^2 entries for component 1,
        next N^2 entries for component 2.
    N : int
        Number of grid points in each spatial dimension.
    hx : float
        Spatial step size (same in both dimensions).
    tn : float
        Current time (used for the forcing term).

    Returns
    -------
    F : ndarray of shape (2*N^2)
        Right-hand side vector.
    """
    v1 = 0.2
    Nsq = N * N
    F1 = np.zeros(Nsq)
    F2 = np.zeros(Nsq)

    # Pre-compute forcing term G (depends only on tn, N, hx)
    G = g(tn, N, hx)

    for j in range(N):  # j = 0..N-1 (y-direction index)
        for j1 in range(1, N + 1):  # j1 = 1..N (x-direction index)
            idx = j * N + (j1 - 1)  # linear index for current point

            # --- Neighbors in x-direction (periodic) ---
            if j1 == 1:  # left boundary
                x_left = Y[j * N + (N - 1)]  # Y(j*N + N)   (last point in the same row)
                x_right = Y[j * N + j1]  # Y(j*N + j1+1) -> j1=1 -> index 1
            elif j1 == N:  # right boundary
                x_left = Y[j * N + (j1 - 2)]  # Y(j*N + j1-1)
                x_right = Y[j * N + 0]  # Y(j*N + 1)
            else:
                x_left = Y[j * N + (j1 - 2)]
                x_right = Y[j * N + j1]

            # --- Neighbors in y-direction (periodic) ---
            j_prev = (j - 1) % N
            j_next = (j + 1) % N
            y_prev = Y[j_prev * N + (j1 - 1)]
            y_next = Y[j_next * N + (j1 - 1)]
            y_center = Y[idx]

            # --- Diffusion term (same for both components) ---
            diff_x = (x_left - 2 * y_center + x_right) / (hx * hx)
            diff_y = (y_prev - 2 * y_center + y_next) / (hx * hx)
            lap = v1 * (diff_x + diff_y)

            # --- Component 1 (F1) ---
            F1[idx] = lap + 1 - 4.4 * Y[idx] + (Y[idx] ** 2) * Y[Nsq + idx] + G[idx]

            # --- Component 2 (F2) ---
            # For component 2, the indexing in Y is offset by Nsq
            y2_center = Y[Nsq + idx]
            # x-direction neighbors for component 2
            if j1 == 1:
                x2_left = Y[Nsq + j * N + (N - 1)]
                x2_right = Y[Nsq + j * N + j1]
            elif j1 == N:
                x2_left = Y[Nsq + j * N + (j1 - 2)]
                x2_right = Y[Nsq + j * N + 0]
            else:
                x2_left = Y[Nsq + j * N + (j1 - 2)]
                x2_right = Y[Nsq + j * N + j1]
            # y-direction neighbors for component 2
            y2_prev = Y[Nsq + j_prev * N + (j1 - 1)]
            y2_next = Y[Nsq + j_next * N + (j1 - 1)]
            diff_x2 = (x2_left - 2 * y2_center + x2_right) / (hx * hx)
            diff_y2 = (y2_prev - 2 * y2_center + y2_next) / (hx * hx)
            lap2 = v1 * (diff_x2 + diff_y2)

            F2[idx] = lap2 + 3.4 * Y[idx] - (Y[idx] ** 2) * Y[Nsq + idx]

    # Concatenate the two components
    F = np.concatenate([F1, F2])
    return F




N = 128
hx = 1.0 / N
t_end = 11.5

y0 = np.zeros(2 * N * N)

for j in range(N):
    y = (j + 1) * hx
    for i in range(N):
        y0[j * N + i] = 22 * y * (1 - y) ** 1.5

for i in range(N):
    x = (i + 1) * hx
    for j in range(N):
        y0[N * N + j * N + i] = 27 * x * (1 - x) ** 1.5

tn = 0
d = 0.2
j1 = 1
tol = 1e-3
linear = 1
yz = np.loadtxt('bruss2dyz.txt')

nfe, err1, smaxz, nac, nre = rkce(y0, N, hx, j1, d, 0, t_end, tol, tol, yz, linear, 2, df=bruss2ddf)

print("nfe =", nfe)
print("err1 =", err1)
print("smaxz =", smaxz)
print("nac =", nac)
print("nre =", nre)
