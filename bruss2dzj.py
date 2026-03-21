import numpy as np
import time
from dopri5 import dopri5


def g(tn, N, hx):
    """

    """
    G = np.zeros(N * N)
    if tn >= 1.1:
        for i in range(N):
            for j in range(N):
                x = (j + 1) * hx
                y = (i + 1) * hx
                if (x - 0.3) ** 2 + (y - 0.6) ** 2 <= 0.01:
                    G[i * N + j] = 5.0
    return G


def bruss2ddf(Y, N, hx, tn):
    """



    """
    v1 = 0.2
    F1 = np.zeros(N * N)  # du/dt
    F2 = np.zeros(N * N)  # dv/dt
    G_vec = g(tn, N, hx)  # 源项

    inv_hx2 = 1.0 / (hx * hx)

    for i in range(N):
        for j in range(N):
            idx = i * N + j

            i_up = (i - 1) % N
            i_down = (i + 1) % N
            j_left = (j - 1) % N
            j_right = (j + 1) % N

            center_u = idx
            left_u = i * N + j_left
            right_u = i * N + j_right
            up_u = i_up * N + j
            down_u = i_down * N + j

            laplace_u_x = (Y[left_u] - 2 * Y[center_u] + Y[right_u]) * inv_hx2
            laplace_u_y = (Y[up_u] - 2 * Y[center_u] + Y[down_u]) * inv_hx2
            laplace_u = v1 * (laplace_u_x + laplace_u_y)

            u_val = Y[center_u]
            v_val = Y[N * N + idx]
            reaction_u = 1 - 4.4 * u_val + (u_val ** 2) * v_val

            source = G_vec[idx]

            F1[idx] = laplace_u + reaction_u + source

            center_v = N * N + idx
            left_v = N * N + (i * N + j_left)
            right_v = N * N + (i * N + j_right)
            up_v = N * N + (i_up * N + j)
            down_v = N * N + (i_down * N + j)

            laplace_v_x = (Y[left_v] - 2 * Y[center_v] + Y[right_v]) * inv_hx2
            laplace_v_y = (Y[up_v] - 2 * Y[center_v] + Y[down_v]) * inv_hx2
            laplace_v = v1 * (laplace_v_x + laplace_v_y)

            reaction_v = 3.4 * u_val - (u_val ** 2) * v_val

            F2[idx] = laplace_v + reaction_v

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

h = t_end * 2.0 ** (-18)
tn = 0

yz = y0.copy()

start_time = time.time()

num_steps = int(t_end / h)
for k in range(num_steps):
    yz = dopri5(yz, N, hx, h, tn, bruss2ddf)
    tn += h

elapsed_time = time.time() - start_time
print(f"Elapsed time: {elapsed_time:.6f} seconds")

np.savetxt('bruss2dyz.txt', yz, fmt='%.15e')
