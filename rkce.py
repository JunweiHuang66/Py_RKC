import numpy as np
import time
import matplotlib.pyplot as plt


 #    Author: Junwei Huang
 #    School of Mathematics and Computational Science & Hunan Key Laboratory for Computation and
 #    Simulation in Science and Engineering, Xiangtan University, Hunan 411105, China.
 #    e-mail: junweihuang@smail.xtu.edu.cn
 #    Version of March 2026
 #
 #
 #
 # Adaptive RKC (Runge-Kutta-ChebYeshev) method for stiff ODEs resulting from PDE spatial discretization.
 # This function integrates the system from t0 to t_end with adaptive time-stepping.
 #
 # Inputs:
 #   y0      - Initial condition vector (length N)
 #   n       - Number of spatial grid points
 #   hx      - Spatial step size
 #   j1      - Switch for error estimation method (1,2,3) used in rkc2
 #   d       - Diffusion coefficient (used in df, but also passed but not directly used here)
 #   t0      - Initial time
 #   t_end   - Final time
 #   atol    - Absolute tolerance for error control
 #   rtol    - Relative tolerance for error control
 #   yz      - Reference solution (exact or high-accuracy) for final error computation
 #   linear  - linear  Switch: The stiffness mainly arises from the diffusion term, which is set to 1, and 0 for all other cases.
 #
 # Outputs:
 #  nfe     - Total number of function evaluations (calls to df)
 #  err1    - Final global error (L2 norm of difference between computed solution and yz)
 #  smaxz   - Maximum stage number (s) used throughout the integration
 #  nac     - Number of accepted steps
 #  nre     - Number of rejected steps
















def err(Y, Y1, Y2, atol, rtol):
    """
    Weighted RMS error estimate.
    """
    ln = len(Y)
    wt = atol + rtol * np.maximum(np.abs(Y), np.abs(Y1))
    E = np.sqrt(np.sum((Y2 / wt) ** 2) / ln)
    return E


def rodf(Y, N, hx,df):
    """
    Estimate spectral radius.
    """
    e = 1e-8
    Rv = Y.copy()
    ln = len(Y)
    for j in range(ln):
        if Y[j] == 0:
            Rv[j] = 0.5 * e
        else:
            Rv[j] = Y[j] * (1 + 0.5 * e)

    e = max(e, e * np.linalg.norm(Rv))
    Rv1 = Y.copy()
    f1 = df(Rv1, N, hx)
    f2 = df(Rv, N, hx)
    # Perturb in direction of f1-f2
    denom = np.linalg.norm(f1 - f2)
    if denom == 0:
        denom = 1.0
    Rv1 = Rv + e / denom * (f1 - f2)
    f1 = df(Rv1, N, hx)
    R = (1.0 / e) * np.linalg.norm(f1 - f2)
    Rr = R
    fg = Rr
    fg1 = 0
    while fg > 1e-4 * R and fg1 < 40:
        f1 = df(Rv1, N, hx)
        denom = np.linalg.norm(f1 - f2)
        if denom == 0:
            denom = 1.0
        Rv1 = Rv + e / denom * (f1 - f2)
        f1 = df(Rv1, N, hx)
        R = (1.0 / e) * np.linalg.norm(f1 - f2)
        fg = abs(R - Rr)
        fg1 += 1
        Rr = R
    if fg1 == 40:
        R = 1.1 * R
    return R, fg1


def rkc2(Y, h, t, s, N, hx, atol, rtol, method, df):
    """
    One step of the RKC method with embedded error estimator.

    Parameters
    ----------
    Y : ndarray
        Current solution vector (length n).
    h : float
        Step size.
    t : float
        Current time.
    s : int
        Number of stages (s>=2).
    N : int
        Number of spatial grid points.
    hx : float
        Spatial step size.
    atol, rtol : float
        Absolute and relative tolerances for error control.
    method : int
        Error estimation method (1, 2, or 3).
    df : function
        Function computing the right-hand side.
    err : function
        Function computing weighted RMS error: err(Y, Y1, Y2, atol, rtol) -> float.

    Returns
    -------
    yerr : float
        Estimated local error (scalar).
    y : ndarray
        New solution after one step.
    """
    # Pre-allocate coefficient arrays (size s+1, indices 0..s)
    u1 = np.zeros(s + 1)
    u2 = np.zeros(s + 1)
    u3 = np.zeros(s + 1)
    u4 = np.zeros(s + 1)
    Rc = np.zeros(s + 1)
    b = np.zeros(s + 1)

    eta = 2.0 / 13.0
    omiga0 = 1.0 + eta / (s ** 2)

    # Compute omiga1 using hyperbolic functions
    acosh_omiga0 = np.arccosh(omiga0)
    sinh_s = np.sinh(s * acosh_omiga0)
    cosh_s = np.cosh(s * acosh_omiga0)

    t1 = (s * sinh_s) / np.sqrt((omiga0 - 1) * (omiga0 + 1))
    t2 = (s ** 2 * cosh_s) / ((omiga0 - 1) * (omiga0 + 1)) \
         - (s * sinh_s) / (2 * (omiga0 - 1) ** (3.0 / 2.0) * np.sqrt(omiga0 + 1)) \
         - (s * sinh_s) / (2 * np.sqrt(omiga0 - 1) * (omiga0 + 1) ** (3.0 / 2.0))
    omiga1 = t1 / t2

    # Compute coefficients b(j), u1(j), u2(j), u3(j), u4(j), Rc(j) for j = 1..s+1
    for j in range(1, s + 2):  # j from 1 to s+1 inclusive
        idx = j - 1  # Python index (0-based)
        if j == 1:
            # j = 1
            num = ((j + 1) ** 2 * np.cosh((j + 1) * acosh_omiga0)) / (omiga0 ** 2 - 1) \
                  - ((j + 1) * np.sinh((j + 1) * acosh_omiga0)) / (2 * (omiga0 - 1) * np.sqrt(omiga0 ** 2 - 1)) \
                  - ((j + 1) * np.sinh((j + 1) * acosh_omiga0)) / (2 * (omiga0 + 1) * np.sqrt(omiga0 ** 2 - 1))
            den = ((j + 1) * np.sinh((j + 1) * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1)) ** 2
            b[0] = num / den
            u1[0] = b[0] * omiga1
            u2[0] = 0
            u3[0] = 0
            u4[0] = 0
            u2[0] = omiga1 * b[0] * (j * np.sinh(j * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1))
        elif j == s + 1:
            # j = s+1
            num = (j ** 2 * np.cosh(j * acosh_omiga0)) / (omiga0 ** 2 - 1) \
                  - (j * np.sinh(j * acosh_omiga0)) / (2 * (omiga0 - 1) * np.sqrt(omiga0 ** 2 - 1)) \
                  - (j * np.sinh(j * acosh_omiga0)) / (2 * (omiga0 + 1) * np.sqrt(omiga0 ** 2 - 1))
            den = (j * np.sinh(j * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1)) ** 2
            b[s] = num / den
            u1[s] = 2 * omiga1 * b[s] / b[s - 1]
            u2[s] = 2 * omiga0 * b[s] / b[s - 1]
            u3[s] = -b[s] / b[s - 2]
            u4[s] = -(1 - b[s - 1] * np.cosh(s * acosh_omiga0)) * u1[s]
            Rc[s] = omiga1 * b[s] * (j * np.sinh(j * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1))
        else:
            # 2 <= j <= s
            num = (j ** 2 * np.cosh(j * acosh_omiga0)) / (omiga0 ** 2 - 1) \
                  - (j * np.sinh(j * acosh_omiga0)) / (2 * (omiga0 - 1) * np.sqrt(omiga0 ** 2 - 1)) \
                  - (j * np.sinh(j * acosh_omiga0)) / (2 * (omiga0 + 1) * np.sqrt(omiga0 ** 2 - 1))
            den = (j * np.sinh(j * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1)) ** 2
            b[idx] = num / den
            u1[idx] = 2 * omiga1 * b[idx] / b[idx - 1]
            u2[idx] = 2 * omiga0 * b[idx] / b[idx - 1]
            u4[idx] = -(1 - b[idx - 1] * np.cosh((j - 1) * acosh_omiga0)) * u1[idx]
            if j == 2:
                u3[idx] = -b[idx] / b[idx - 1]
            else:
                u3[idx] = -b[idx] / b[idx - 2]
            Rc[idx] = omiga1 * b[idx] * (j * np.sinh(j * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1))

    # Determine stage s1 for error estimation
    if 2 <= s <= 5:
        s1 = s - 1
    elif 6 <= s <= 8:
        s1 = s - 2
    elif 9 <= s <= 10:
        s1 = s - 3
    else:
        s1 = int(np.floor(0.8 * s))

    # Compute  factor (for method 3)
    if s1 >= 1:
        denom = omiga1 * b[s1 - 1] * (s1 * np.sinh(s1 * acosh_omiga0) / np.sqrt(omiga0 ** 2 - 1))
        Rk = 1.0 / denom if denom != 0 else 1.0
    else:
        Rk = 1.0

    # Time stepping: j = 0..s
    K1 = Y.copy()
    f1 = None
    for j in range(0, s + 1):
        if j == 0:
            # K1 already set
            continue
        elif j == 1:
            # Stage 1
            f1 = df(Y, N, hx, t + Rc[0] * h)
            K2 = K1 + h * u1[0] * f1
        else:
            # Stages j >= 2
            f = df(K2, N, hx, t + Rc[j - 1] * h)
            y_new = (h * u1[j - 1] * f
                     + u2[j - 1] * K2
                     + h * u4[j - 1] * f1
                     + u3[j - 1] * K1
                     + (1 - u2[j - 1] - u3[j - 1]) * Y)
            K1, K2 = K2, y_new
        if j == s1:
            Kz = K2.copy()

    y = K2  # final solution
    f2 = df(y, N, hx, t + h)

    # Error estimate according to method
    if method == 1:
        err1 = (1.0 / 15.0) * (12 * (Y - y) + 6 * h * (f1 + f2))
    elif method == 2:
        err1 = y - (Y + h * f2)
    elif method == 3:
        err1 = y - ((1 - Rk) * Y + Rk * Kz)
    else:
        err1 = np.zeros_like(y)

    yerr = err(Y, y, np.abs(err1), atol, rtol)
    return yerr, y


def rkce(y0, N, hx, j1, d, t0, t_end, atol, rtol, yz, linear,df):
    """
    Adaptive RKC method for stiff ODEs.
    Returns: nfe, err1, smaxz, nac, nre
    """
    facmax = 10.0
    facmin = 0.1
    smax = 1000
    smaxz = 0
    nac = 0
    nre = 0

    if linear == 1:
        # Diffusion dominated: estimate spectral radius from diffusion term
        lop1 = 4.0 * d * N ** 2
        nfe = 0
    else:
        R, fg1 = rodf(y0, N, hx,df)
        lop1 = R
        nfe = 2*fg1+3

    hmax = t_end - t0

    if j1 == 1:
        p = 1.0 / 3.0
    else:
        p = 1.0 / 2.0

    # Initial step size selection (same as in original code)
    h = hmax
    if h * lop1 > 1:
        h = 1.0 / lop1
    h = max(h, 10.0 * np.finfo(float).eps * max(abs(t0), abs(t_end - t0)))

    s = max(2, 1 + int(np.floor(np.sqrt(1.54 * h * lop1 + 1.0))))
    smaxz = s
    t = t0
    _, y = rkc2(y0, h, t, s, N, hx, atol, rtol, j1,df)
    nfe += s
    f00 = df(y0, N, hx, t)
    f01 = df(y0 + h * f00, N, hx, t)
    nfe += 2
    err1_vec = h * (f01 - f00)
    E0 = err(y0, y, np.abs(err1_vec), atol, rtol)

    if abs(E0) < 1e-15:
        E0 = 1e-15

    if E0 <= 1:
        if 0.1 * h < (t_end - t0) * np.sqrt(E0):
            h = max(0.1 * h / np.sqrt(E0), 10.0 * np.finfo(float).eps * max(abs(t0), abs(t_end - t0)))
        else:
            h = t_end - t0
    else:
        h = 0.1 * h / np.sqrt(E0)
        h = max(h, 10.0 * np.finfo(float).eps * max(abs(t0), abs(t_end - t0)))

    if h > (t_end - t0):
        h = t_end - t0

    if s > smax:
        s = smax
        h = (s ** 2 - 1) / (1.54 * lop1)
        if h > (t_end - t0):
            h = t_end - t0

    hn = h
    ys = y0.copy()
    tn = t0

    # Main time stepping loop
    while tn < t_end:
        yerr, y = rkc2(ys, h, tn, s, N, hx, atol, rtol, j1,df)
        nfe += s
        E = yerr

        if E <= 1:
            ys = y.copy()
            tn = tn + h
            nac += 1
            if linear==0:
                if nac % 25 == 0:
                    R, fg1 = rodf(ys, N, hx, df)
                    lop1 = R
                nfe += 2 * fg1 + 3
            if nac == 1:
                fac_final = 0.8 * (1.0 / E) ** p
            else:
                fac_final = 0.8 * min((1.0 / E) ** p, (1.0 / E) ** p * (E0 / E) ** p * h / hn)
            faczjz = min(facmax, max(fac_final, facmin))
            hn = h
            E0 = E
            h = h * faczjz
        else:
            if linear==0:
               R, fg1 = rodf(ys, N, hx, df)
               lop1 = R
               nfe += 2 * fg1 + 3
            fac_final = 0.8 * (1.0 / E) ** p
            faczjz = min(facmax, max(fac_final, facmin))
            h = h * faczjz
            nre += 1

        if h > (t_end - tn):
            h = t_end - tn

        s = max(2, 1 + int(np.floor(np.sqrt(1.54 * h * lop1 + 1.0))))
        if s > smax:
            s = smax
            h = (s ** 2 - 1) / (1.54 * lop1)
            if h > (t_end - tn):
                h = t_end - tn

        if s > smaxz:
            smaxz = s

    # Final error against reference solution yz
    le1 = len(ys)
    err1 = np.sqrt(np.sum((abs(ys - yz)) ** 2) / le1)

    if j1 != 3:
        nfe += nre

    return nfe, err1, smaxz, nac, nre


