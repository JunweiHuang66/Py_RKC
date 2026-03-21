import numpy as np
import time
import matplotlib.pyplot as plt


#    Parameters:
#    y  : current state vector
#   n  : number of grid points (used in df)
#   hx  : spatial step size (used in df)
#   h   : time step size
#   tn  : current time

def dopri5(y, n, hx, h, tn, df):


    K0 = h * df(y, n, hx, tn)  # K1
    Y0 = y + (1 / 5) * K0
    K1 = h * df(Y0, n, hx, tn + (1 / 5) * h)  # K2
    Y1 = y + (3 / 40) * K0 + (9 / 40) * K1
    K2 = h * df(Y1, n, hx, tn + (3 / 10) * h)  # K3
    Y2 = y + (44 / 45) * K0 - (56 / 15) * K1 + (32 / 9) * K2
    K3 = h * df(Y2, n, hx, tn + (4 / 5) * h)  # K4
    Y3 = y + (19372 / 6561) * K0 - (25360 / 2187) * K1 + (64448 / 6561) * K2 - (212 / 729) * K3
    K4 = h * df(Y3, n, hx, tn + (8 / 9) * h)  # K5
    Y4 = y + (9017 / 3168) * K0 - (355 / 33) * K1 + (46732 / 5247) * K2 + (49 / 176) * K3 - (5103 / 18656) * K4
    K5 = h * df(Y4, n, hx, tn + h)  # K6
    y_end = y + (35 / 384) * K0 + (500 / 1113) * K2 + (125 / 192) * K3 - (2187 / 6784) * K4 + (11 / 84) * K5
    return y_end
