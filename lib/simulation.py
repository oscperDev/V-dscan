import numpy as np
from scipy.fft import fft, ifft, ifftshift


def simulateDSCAN(w, insertion, n, Ew):
    """Function to simulate a scalar d-scan trace:
    Inputs
    w: frequency axis (in rad/fs)
    insertion: glass thickness axis (in mm)
    n: the refractive index of the material to perform d-scan
    Ew: the electric field in the frequency domain

    Output
    trace_sim: simulated d-scan trace normalized to its maximum value"""

    N = len(Ew)
    c = 3e2

    prop_phase = np.outer(insertion*1e6, n*w/c)

    Ew_prop = np.multiply(Ew[np.newaxis, :], np.exp(-1j * prop_phase))
    Et_prop = ifftshift(ifft(Ew_prop, N, 1), 1)
    Et_SHG = Et_prop ** 2
    Ew_SHG = fft(ifftshift(Et_SHG, 1), N, 1)
    trace_sim = np.abs(Ew_SHG) ** 2 / np.max(np.abs(Ew_SHG) ** 2)

    return trace_sim