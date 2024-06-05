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


def superGaussian(axis, FWHM, exponent, center=0, amplitude=1):
    """
        This function returns a super Gaussian function intended to create spectral filters.
        :param axis: the time axis in which the electric field is defined
        :type axis: any
        :param FWHM: the full width half maximum of super Gaussian
        :type FWHM: float
        :param exponent: the exponent of the super Gaussian (a Gaussian function has exponent=2)
        :type exponent: int
        :param center: the position of the maximum of the super Gaussian
        :type center: float
        :param amplitude: the maximum value of the super Gaussian function
        :type amplitude: float
        :returns: np.array ontaining a super Gaussian function
        :rtype: ndarray
    """
    c = FWHM/(2*np.log(2))

    return amplitude * np.exp(-(axis - center)**exponent / (2*(c**exponent)))