import numpy as np
from scipy.interpolate import interp1d

def readSpectrum(filePath, w):
    """ Function to read a spectrum from a file
    Inputs
    filePath: complete path to the spectrum
    w: tre frequency axis to interpolate the spectrum into

    It returns an array of complex values containing in the electric field in the frequency axis domain"""

    spectrum = np.loadtxt(filePath)
    spectrum[:, 0] = spectrum[:, 0] * 1e9

    modAxis = np.divide(2 * np.pi * 3e2, spectrum[:, 0], where=spectrum[:, 0] != 0)
    freqSpec_f = interp1d(modAxis, spectrum[:, 1] * (spectrum[:, 0]**2/(2*np.pi*3e2)), kind='cubic',
                          bounds_error=False,
                          fill_value=0)
    freqSpec = np.abs(freqSpec_f(w))
    freqSpec = freqSpec/np.max(freqSpec)
    freqPhase_f = interp1d(modAxis, spectrum[:, 2], kind='cubic', bounds_error=False, fill_value=0)
    freqPhase = freqPhase_f(w)

    return np.sqrt(freqSpec)*np.exp(1j*freqPhase)

def readTrace(tracePath, landaPath, w):
    """ Function to read a spectrum from a file
    Inputs
    filePath: complete path to the spectrum
    w: tre frequency axis to interpolate the spectrum into

    It returns an array of complex values containing in the electric field in the frequency axis domain"""
    trace = np.loadtxt(tracePath)
    landa = np.loadtxt(landaPath)*1e9

    modAxis = np.divide(2 * np.pi * 3e2, landa, where=landa != 0)
    freqTrace_f = interp1d(modAxis, trace * (landa ** 2 / (2 * np.pi * 3e2)), kind='cubic',
                          bounds_error=False,
                          fill_value=0)
    freqTrace = np.abs(freqTrace_f(w))
    freqTrace = freqTrace / np.max(freqTrace)

    return freqTrace