import numpy as np
from scipy.interpolate import interp1d

def calculateFWHM(t, pulse, N = 2**15):

    t_new = np.arange(t[0],t[-1],(t[-1]-t[0])/(N-1))
    pulse_int_f = interp1d(t, pulse, bounds_error=None, fill_value=0)
    pulse_int = pulse_int_f(t_new)

    pulse_int = np.abs(pulse_int)/np.max(np.abs(pulse_int))

    pos_izq = np.where(np.abs(pulse_int-0.5) == np.min(np.abs(pulse_int[0:int(N/2-1)]-0.5)))[0][0]
    pos_der = np.where(np.abs(pulse_int-0.5) == np.min(np.abs(pulse_int[int(N/2):]-0.5)))[0][0]


    FWHM = t_new[pos_der] - t_new[pos_izq]

    return FWHM