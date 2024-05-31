import numpy as np
from scipy.fft import fft, ifft, ifftshift
import matplotlib.pyplot as plt
from alive_progress import alive_bar
from tqdm import tqdm

def ptychographicScalar(w, Ew_ini, exp_trace, n, motor_step, position= None, N_motors=2, wedge_angle = 8, N_iters=150, lims=None, force_spec=1):
    """Ptychographic core to retrieve scalar dscan traces. Inputs:
        w: frequency axis
        Ew_ini: initial guess of the electric field in the spectral domain
        exp_trace: experimental trace to be retrieved
        n: refractive index of the material to perform dispersion scan
        motor_step: the distance that the motor moves
        position: the step of the trace in which the spectrum and spectral phase is returned
        N_motors: the number of stepper motors
        wedge_angle: the angle of the wedges
        N_iters: the number of iterations to execute the ptychographic algorithm
        lims: the limits in which the error function is calculated
        force_spec: a number to control when the algorithm forces the retrieved electric field to have the experimental
                    spectrum. The number chosen means every force_spec number of interations the field is forced to have
                    the experimental spectrum. Personal experience: 20-30 works good, 1 is too restrictive."""

    c = 3e2

    # Creation of the glass thickness axis
    insertion_step = motor_step*N_motors*np.tan(wedge_angle*np.pi/180)
    Nz = exp_trace.shape[0]

    marginal = np.sum(exp_trace, axis=1)
    center = np.where(marginal == np.max(marginal))[0][0]

    maxInsertion = (Nz/2)*insertion_step

    offset = (Nz/2-center)*insertion_step
    insertion = np.arange(-maxInsertion + offset, maxInsertion + offset, insertion_step)

    prop_phase = np.outer(insertion*1e6, n*w/c)

    N = len(Ew_ini)

    # weights = np.abs(range(Nz) - center)
    # weights[center] = 1
    # print(weights**0.2)
    Ew = Ew_ini

    G_min = 100
    G_prev = 0
    counter = 0
    threshold = 1000

    for i in tqdm(range(N_iters), ncols=100, desc='Retrieval process'):

        # norma = np.max(np.abs(Ew))/np.max(np.abs(Ew_ini))
        # Ew = norma* np.divide(np.abs(Ew_ini) * Ew, np.abs(Ew), where=np.abs(Ew) != 0)

        Ew_prop = np.multiply(Ew[np.newaxis, :], np.exp(-1j*prop_phase))
        Et_prop = ifftshift(ifft(Ew_prop, N, 1), 1)
        Et_SHG = Et_prop ** 2
        Ew_SHG = fft(ifftshift(Et_SHG, 1), N, 1)
        Ew_SHG = Ew_SHG / np.max(np.abs(Ew_SHG))
        sim_trace = np.abs(Ew_SHG)**2
        mu = spectralResponse(exp_trace, sim_trace)

        G = errorG(exp_trace, mu*sim_trace, lims)

        if G<G_min:
            G_min = G
            iter = i
            Ew_ret = Ew
            # Ew_ret = np.divide(np.abs(Ew_ini) * Ew, np.abs(Ew), where=np.abs(Ew) != 0)
            ret_trace = mu*sim_trace/np.max(mu*sim_trace)

        alpha = np.random.rand(1)*2
        cal_exp_trace = np.divide(exp_trace, mu, where=mu!=0)
        norma2 = np.max(np.abs(Ew_SHG)/np.max(np.abs(np.sqrt(cal_exp_trace))))
        Ew_new_SHG = norma2*np.divide(np.sqrt(cal_exp_trace)*Ew_SHG, np.abs(Ew_SHG), where=np.abs(Ew_SHG) != 0)
        Et_new_SHG = ifftshift(ifft(Ew_new_SHG, N, 1), 1)
        Et_new_prop = (1/2)*(2*Et_prop+alpha*(np.conj(Et_prop)/np.max(np.abs(Et_prop))**2)*(Et_new_SHG-Et_SHG))
        Ew_new_prop = fft(ifftshift(Et_new_prop, 1), N, 1)
        Ew_new_array = Ew_new_prop*np.exp(1j * prop_phase)

        Ew_new_array = Ew_new_array
        Ew_new = np.average(Ew_new_array, axis=0)

        if i%force_spec == 0:
            # Ew = np.abs(Ew_ini)*np.exp(1j*np.angle(Ew_new))
            norma = np.max(np.abs(Ew)) / np.max(np.abs(Ew_ini))
            Ew = norma*np.divide(np.abs(Ew_ini) * Ew_new, np.abs(Ew_new), where=np.abs(Ew_new) != 0)
        else:
            Ew = Ew_new

        # Ew = Ew_new

        # Method to introduce a perturbation in the electric field to force new updates
        if counter > threshold and G > G_prev:
            Ew = Ew + 5*np.random.rand(len(Ew))
            counter = 0
        G_prev = G
        counter = counter + 1

    if position:
        Ew_prop_ret = np.multiply(Ew_ret[np.newaxis, :], np.exp(-1j*prop_phase))
        Ew_ret_new = Ew_prop_ret[position, :]
        Et_ret = ifftshift(ifft(Ew_ret_new, N))
        posMax = np.where(np.abs(Et_ret) == np.max(np.abs(Et_ret)))[0][0]
        Et_ret_new = np.roll(Et_ret, -posMax)
        Ew_ret = fft(ifftshift(Et_ret_new), N)

    return Ew_ret, ret_trace, insertion, G_min, iter, mu


def errorG(expT, simT, lims=None):

    if lims:
        exp = expT[:, lims[0]:lims[1]]
        sim = simT[:, lims[0]:lims[1]]
    else:
        exp = expT
        sim = simT

    exp = np.abs(exp) / np.max(np.abs(exp))
    sim = np.abs(sim) / np.max(np.abs(sim))

    return np.sqrt((1/(exp.shape[0]*exp.shape[1])) * np.sum(np.sum((exp-sim)**2)))

def spectralResponse(expT, simT):

    num = np.sum(expT*simT, axis=0)
    den = np.sum(simT**2, axis=0)

    return np.divide(num, den, where=den != 0)


