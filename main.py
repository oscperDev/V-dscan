import random

import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.optimize import minimize

from lib.files import readSpectrum

c = 3e2  # nm/fs



def errorG(x, w, w0, I_x, I_y, prop_phase, angles, trace_exp, lims, N_terms):
    #
    # phase_x = (1/2)*x[0]*(w-w0)**2 + (1/6)*x[1]*(w-w0)**3 + (1/24)*x[2]*(w-w0)**4 + (1/120)*x[3]*(w-w0)**5
    # phase_y = (x[4] + x[5]*(w-w0) + (1/2)*x[6]*(w-w0)**2 + (1/6)*x[7]*(w-w0)**3 + (1/24)*x[8]*(w-w0)**4 +
    #            (1/120)*x[9]*(w-w0)**5)

    phase_x = np.zeros(len(trace_exp[0, :]))
    phase_y = np.zeros(len(trace_exp[0, :]))

    for i in range(N_terms):
        phase_x = phase_x + x[i] * np.cos(2*w*i*np.pi/2.5) + x[N_terms+i]*np.sin(2*w*i*np.pi/2.5)
        phase_y = phase_y + x[2*N_terms+i]*np.cos(2*w*i*np.pi/2.5) + x[3*N_terms+i] * np.sin(2*w*i*np.pi/2.5)

    phase_y = phase_y + x[40] + x[41]*(w-w0)

    Ew = np.array([np.sqrt(I_x)*np.exp(1j*phase_x), np.sqrt(I_y)*np.exp(1j*phase_y)])
    Ew_prop = np.multiply(Ew[:, np.newaxis, :], np.exp(-1j * prop_phase))
    Et_prop = ifftshift(ifft(Ew_prop, N, 2), 2)
    Et_proy = Et_prop[0, :, :] * np.cos(angles) + Et_prop[1, :, :] * np.sin(angles)
    Et_SHG = Et_proy ** 2
    Ew_SHG = fft(ifftshift(Et_SHG, 1), N, 1)
    trace_sim = np.abs(Ew_SHG)**2 / np.max(np.abs(Ew_SHG)**2)

    diff = np.sum(np.sum((trace_sim[:, lims[0]:lims[1]] - trace_exp[:, lims[0]:lims[1]])**2))
    G = np.sqrt((1/(trace_exp.shape[0]*(lims[1]-lims[0])))*diff)

    return G




# Definition of the temporal and frequency axes
N = 2**11
dt = 1                       # fs
tmax = dt*N/2                   # fs
t = np.arange(-tmax, tmax, dt)  # fs
wmax = 2*np.pi/dt               # rad/fs
dw = wmax/N                     # rad/fs
w = np.arange(0, wmax, dw)      # rad/fs


Ew_x = readSpectrum('Data\\Espectro_reconstruido.txt', w)
Ew_y = readSpectrum('Data\\Espectro_reconstruido.txt', w)

delta = 0
tau = 3
dephasing = w*tau + delta

Ew = np.array([Ew_x, Ew_y*np.exp(1j*dephasing)])


x = np.arange(0.3, 5, 0.01)  # um
n = (1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**0.5
w_no = 2*np.pi*c*1e-3/x
nw_f = interp1d(w_no, n, bounds_error=False, fill_value=1)
nw = nw_f(w)


motor_step = 0.125  # mm
wedge_angle = 8     # degree
insertion_step = 0.125*2*np.tan(wedge_angle*np.pi/180)
Nz = 180
maxInsertion = (Nz/2)*insertion_step


insertion = np.arange(-maxInsertion, maxInsertion, insertion_step)
prop_phase = np.outer(insertion*1e6, nw*w/c)
Ew_prop = np.multiply(Ew[:, np.newaxis, :], np.exp(-1j*prop_phase))


Et_prop = ifftshift(ifft(Ew_prop, N, 2), 2)

angles = np.reshape(np.arange(0, 360, 2)*np.pi/180, (Nz, 1))

Et_proy = Et_prop[0, :, :]*np.cos(angles) + Et_prop[1, :, :]*np.sin(angles)
Et_SHG = Et_proy**2
Ew_SHG = fft(ifftshift(Et_SHG, 1), N, 1)
trace_exp = np.abs(Ew_SHG)**2 / np.max(np.abs(Ew_SHG)**2)

w_izq = 3.8
w_der = 5.4

lims = []
lims.append(np.where(np.abs(w-w_izq) == np.min(np.abs(w-w_izq)))[0][0])
lims.append(np.where(np.abs(w-w_der) == np.min(np.abs(w-w_der)))[0][0])


plt.figure()
plt.pcolormesh(w, angles, trace_exp, cmap='turbo')
ax0 = plt.gca()
ax0.set_xlim(w[lims])
plt.show()

N_terms = 10

x_ini = np.random.rand(4*N_terms+2)*0.001

res = minimize(errorG, x_ini, method='BFGS',
               args=(w, 2.4, np.abs(Ew_x)**2, np.abs(Ew_y)**2, prop_phase, angles, trace_exp, lims, N_terms),
               options={'disp': True})


x = res.x
w0 = 2.4

phase_x = np.zeros(len(trace_exp[0, :]))
phase_y = np.zeros(len(trace_exp[0, :]))

for i in range(N_terms):
    phase_x = phase_x + x[i] * np.cos(2*w*i*np.pi/2.5) + x[N_terms+i]*np.sin(2*w*i*np.pi/2.5)
    phase_y = phase_y + x[2*N_terms+i]*np.cos(2*w*i*np.pi/2.5) + x[3*N_terms+i]*np.sin(2*w*i*np.pi/2.5)

    phase_y = phase_y + x[40] + x[41]*(w-w0)

# phase_x = (1/2)*x[0]*(w-w0)**2 + (1/6)*x[1]*(w-w0)**3 + (1/24)*x[2]*(w-w0)**4 + (1/120)*x[3]*(w-w0)**5
# phase_y = (x[4] + x[5]*(w-w0) + (1/2)*x[6]*(w-w0)**2 + (1/6)*x[7]*(w-w0)**3 + (1/24)*x[8]*(w-w0)**4 +
#            (1/120)*x[9]*(w-w0)**5)

Ew_ret = np.array([np.abs(Ew_x)*np.exp(1j*phase_x), np.abs(Ew_y)*np.exp(1j*phase_y)])
Ew_prop = np.multiply(Ew_ret[:, np.newaxis, :], np.exp(-1j * prop_phase))
Et_prop = ifftshift(ifft(Ew_prop, N, 2), 2)
Et_proy = Et_prop[0, :, :] * np.cos(angles) + Et_prop[1, :, :] * np.sin(angles)
Et_SHG = Et_proy ** 2
Ew_SHG = fft(ifftshift(Et_SHG, 1), N, 1)
trace_sim = np.abs(Ew_SHG)**2 / np.max(np.abs(Ew_SHG)**2)

plt.figure()
plt.pcolormesh(w, angles, trace_sim, cmap='turbo')
ax = plt.gca()
ax.set_xlim(w[lims])
plt.show()


plt.figure()
plt.plot(w, np.abs(Ew_x)**2, 'r')
ax = plt.gca()
ax1 = plt.twinx(ax)
ax1.plot(w, np.unwrap(np.angle(Ew[1])) - np.unwrap(np.angle(Ew[0])), 'k')
ax1.plot(w, phase_y - phase_x, 'b')
ax.set_xlim([1.8, 2.9])
ax1.set_ylim([-10, 5])
plt.show()

Et = ifftshift(ifft(Ew))
Et_ret = ifftshift(ifft(Ew_ret))

plt.figure()
plt.plot(t, np.abs(Et[0, :])**2, 'r')
plt.plot(t, np.abs(Et[1, :])**2, ':r')
ax = plt.gca()
ax1 = plt.twinx(ax)
ax1.plot(t, np.abs(Et_ret[0, :])**2, 'b')
ax1.plot(t, np.abs(Et_ret[1, :])**2, ':b')
ax.set_xlim([-60, 120])
plt.show()
