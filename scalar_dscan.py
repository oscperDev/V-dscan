import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift
from scipy.optimize import minimize

from lib.files import readSpectrum, readTrace
from lib.simulation import simulateDSCAN
from lib.retrieval import ptychographicScalar

import time

c = 3e2  # nm/fs

# Definition of the temporal and frequency axes
N = 2**9
dt = 1                          # fs
tmax = dt*N/2                   # fs
t = np.arange(-tmax, tmax, dt)  # fs
wmax = 2*np.pi/dt               # rad/fs
dw = wmax/N                     # rad/fs
w = np.arange(0, wmax, dw)      # rad/fs


# Definition of the refractive index of the material to perform the dispersion scan
# This is the refractive index of BK7 directly obtained from refractiveindex.info
x = np.arange(0.3, 5, 0.01)  # um
n = (1+1.03961212/(1-0.00600069867/x**2)+0.231792344/(1-0.0200179144/x**2)+1.01046945/(1-103.560653/x**2))**0.5
w_no = 2*np.pi*c*1e-3/x
nw_f = interp1d(w_no, n, bounds_error=False, fill_value=1)
nw = nw_f(w)


# Creation of the glass thickness axis
motor_step = 0.125  # mm
wedge_angle = 8     # degree
insertion_step = motor_step*2*np.tan(wedge_angle*np.pi/180)
Nz = 180
maxInsertion = (Nz/2)*insertion_step
offset = 0          # mm
insertion = np.arange(-maxInsertion+offset, maxInsertion+offset, insertion_step)


# Loading a spectrum and a spectral phase to define an electric field
Ew = readSpectrum('Data\\Espectro_reconstruido_traza.txt', w)


# Simulation of a d-scan trace
trace_sim = simulateDSCAN(w, insertion, nw, Ew)

trace_sim = readTrace('Data\\Traza_experimental.txt', 'Data\\landa_ocean.txt', w)

w_izq = 3.6
w_der = 5.8
w0 = 2.4

lims = []
lims.append(np.where(np.abs(w-w_izq) == np.min(np.abs(w-w_izq)))[0][0])
lims.append(np.where(np.abs(w-w_der) == np.min(np.abs(w-w_der)))[0][0])
posw0 = np.where(np.abs(w-w0) == np.min(np.abs(w-w0)))[0][0]

Ew_ini = np.abs(Ew)*np.exp(1j*np.random.rand(len(Ew)))

start = time.time()
Ew_ret, trace_ret, insertion, G, iteration, mu = ptychographicScalar(w, Ew_ini, trace_sim, nw, motor_step, position=None, lims=lims, N_iters=500)
end = time.time()

print('\nRETRIEVAL RESULTS:\n')
print('Error G =', round(G, 4))
print('Minimum of G achieved on interation =', iteration)
print('Retrieval time =', round(end-start, 0), ' s')

ret_spec = np.abs(Ew_ret)**2/np.max(np.abs(Ew_ret)**2)
ret_phase = np.unwrap(np.angle(Ew_ret), period=np.pi)
ret_phase = ret_phase - ret_phase[posw0]

sim_spec = np.abs(Ew)**2/np.max(np.abs(Ew)**2)
sim_phase = np.unwrap(np.angle(Ew))
sim_phase = sim_phase - sim_phase[posw0]

Et_sim = ifftshift(ifft(Ew, N))
Et_ret = ifftshift(ifft(Ew_ret, N))

plt.figure()
plt.pcolormesh(w, insertion, trace_sim, cmap='turbo')
ax = plt.gca()
ax.set_xlim([w_izq, w_der])
plt.show()

plt.figure()
plt.pcolormesh(w, insertion, trace_ret, cmap='turbo')
ax = plt.gca()
ax.set_xlim([w_izq, w_der])
plt.show()

plt.figure()
plt.plot(w, sim_spec, 'b')
ax = plt.gca()
ax.plot(w, ret_spec, 'r')
ax1 = ax.twinx()
ax1.plot(w, sim_phase, 'b:')
ax1.plot(w, ret_phase, 'r:')
ax.set_xlim([1.8, 2.9])
ax1.set_ylim([-5, 5])
plt.show()

plt.plot()
plt.plot(t, np.abs(Et_sim)**2/np.max(np.abs(Et_sim)**2), 'b')
ax = plt.gca()
ax.plot(t, np.abs(Et_ret)**2/np.max(np.abs(Et_ret)**2), 'r')
ax1 = ax.twinx()
ax1.plot(t, np.unwrap(np.angle(Et_sim)), 'b:')
ax1.plot(t, np.unwrap(np.angle(Et_ret)), 'r:')
ax.set_xlim([-30, 30])
ax1.set_ylim([-5, 5])
plt.show()

plt.figure()
plt.plot(w, mu)
ax = plt.gca()
ax.set_xlim([w_izq, w_der])
ax.set_ylim([0, 10])
plt.show()