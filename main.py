import numpy as np
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt
from scipy.fft import fft, ifft, fftshift, ifftshift

c = 3e2  # nm/fs

def readSpectrum(filePath, w):
    spectrum = np.loadtxt(filePath)
    spectrum[:, 0] = spectrum[:, 0] * 1e9

    modAxis = np.divide(2 * np.pi * c, spectrum[:, 0], where=spectrum[:, 0] != 0)
    freqSpec_f = interp1d(modAxis, spectrum[:, 1] * (spectrum[:, 0]**2/(2*np.pi*c)), kind='cubic',
                          bounds_error=False,
                          fill_value=0)
    freqSpec = np.abs(freqSpec_f(w))
    freqSpec = freqSpec/np.max(freqSpec)
    freqPhase_f = interp1d(modAxis, spectrum[:, 2], kind='cubic', bounds_error=False, fill_value=0)
    freqPhase = freqPhase_f(w)

    return np.sqrt(freqSpec)*np.exp(1j*freqPhase)


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

print(Ew_prop.shape)

Et_prop = ifftshift(ifft(Ew_prop, N, 2), 2)

angles = np.reshape(np.arange(0, 360, 2), (Nz, 1))

Et_proy = Et_prop[0, :, :]*np.cos(angles*np.pi/180) + Et_prop[1, :, :]*np.sin(angles*np.pi/180)
Et_SHG = Et_proy**2
Ew_SHG = fft(ifftshift(Et_SHG, 1), N, 1)

print(Et_SHG.shape)

plt.figure()
plt.pcolormesh(w, angles, np.abs(Ew_SHG)**2, cmap='turbo')
plt.show()



N_iters = 300

Ew_ret = np.array([np.abs(Ew_x), np.abs(Ew_y)])
Ew_ini = Ew_ret

for i in range(N_iters):

    
    Ew_ret_prop = np.multiply(Ew_ret[:, np.newaxis, :], np.exp(-1j * prop_phase))
    Et_ret_prop = ifftshift(ifft(Ew_ret_prop, N, 2), 2)
    Et_ret_proy = Et_ret_prop[0, :, :] * np.cos(angles * np.pi / 180) + Et_ret_prop[1, :, :] * np.sin(angles * np.pi / 180)
    Et_ret_SHG = Et_ret_proy ** 2
    Ew_ret_SHG = fft(ifftshift(Et_ret_SHG, 1), N, 1)

    Ew_new_ret_SHG = np.divide(np.abs(Ew_SHG)*Ew_ret_SHG, np.abs(Ew_ret_SHG), where=np.abs(Ew_ret_SHG)!=0)
    Et_new_ret_SHG = ifftshift(ifft(Ew_new_ret_SHG, N, 1), 1)
    Et_new_ret_proy = 1/2*(2*Et_ret_proy+(np.conjugate(Et_ret_proy)/(np.max(np.abs(Et_ret_proy)**2)))*(Et_new_ret_SHG-Et_ret_SHG))
    
    # Et_new_ret_prop = np.zeros((2, Nz, N), dtype='complex_')
    Et_ret_prop[0, :, :] = np.divide((Et_new_ret_proy - np.sin(angles * np.pi / 180)*Et_ret_prop[1, :, :]),  np.cos(angles * np.pi / 180), where=np.cos(angles * np.pi / 180)!=0)

    Et_ret_proy = Et_ret_prop[0, :, :] * np.cos(angles * np.pi / 180) + Et_ret_prop[1, :, :] * np.sin(angles * np.pi / 180)
    Et_ret_SHG = Et_ret_proy ** 2
    Ew_ret_SHG = fft(ifftshift(Et_ret_SHG, 1), N, 1)

    Ew_new_ret_SHG = np.divide(np.abs(Ew_SHG) * Ew_ret_SHG, np.abs(Ew_ret_SHG), where=np.abs(Ew_ret_SHG) != 0)
    Et_new_ret_SHG = ifftshift(ifft(Ew_new_ret_SHG, N, 1), 1)
    Et_new_ret_proy = 1 / 2 * (2 * Et_ret_proy + (np.conjugate(Et_ret_proy) / (np.max(np.abs(Et_ret_proy) ** 2))) * (
                Et_new_ret_SHG - Et_ret_SHG))

    Et_ret_prop[1, :, :] = np.divide((Et_new_ret_proy - np.cos(angles * np.pi / 180)*Et_ret_prop[0, :, :]), np.sin(angles * np.pi / 180), where=np.sin(angles * np.pi / 180)!=0)

    # Et_new_ret_prop[0, :, :] = (Et_new_ret_proy - np.sin(angles * np.pi / 180)*Et_ret_prop[1, :, :])
    # Et_new_ret_prop[1, :, :] = (Et_new_ret_proy - np.cos(angles * np.pi / 180)*Et_ret_prop[0, :, :])


    Ew_new_ret_prop = fft(ifftshift(Et_ret_prop, 2), N, 2)
    Ew_new_ret = Ew_new_ret_prop * np.exp(+1j * prop_phase)


    Ew_ret = np.average(Ew_new_ret, axis=1)
    # Ew_ret = Ew_ret/np.max(np.abs(Ew_ret))

    Ew_ret = np.abs(Ew_ini)*np.exp(1j*np.angle(Ew_ret))
    print(i)

plt.figure()
plt.pcolormesh(w, angles, np.abs(Ew_ret_SHG)**2, cmap='turbo')
plt.show()

plt.figure()
plt.plot(w, np.abs(Ew_ret[0, :]))
ax1 = plt.gca()
ax2 = plt.twinx(ax1)
ax2.plot(w, np.unwrap(np.angle(Ew_ret[0, :])))
ax2.plot(w, np.unwrap(np.angle(Ew_ret[1, :])))
ax2.plot(w, np.unwrap(np.angle(Ew_x)))
plt.show()

plt.figure()
plt.pcolormesh(np.abs(ifftshift(ifft(Ew_new_ret[0,:,:], N, 1), 1)), cmap='turbo')
plt.show()
