import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sl

"""
Time evolution of Schrodinger equation of (standard) 2D harmonic oscillator

i psi_t = - (psi_xx + psi_yy) / 2 + (x^2 + y^2) psi / 2

by the method in https://aip.scitation.org/doi/abs/10.1063/1.168415
an explicit and stable method
"""

ds = 0.01  # dx = dy = ds
x_ax = np.arange(0, 1, ds)
y_ax = np.arange(0, 1, ds)
X, Y = np.meshgrid(x_ax, y_ax)
V = 10 * ((X-0.5) ** 2 + (Y-0.5) ** 2) / 2

dt = 0.00005

# Setup initial wavefunction
x0, y0 = 0.5, 0.5
px0, py0 = 1, 1
wavef = np.exp(- ((X - x0) ** 2 + (Y - y0) ** 2) / (4 * 0.01) + 1.j * (px0 * (X - x0) + py0 * (Y - y0)))
wavef /= np.linalg.norm(wavef)

def x_diff2(wf):
    roll_p = np.zeros(wf.shape, dtype=wf.dtype)
    roll_n = np.zeros(wf.shape, dtype=wf.dtype)
    roll_p[1:] = wf[:-1]
    roll_n[:-1] = wf[1:]
    return (roll_p + roll_n - 2 * wf) * (dt / (ds ** 2))


def y_diff2(wf):
    return x_diff2(wf.T).T


def hamil_kinetic(wf):
    return - (x_diff2(wf) + y_diff2(wf)) / 2


def hamil(wf):
    return hamil_kinetic(wf) + V * dt * wf


def next(psi_r, psi_i):
    psi_r = psi_r + hamil(psi_i)
    return psi_r, psi_i - hamil(psi_r)


# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as anm

# Initialization the wavefunction and half_time one
x = wavef
x_h = x - 1.j * hamil(x) * dt / 2
x_h /= np.linalg.norm(x_h)
x_v = x + 1.j * hamil(x) * dt / 2
x_v /= np.linalg.norm(x_v)
psir = np.real(x)
psii = np.imag(x_h)
temp = np.imag(x_v)

fig, ax = plt.subplots()

probs = []
z = psir ** 2 + psii * temp
z = z[:-1, :-1]
probs.append(z)

drw = ax.pcolormesh(X, Y, probs[0])

for _ in range(200):
    for k in range(int(0.005 / dt)):
        temp = psii
        psir, psii = next(psir, psii)

    z = psir ** 2 + psii * temp
    z = z[:-1, :-1]
    probs.append(z)



def anim(i):
    drw.set_array(probs[i].flatten())


ani = anm.FuncAnimation(fig, anim, interval=100, frames=len(probs)-1, repeat_delay=1000)
ani.save("/home/leonard/temp_fast.mp4")