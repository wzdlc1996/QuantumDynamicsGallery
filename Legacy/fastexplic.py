import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sl

"""
Time evolution of Schrodinger equation of (standard) 1d harmonic oscillator

i psi_t = - psi_xx / 2 + x^2 psi / 2

by the method in https://aip.scitation.org/doi/abs/10.1063/1.168415
an explicit and stable method
"""

# Setup parameters
x_max = 10
x_min = -10
dim = 100
X, dx = np.linspace(x_min, x_max, dim, retstep=True)
dt = 0.01
# V = X ** 2 / 2
V = np.array([0. if 0 < x < 2 else 1. for x in X]) * 10
# V = 0 * X

# Setup initial wavefunction
x0 = -5
p0 = 2
wavef = np.exp(- (X - x0) ** 2 / (4 * 0.1) + 1.j * p0 * (X - x0))
wavef[0] = wavef[-1] = 0
wavef /= np.linalg.norm(wavef)

# Crank-Nicolson Subroutine
diag = -2 * np.ones(dim)
diag[0] = diag[-1] = 1
diag *= -0.5 * dt / dx ** 2
bdflg = np.ones(dim)
bdflg[0] = bdflg[-1] = 0
diag += V * dt * bdflg
upper = np.ones(dim) * (-0.5 * dt / dx ** 2)
upper[0] = upper[1] = 0
lower = np.ones(dim) * (-0.5 * dt / dx ** 2)
lower[-1] = lower[-2] = 0
Hdt = sp.dia_matrix(([diag, upper, lower], [0, 1, -1]), shape=(dim, dim), dtype=np.double)


def next_crank(psi, delt_t=dt):
    Hdelt_t = Hdt * (delt_t / dt)
    b = (sp.eye(dim) - 1.j * Hdelt_t / 2).dot(psi)
    A = (sp.eye(dim) + 1.j * Hdelt_t / 2)
    return sl.spsolve(A, b)


# Unstable, need review
def next(psi_r, psi_i):
    psi_r = psi_r + Hdt.dot(psi_i)
    return psi_r, psi_i - Hdt.dot(psi_r)


# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as anm

# Initialization the wavefunction and half_time one
x = wavef
x_h = next_crank(x, dt / 2)
x_v = next_crank(x, -dt / 2)
temp = np.imag(x_v)
psir = np.real(x)
psii = np.imag(x_h)


ylim = [min(np.abs(x) ** 2), max(np.abs(x) ** 2)]
fig, ax = plt.subplots()
ax.set_ylim(top=max(np.abs(x) ** 2))
frames = []
for k in range(10000):
    temp = psii
    psir, psii = next(psir, psii)
    if k % 10 == 0:
        prob = psir ** 2 + psii * temp
        fm = ax.plot(X, prob, animated=(k != 0), color="blue")
        frames.append(fm)

ani = anm.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
ani.save("/home/leonard/temp_fast.mp4")
# plt.show()