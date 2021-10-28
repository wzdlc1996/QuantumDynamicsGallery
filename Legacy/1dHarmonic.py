import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sl

"""
Time evolution of Schrodinger equation of (standard) 1d harmonic oscillator

i psi_t = - psi_xx / 2 + x^2 psi / 2

By Crank-Nicolson method of second order 
"""


# Setup parameters
x_max = 10
x_min = -10
dim = 1000
X, dx = np.linspace(x_min, x_max, dim, retstep=True)
dt = 0.1
# V = X ** 2 / 2
V = np.array([1. if 0 < x < 2 else 0. for x in X]) * 10

# Setup initial wavefunction
x0 = -5
p0 = -2
wavef = np.exp(- (X - x0) ** 2 / (4 * 0.1) - 1.j * p0 * (X - x0))
wavef[0] = wavef[-1] = 0
wavef /= np.linalg.norm(wavef)

# Crank-Nicolson Subroutine
# Kinetic energy term
diag = -2 * np.ones(dim)
diag[0] = diag[-1] = 1
diag *= -0.5 * dt / dx ** 2
upper = np.ones(dim) * (-0.5 * dt / dx ** 2)
upper[0] = upper[1] = 0
lower = np.ones(dim) * (-0.5 * dt / dx ** 2)
lower[-1] = lower[-2] = 0
# Add potential term
bdflg = np.ones(dim)
bdflg[0] = bdflg[-1] = 0
diag += V * dt * bdflg
Hdt = sp.dia_matrix(([diag, upper, lower], [0, 1, -1]), shape=(dim, dim), dtype=np.complex64)

def next(psi):
    b = (sp.eye(dim) - 1.j * Hdt / 2).dot(psi)
    A = (sp.eye(dim) + 1.j * Hdt / 2)
    return sl.spsolve(A, b)

# Visualization
import matplotlib.pyplot as plt
import matplotlib.animation as anm

x = wavef
ylim = [min(np.abs(x) ** 2), max(np.abs(x) ** 2)]
fig, ax = plt.subplots()
ax.set_ylim(top=max(np.abs(x) ** 2))
frames = []
for k in range(1000):
    x = next(x)
    fm = ax.plot(X, np.abs(x) ** 2, animated=(k != 0), color="blue")
    frames.append(fm)

ani = anm.ArtistAnimation(fig, frames, interval=100, blit=True, repeat_delay=1000)
ani.save("/home/leonard/temp.mp4")
# plt.show()








