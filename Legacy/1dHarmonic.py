import numpy as np
import scipy
import scipy.sparse as sp
import scipy.sparse.linalg as sl

"""
Time evolution of Schrodinger equation of (standard) 1d harmonic oscillator

i psi_t = - psi_xx / 2 + x^2 psi / 2
"""


# Setup parameters
x_max = 10
x_min = -10
dim = 1000
X, dx = np.linspace(x_min, x_max, dim, retstep=True)
dt = 0.1
V = X ** 2 / 2

# Setup initial wavefunction
x0 = 1
p0 = 1
wavef = np.exp(- (X - x0) ** 2 / (4 * 0.1) + 1.j * p0 * (X - x0))
wavef /= np.linalg.norm(wavef)

# Crank-Nicolson Subroutine
diag = -2 * np.ones(dim)
diag[0] = diag[-1] = 1
diag *= -0.5 / dx ** 2
diag += V
upper = np.ones(dim) * (-0.5 / dx ** 2)
upper[0] = upper[1] = 0
lower = np.ones(dim) * (-0.5 / dx ** 2)
lower[-1] = lower[-2] = 0
hamil = - 0.5 * (1 / dx ** 2) * sp.dia_matrix(([diag, upper, lower], [0, 1, -1]), shape=(dim, dim), dtype=np.complex64)

def next(psi):
    b = (sp.eye(dim) - 1.j * dt * hamil / 2).dot(psi)
    A = (sp.eye(dim) + 1.j * dt * hamil / 2)
    return sl.spsolve(A, b)

# Visualization
import matplotlib.pyplot as plt
x = wavef
for _ in range(1000):
    x = next(x)
plt.plot(X, np.abs(x) ** 2)
plt.show()








