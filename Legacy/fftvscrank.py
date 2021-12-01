"""
Compare fft algorithm and Crank-Nicolson algorithm on free 1d particle

FFT algorithm is 10 times faster than Crank-Nicolson while the error is less than 1e-3
"""
import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as sl
import time

#  Parameter Setup
dim = 1000
dt = 0.1

#  Setup X mesh
x_min = -10.
x_max = 30.
x_range = x_max - x_min
x0 = x_min
dx = x_range / dim

#  Define k mesh
k0 = -np.pi / dx
dk = 2. * np.pi / x_range

X = x0 + np.arange(dim) * dx
kmesh = k0 + np.arange(dim) * dk
phase_shift = kmesh ** 2 / 2

#  Functions
def normalize(x: np.ndarray) -> np.ndarray:
    return x / np.linalg.norm(x)


_ph1 = np.exp(-1.j * k0 * (X - x0))
_ph2 = np.conj(_ph1)
def evo_fft(x: np.ndarray, nt: int) -> np.ndarray:
    x = np.fft.fft(x * _ph1)
    x = x * np.exp(-1.j * nt * dt * phase_shift)
    return np.fft.ifft(x) * _ph2


hamil = - 0.5 * sp.dia_matrix(([-2*np.ones(dim), np.ones(dim), np.ones(dim)], [0, 1, -1]), shape=(dim, dim), dtype=complex)
def evo_cn(x: np.ndarray, nt:int) -> np.ndarray:
    rat = dt / (dx ** 2)
    b = (sp.eye(dim) - 1.j * rat * hamil / 2).dot(x)
    A = (sp.eye(dim) + 1.j * rat * hamil / 2)
    return sl.spsolve(A, b)


def get_probability(x: np.ndarray) -> np.ndarray:
    return np.abs(x) ** 2


def overlap(x: np.ndarray, y: np.ndarray) -> float:
    return np.dot(x, np.conj(y))


#  Model setup
inipsi = normalize(np.exp(- X ** 2 / 4 + 1.j * X))


def theo_wavefunc(nt: int) -> np.ndarray:
    t = nt * dt
    wv = np.exp(-1 + 1.j * (-2j + X)**2 / (4 * (-1.j + t/2))) / np.sqrt(2+1.j*t)
    return normalize(wv)


wv_th = np.copy(inipsi)
wv_fft = np.copy(inipsi)
wv_cn = np.copy(inipsi)



for n in range(50):
    wv_th = theo_wavefunc(n)
    t1 = time.time()
    wv_fft = evo_fft(wv_fft, 1)
    t2 = time.time()
    wv_cn = evo_cn(wv_cn, 1)
    t3 = time.time()

    print(f"At loop {n}: \n"
          f"\tTime cost: FFT = {t2-t1}\tCrank-Nicolson = {t3 - t2}\n"
          f"\t<fft|theo> = {overlap(wv_th, wv_fft)}\n"
          f"\t<cn|theo> = {overlap(wv_th, wv_cn)}\n"
          f"\t<cn|fft> = {overlap(wv_cn, wv_fft)}\n")





