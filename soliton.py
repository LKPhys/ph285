import numpy as np
import timeit
import matplotlib.pyplot as plt
import scipy.integrate as ode
from mpl_toolkits.mplot3d import Axes3D

L = 88  # length of fibre
n = 20  # number of divisions of L
g = 1.4e-3  # intensity parameter
tp = 1e-12  # half pulse width
b2 = 1.8e-26  # dispersion parameter
N = np.array([1, 2, 3])  # N array
gridpts = 2048  # grid
t, tstep = np.linspace(-20 * tp, 20 * tp, gridpts, True, True)  # time array and step
w = np.fft.fftfreq(gridpts, tstep)  # frequency array taken as fft of time array
dl = np.linspace(0, L, 401, True)  # length array to be integrated over
A0 = N * np.sqrt(b2/(g * tp**2))  # initial amplitude values for each N


def intensity(x):
    """
    Takes the modulus squared of a complex input
    """
    return x * np.conj(x)


def sech(x):
    """
    hyperbolic secant
    """
    return 2/(np.exp(x) + np.exp(-x))


def dispersion(t, y0, b2, w):
    """
    computes the dispersive part of the solution to the nonlinear schrodinger eqn
    w = frequency array,
    b2 = dispersion parameter,
    y = pulse.
    """
    dadt = -1j * b2 * (w * 2 * np.pi)**2 * y0/2
    return dadt


def nonlinear(t, y0, g):
    """
    solution to the nonlinear schrodinger eqn
    y = pulse,
    g = nonlinear constant.
    """
    dadt = 1j * g * intensity(y0) * y0
    return dadt


start = timeit.default_timer()

# DISPERSION

pulse = []
for i in range(len(A0)):
    pulse.append(A0[i] * sech(t/tp) + 0j)  # computes the input pulse profile in the time domain for each A0

pulse = np.array(pulse)
pulsefft = np.fft.fft(pulse/gridpts)  # fourier transform of the pulse into frequency domain


# solves the ODE given by the dispersive solution to NLSE, integrating wrt length. Initial value here is the pulse at
# t=0 in freq domain. Repeats for each A0 and stores solution in empty array.
dispsol = []
for i in range(len(pulsefft)):
    dispsol.append(ode.solve_ivp(lambda t, y: dispersion(t, y, b2, w), (dl[0], dl[-1]),
                   pulsefft[i], t_eval=dl, method='BDF'))


# extracting the solutions from dispersion_sols in both time and frequency domain. We take every 40th solution because
# the dispersive segment of the fibre begins at L=0 and the next at L/2n repeating.
d_time = []
d_freq = []
for i in range(len(A0)):
    freqsol = dispsol[i].y.T  # extracts the solution for each N
    d_sols = freqsol[0: -1: 2 * n, :]
    timesol = np.fft.fft(d_sols)  # inverse fourier transform back to time domain  # finds intensity of time solutions
    d_time.append(intensity(timesol))  # takes every 40th solution (dispersive segments)
    d_freq.append(intensity(d_sols))

d_pulse = np.array(d_time)
d_spectral = np.array(d_freq)


# PLOTS

Z = dl[0: -1: 2 * n]  # Creating a Z axis with every 40th data point in the distance array
for k in range(len(A0)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(len(Z)):
        Z_axis = np.full((1, 2048), Z[j])  # "Padding out" the axis to include 2048 data points of those points in Z
        ax.plot(t * 1e11, Z_axis[0], d_pulse[k, j])
        ax.set_title('Dispersion pulse profile for N = {}'.format(N[k]), loc='left')
        ax.set(xlabel='Time (ps)', ylabel='Distance(m)')
    plt.show()


for k in range(len(A0)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(len(Z)):
        Z_axis = np.full((1, 2048), Z[j])  # "Padding out" the axis to include 2048 data points of those points in Z
        ax.plot(w,  Z_axis[0], d_spectral[k, j])
        ax.set_title('Dispersion spectral profile for N = {}'.format(N[k]), loc='left')
        ax.set(xlabel='Frequency (Hz)', ylabel='Distance(m)')
        ax.set_xlim((-0.1e13, 0.1e13))
    plt.show()


# NONLINEAR

# solves the ODE given by the nonlinear solution to NLSE, integrating wrt length. Initial value here is the pulse at
# t=0 in time domain. Repeats for each A0 and stores solution in empty array.
nonlinear_sols = []
for i in range(len(pulse)):
    nonlinear_sols.append(ode.solve_ivp(lambda t, y: nonlinear(t, y, g), (dl[0], dl[-1]),
                          pulse[i], t_eval=dl, method='BDF'))

# extracting the solutions from nonlinear_sols in both time and frequency domain. We take every 40th solution again
# but instead take we take every 40th solution beginnning at L=L/4n.
nl_freq = []
nl_time =[]
for i in range(len(A0)):
    timesol = nonlinear_sols[i].y.T
    nl_sols = timesol[20: -1: n * 2, :]  # selecting every 40th
    freqsol = np.fft.fft(nl_sols)  # frequency solutions
    nl_freq.append(intensity(freqsol))
    nl_time.append(intensity(nl_sols))

nl_spectral = np.array(nl_freq)
nl_pulse = np.array(nl_time)


# PLOTS

Z = dl[20: -1: n * 2]  # creating axis
for k in range(len(A0)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(len(Z)):
        Z_axis = np.full((1, 2048), Z[j])
        ax.plot(t * 1e11, Z_axis[0], nl_pulse[k, j, :])
        ax.set_title('Nonlinear pulse profile for N = {}'.format(N[k]), loc='left')
        ax.set(xlabel='Time (ps)', ylabel='Distance(m)')
    plt.show()


for k in range(len(A0)):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    for j in range(len(Z)):
        Z_axis = np.full((1, 2048), Z[j])
        ax.plot(w, Z_axis[0], nl_spectral[k, j, :])
        ax.set_title('Nonlinear spectral profile for N = {}'.format(N[k]), loc='left')
        ax.set(xlabel='Frequency (Hz)', ylabel='Distance(m)')
        ax.set_xlim((-0.1e13, 0.1e13))
    plt.show()



end = timeit.default_timer()
print('Runtime: {0:3f} min'.format((end-start)/60))
