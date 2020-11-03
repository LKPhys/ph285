import numpy as np
import matplotlib.pyplot as plt
import scipy.integrate as ode
import timeit
import matplotlib.gridspec as gridspec
import scipy.optimize as opt


# homework project - graded 100%

# initial parameters
r30 = 2.31e-15  # pump rate parameter s^-1
r21 = 8.4e-15  # gain rate parameter s^-1
Y21 = 4350  # decay rate from 2 to 1 s^-1
Y32 = 1e12  # decay rate from 3 to 2 s^-1
Y10 = 1e12  # decay rate from 1 to 0 s^-1
Yc = 2.0e8  # cavity decay rate s^-1
N00 = 1.38e26  # population density in Level 0 m^-3
N10 = 9e21  # population density in Level 1 m^-3
N20 = 2e4  # population in Level 2 m^-3
N30 = 0  # population in Level 3 m^-3
A0 = np.array([1e12])  # initial photon number
P = 1e18  # number of pump photons
NT = N00 + N10 + N20 + N30  # Total population
time = np.arange(0, 100e-9, 1e-12)


def cavloss(t, A):
    """
    t: factor of integration
    A: initial value y0 = A
    returns: dA/dt as in equation (1)
    """
    return -Yc * A


A_loss = ode.solve_ivp(cavloss, [time[0], time[-1]], A0, t_eval=time)  # Computes solutions to our ODE

#  A_loss returns an OdeResult object containing t vals and y vals which can be extracted by calling
#  A_loss.t, A_loss.y. These values were then plotted.

Yg = np.array([1.1 * Yc, 1.5 * Yc, 2.0 * Yc])


def cavlossgain(t, A, Yg):
    """
    t: factor of integration
    A: initial condition
    Yg: gain rate
    dA/dt as in equation (2)
    """
    return (Yg - Yc) * A


start_time = timeit.default_timer()  # time stamp note for measuring runtimes

sols_ivp = []  # empty list to append solutions to
for i in range(len(Yg)):  # for loop iterating through Yg vals to compute ODE solutions for each value
    A_gain = ode.solve_ivp(lambda t, y: cavlossgain(t, y, Yg[i]), [time[0], time[-1]], A0,
                           t_eval=time)  # lambda function required to pass in Yg[i]
    sols_ivp.append(A_gain)  # appends solutions to empty list

end_time = timeit.default_timer()
ivp_time = end_time - start_time  # difference between start and end time for runtime measurement

A_Yg1 = sols_ivp[0].y.transpose()  # A values for Yg = 1.1 * Yc, 1.5 * Yc, 2.0 * Yc
A_Yg2 = sols_ivp[1].y.transpose()
A_Yg3 = sols_ivp[2].y.transpose()

start_time = timeit.default_timer()
sols_odeint = []
for i in range(len(Yg)):
    A_gain = ode.odeint(cavlossgain, A0, time, args=(Yg[i],),
                        tfirst=True)  # ODE solutions with odeint, allowing args=() use.
    sols_odeint.append(A_gain)  # Appends our solutions to sols_odeint in shape (len(time), len(Yg)) i.e (100000, 3)

# The solutions held in sols_odeint for Yg[i] are called using sols_odeint[:, i] where i = 0, 1, 2.

end_time = timeit.default_timer()
odeint_time = end_time - start_time

print('Part 2. Odeint has a runtime of {0:.2f}s'.format(odeint_time))
print('Part 2. Solve_ivp lambda has a runtime of {0:.2f}s'.format(ivp_time))

pop0 = np.array([N00, N10, N20, N30, A0[0], NT])  # initial values for population density
constants = [Y10, Y21, Y32, Yc, r30, r21]  # list holding all values which will not be varied
time_array = np.arange(0, 2e-6, 1e-9)  # time array


def rateeqns(t, pop, constants, P):
    """
    t: integrating factor
    pop: initial value array
    constants: array containing our constants
    P: pump parameter
    return: matrice containing solved ODEs for the initial values passed in
    """
    odes = np.zeros(6)  # empty array containing zeros to later hold ODE solutions
    odes[0] = (Y10 * pop[1]) - ((pop[0] - pop[3]) * r30 * P)  # dN0_dt
    odes[1] = (Y21 * pop[2]) + ((pop[2] - pop[1]) * r21 * pop[4]) - (Y10 * pop[1])  # dN1_dt
    odes[2] = (Y32 * pop[3]) - ((pop[2] - pop[1]) * r21 * pop[4]) - (Y21 * pop[2])  # dN2_dt
    odes[3] = ((pop[0] - pop[3]) * r30 * P) - (Y32 * pop[3])  # dN3_dt
    odes[4] = ((pop[2] - pop[1]) * r21 - Yc) * pop[4]  # dA_dt
    return odes


rates = ode.odeint(rateeqns, pop0, time_array, args=(constants, P), tfirst=True)  # ODE solver for constant P
normalised_rate = rates / NT  # normalised rates for N values, as A is not normalised to NT we use the rates array to plot A instead

pump = np.array([0.1, 0.25, 0.5, 0.75, 1.0, 2.0, 5.0, 10.0]) * 1e18  # array holding pump parameter
start_time = timeit.default_timer()

Pbuild_up = []  # empty list to later append values of t where photon no. is max
for i in range(len(pump)):
    sols = ode.odeint(rateeqns, pop0, time_array, args=(constants, pump[i]),
                      tfirst=True)  # computes sols for each pump[i]
    A_vals = sols[:, 4]  # selects the 5th column of the solutions where A data is stored
    index_no = np.where(A_vals == A_vals.max())  # finds the indices where total population number is max
    t_max = time_array[index_no[0]]  # returns the time value of the first instance where population is max
    Pbuild_up.append(t_max)  # appends resulting t_max value to our list for plotting

end_time = timeit.default_timer()
print('Part 4. Odeint has a runtime of {0:.2f}s'.format(end_time - start_time))

start_time = timeit.default_timer()

Pivp_buildup = []
for i in range(len(pump)):
    sols = ode.solve_ivp(lambda t, pop0: rateeqns(t, pop0, constants, pump[i]), (time_array[0], time_array[-1]), pop0,
                         t_eval=time_array)  # lambda function required to pass in and iterate through pump[i]
    A_vals = sols.y[4]  # selects the 5th solution, which corresponds to dA/dt
    index_no = np.where(A_vals == A_vals.max())
    t_max = time_array[index_no]
    Pivp_buildup.append(t_max)

end_time = timeit.default_timer()
print('Part 4. Solve_ivp has a runtime of {0:.2f}s'.format(end_time - start_time))

A_arr = np.array([0.01, 0.1, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0,
                  1000.0, 2500.0, 5000.0, 10000.0]) * 1e10  # Array holding varying A values

Abuild_up = []

for i in range(len(A_arr)):
    pop0[4] = A_arr[i]  # sets A0 value in initial value array, pop0, to the value of A0 at the given index i
    sols = ode.odeint(rateeqns, pop0, time_array, args=(constants, P), tfirst=True)  # P is held constant here again
    A_vals = sols[:, 4]
    index_no = np.where(A_vals == A_vals.max())
    t_max = time_array[index_no[0]]
    Abuild_up.append(t_max)

# Part 1 plots

plt.figure()
plt.title('Photon decay as a function of time\n')
plt.ylabel('Photon Number A0\n')
plt.xlabel('\nTime (s)')
plt.plot(A_loss.t, A_loss.y.transpose())
plt.grid(b=None, which='major', axis='both')
plt.show()

# Part 2 plots

plt.figure(figsize=(12, 12))
grid = gridspec.GridSpec(3, 2)
ax0 = plt.subplot(grid[0, 0])

ax0.plot(time_array * 1e9, normalised_rate[:, 0])
ax0.set_title('\nPopulation change in level 0 with respect to time\n', fontsize=10)
ax0.set_xlabel('\nTime (ns)', fontsize=12)
ax0.set_ylabel('Population Density N0\n')
ax0.grid(b=None, which='major', axis='both')

ax1 = plt.subplot(grid[0, 1])
ax1.plot(time_array * 1e9, normalised_rate[:, 1])
ax1.set_title('\nPopulation change in level 1 with respect to time\n', fontsize=10)
ax1.set_xlabel('\nTime (ns)', fontsize=12)
ax1.set_ylabel('Population Density N1\n')
ax1.set_yscale('log')
ax1.grid(b=None, which='major', axis='both')

ax2 = plt.subplot(grid[1, 0])
ax2.plot(time_array * 1e9, normalised_rate[:, 2])
ax2.set_title('\nPopulation change in level 2 with respect to time\n', fontsize=10)
ax2.set_xlabel('\nTime (ns)', fontsize=12)
ax2.set_ylabel('Population Density N2\n')
ax2.grid(b=None, which='major', axis='both')

ax3 = plt.subplot(grid[1, 1])
ax3.plot(time_array * 1e9, normalised_rate[:, 3])
ax3.set_title('\nPopulation change in level 3 with respect to time\n', fontsize=10)
ax3.set_xlabel('\nTime (ns)', fontsize=12)
ax3.set_ylabel('Population Density N3\n')
ax3.set_yscale('log')
ax3.grid(b=None, which='both', axis='both')

ax4 = plt.subplot(grid[2, 0])
ax4.plot(time_array * 1e9, rates[:, 4])
ax4.set_title('\nPhoton Number A0 variation with respect to time\n', fontsize=10)
ax4.set_xlabel('\nTime (ns)', fontsize=12)
ax4.set_ylabel('Photon Number A\n')
ax4.grid(b=None, which='major', axis='both')
plt.tight_layout()
plt.show()

# Part 3 plots

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(10, 5))
ax1.set_title('Variation in photon number A0 for\nYg = 1.1Yc\n')
ax1.plot(sols_ivp[0].t, A_Yg1, 'b--', time, sols_odeint[0], '-')
ax1.set_xlabel('\nTime (s)')
ax1.set_ylabel('Photon Number A0\n')
ax1.grid(b=None, which='major', axis='both')

ax2.set_title('Variation in photon number A0 for\nYg = 1.5Yc\n')
ax2.plot(sols_ivp[1].t, A_Yg2, 'b--', time, sols_odeint[1], '-')
ax2.set_ylabel('Photon Number A0\n')
ax2.set_xlabel('\nTime (s)')
ax2.grid(b=None, which='major', axis='both')

ax3.set_title('Variation in photon number A0 for\nYg = 2.0Yc\n')
ax3.plot(sols_ivp[2].t, A_Yg3, 'b--', time, sols_odeint[2], '-')
ax3.set_ylabel('Photon Number A0\n')
ax3.set_xlabel('\nTime (s)')
ax3.grid(b=None, which='major', axis='both')
plt.tight_layout()
plt.show()

plt.figure()
plt.plot(pump, Pbuild_up, 'kx', label='Odeint (LSODA)')
plt.plot(pump, Pivp_buildup, '-', label='Solve_ivp (RK45)')
plt.title('Build-up time as a function of Pump Photons P\n', fontsize=12)
plt.xlabel('\nPump Photons P', fontsize=12)
plt.ylabel('Time (s)')
plt.grid(b=None, which='major', axis='both')
plt.legend()
plt.show()

# Part 5 plot

plt.figure()
plt.plot(A_arr, Abuild_up)
plt.title('Build-up time as a function of A0\n')
plt.xlabel('\nInitial Photon Number A0')
plt.ylabel('Build-up time (s)\n')
plt.grid(b=None, which='major', axis='both')
plt.show()
