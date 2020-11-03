import numpy as np
import matplotlib.pyplot as plt


# This was an assessed homework exercise in which I'm being graded on coding ability and commenting.
# Grade - 87%



# Part A - simulating 100 bounces if there were no exit hole present.

# creating variables for initial values

x1_initial = -0.03
y1_initial = 0.03
x2_initial = 0.02
y2_initial = 0.02
d = 0.5
R1 = 0.7
R2 = R1
n = 100

# solving for gradients using variables

grx1_initial = (x2_initial - x1_initial) / d
gry1_initial = (y2_initial - y1_initial) / d
print('Initial Gradient in X: {}. Initial Gradient in Y: {}'.format(grx1_initial, gry1_initial))

# creating 4x4 matrices for mirror reflections and distance

mirror2 = np.array([[1, 0, 0, 0], [-2/R2, 1, 0, 0], [0, 0, 1, 0], [0, 0, -2/R2, 1]])
mirror1 = np.array([[1, 0, 0, 0], [-2/R1, 1, 0, 0], [0, 0, 1, 0], [0, 0, -2/R1, 1]])
distance_matrix = np.array([[1,d,0,0],[0,1,0,0],[0,0,1,d],[0,0,0,1]])

M2D = np.matmul(mirror2, distance_matrix), # array with results of matrix multiplication of mirror2 reflection and distance
M1D = np.matmul(mirror1, distance_matrix)

B1 = np.zeros((4, n)) # sets up 4 x n matrix of zeros for n = 100 for mirror 1
B2 = np.zeros((4, n))

B1[:, 0] = np.array([x1_initial, grx1_initial, y1_initial, gry1_initial]) # sets first column of mirror 1 to the values of the variables passed in
B2[:, 0] = np.matmul(M2D, B1[:, 0]) # solves the linear equations of M2D with B1 and sets the resulting values to the first column of B2, uses eqn 1
M1DM2D = np.matmul(M1D, M2D) # creates array of the result of the matrix multiplication in eqn 2
B1[:, 1] = np.matmul(M1DM2D, B1[:, 0]) # solves the linear equation for the next round of values and sets the column at index 1 equal to these values

# solves for all values up to the nth case at second mirror by iterating through the range of n and performing matrix multiplication to the power of n
for i in range(n):
    B2[:, i] = np.matmul(np.linalg.matrix_power(M2D, i), B1[:, 0])
    B1[:, i] = np.matmul(np.linalg.matrix_power(M1DM2D, i), B1[:, 0])


# 1x2 subplot of positions on each mirror
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 8))
ax1.plot(B1[0, :], B1[2, :], 'o')
ax1.set_title('Optical Cavity A (Mirror 1)\n')
ax1.set_xlabel('\nX-Displacement')
ax1.set_ylabel('Y-Displacement\n')
ax2.plot(B2[0, :], B2[2, :], 'o')
ax2.set_title('Optical Cavity A (Mirror 2)\n')
ax2.set_xlabel('\nX-Displacement')
ax2.set_ylabel('Y-Displacement\n')
plt.show()


# B - Create functions to draw Circles

radius = 0.05
radius_entry_point = 0.005
theta = np.linspace(0,2 * np.pi, 1000)

def create_circle(X, Y, radius,theta):
    """
    CREATES A CIRCLE CENTRED AROUND THE GIVEN X AND Y POSITION, RADIUS AND ANGLE THETA.
    """
    x = X + radius * (np.cos(theta))
    y = Y + radius * (np.sin(theta))
    return np.array([x, y])

origin_circle = create_circle(0, 0, radius, theta) # array containing mirror outline values
entry_circle = create_circle(x1_initial, y1_initial, radius_entry_point, theta) # array containing entry hole boundary values

plt.figure(1)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.plot(B1[0, :], B1[2, :], 'ko') # plots the position values of x and y for mirror 1.
ax1.plot(origin_circle[0, :], origin_circle[1, :], '-') # plots the rows v columns of the origin_circle array, creating mirror 1.
ax1.plot(entry_circle[0, :], entry_circle[1, :], '-') # plots rows v columns of the entry point circle, plotting x v y
ax1.set_title('Optical Cavity B (Mirror 1)\n')
ax1.set_xlabel('\nX-Displacement')
ax1.set_ylabel('Y-Displacement\n')

ax2.plot(B2[0, :], B2[2, :], 'ko') # plots the position values of x and y for mirror 2
ax2.plot(origin_circle[0, :], origin_circle[1, :], '-') # plots mirror 2 shape
ax2.set_title('Optical Cavity B (Mirror 2)\n')
ax2.set_xlabel('\nX-Displacement')
ax2.set_ylabel('Y-Displacement\n')
plt.show()

# C - Removing exit rays from plot

# since some of the plotted positions on mirror 1 fall within the "entry hole", the ray at these positions would in fact
# exit the cavity and no further bounces would be recorded. In this section, a for loop is set up to determine on which bounce the
# ray first exits the entry hole, and the total distance travelled by the ray within the cavity before exiting.

# setting entry hole boundary limits in x
xmax = x1_initial + radius_entry_point
xmin = x1_initial - radius_entry_point


# for loop iterating from 1 to n, discounting 0 since 0 is the column containing the initial entry positions. Performs
# the required matrix multiplications as in part A and B, with if statements checking if the x and y position values fall within
# the boundary limits. The loop then breaks on the first incidence of where the ray exits and doesn't record any more bounces.
# The value of i returned is the nth bounce, and the distance is calculated using 2kd.


for i in range(1, n):
    if B1[0, i] >= xmin and B1[0, i] <= xmax:
        ymax = y1_initial + np.sqrt(radius_entry_point**2 - (B1[0, i] - x1_initial)**2)
        ymin = y1_initial - np.sqrt(radius_entry_point**2 - (B1[0, i] - x1_initial)**2)
        if B1[2, i] >= ymin and B1[2, i] <= ymax:
            print('Ray exits the cavity on bounce {},'.format(i), 'Total distance travelled = {} {}'.format(2*i*d, 'metres'))
            break


B1 = B1[:, 1:i] # starting at column index 1 since the values at column index 0 are initial values and not a bounce
B2 = B2[:, :i] # all columns included since all columns correspond to different bounces

# plot of the final data set, containing only the bounces up to K = 27, where the ray exits.

plt.figure(2)
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 8))
ax1.plot(B1[0, :], B1[2, :], 'ko')
ax1.plot(origin_circle[0, :], origin_circle[1, :], '-')
ax1.plot(entry_circle[0, :], entry_circle[1, :], '-')
ax1.set_title('Optical Cavity (Mirror 1 minus exit position)\n')
ax1.set_xlabel('\nX-Displacement')
ax1.set_ylabel('Y-Displacement\n')

ax2.plot(B2[0, :], B2[2, :], 'ko')
ax2.plot(origin_circle[0, :], origin_circle[1, :], '-')
ax2.set_title('Optical Cavity (Mirror 2 minus bounces after exit)\n')
ax2.set_xlabel('\nX-Displacement')
ax2.set_ylabel('Y-Displacement\n')
plt.show()
