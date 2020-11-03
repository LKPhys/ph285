#!usr/bin/env python
import numpy as np

# Solvers required for project 1

# Gaussian elimination method


def gaussian_elimination(A, b):
    """
    Solves a system of linear equations by gaussian elimination
    :param A: input matrix A
    :param b: resulting matrix
    :return: solutions for x in an array
    """

    n = len(b)
    x = np.zeros(n)
    V = A.copy()
    w = b.copy()

    for i in range(n-1):  # sets coeff = a_ii/a_ji, subtracts this val from jth row for each k element, reducing matrix
        for j in range(i+1, n):
            if V[j][i] == 0:
                continue

            coeff = V[i][i]/V[j][i]  # multiplicative term (a_ij)/(a_ii)
            w[j] = (w[i] - coeff * w[j])

            for k in range(i, n):
                V[j][k] = V[i][k] - coeff * V[j][k]  # multiplies each row by coeff and subtracts it from each element

    x[n-1] = w[n-1] / V[n-1][n-1]  # solves the last row of the upper triangle matrix for X

    for i in range(n-2, -1, -1):  # goes to the next row at n-2, then moves backwards to -1 in steps of -1
        vxsum = 0  # starts counter for the sum of the vx term

        for j in range(i+1, n):
            vxsum += V[i][j] * x[j]

        x[i] = (w[i] - vxsum)/V[i][i]  # solves for given x by subtracting unwanted coefficient

    return x


# gauss seidel method


def gauss_seidel(A,b, iterations=100, x=None):
    """
    Gauss Seidel method for input matrices A and B. Returns solution vector x.
    """

    n = len(b)

    if x is None:
        x = np.zeros(n)

    D = np.zeros((n, n))
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    #find diagonal and lower and upper matrices D, L, U respectively

    for i in range(n):
        for j in range(n):
            if j == i:
                D[i, j] = A[i, j]
            elif j-i > 0:
                L[i, j] = A[i, j]
            else:
                U[i, j] = A[i, j]

    invDL = np.linalg.inv(D + L)

    c = np.zeros(iterations)
    for i in range(len(c)):
        x = np.dot(invDL, (b - np.dot(U, x)))
        c = x

    for i in range(len(c)):
        if c[i] - c[i-1] == 0:
            print('converges at N =:\n', i)
        break


    j = np.dot(np.linalg.inv(D), (L + U))
    lambda1, lambda2 = np.linalg.eig(j)
    specrad1 = np.max(np.absolute(lambda1))
    specrad2 = np.max(np.absolute(lambda2))
    print(specrad1, specrad2)

    return x


# Jacobian iteration method

def jacobian(A, b, N=100, x=None):
    """
    Systems of linear equations solver using jacobian iteration method for input matrices A and b, returning solution vector x.
    """
    n = len(b)
    D = np.zeros((n, n))
    x = np.zeros(n)

    if x is None:
        x = np.zeros(n)

    for i in range(n):
        for j in range(n):
            if i == j:
                D[i][j] = A[i][j]
            else:
                continue

    lowerupper = A - D  # removes diagonal from A
    for i in range(N):
        x = np.dot(np.linalg.inv(D), (b - np.dot(lowerupper, x)))

    j = np.dot(np.linalg.inv(D), lowerupper)
    lambda1, lambda2 = np.linalg.eig(j)
    specrad1 = np.max(np.absolute(lambda1))
    specrad2 = np.max(np.absolute(lambda2))
    print(specrad1, specrad2)

    return x


# SOR method


def successive_over_relaxation(A,b, iterations=100, x=None):
    """
    Successive Over Relaxation iteration method for input matrices A and b. Returns solution vector x.
    """

    n = len(b)
    if x is None:
        x = np.zeros(len(b))

    D = np.zeros((n, n))
    L = np.zeros((n, n))
    U = np.zeros((n, n))

    #find diagonal and lower and upper matrices D, L, U respectively
    for i in range(n):
        for j in range(n):
            if j == i:
                D[i, j] = A[i, j]

            elif j-i > 0:
                U[i, j] = A[i, j]

            else:
                L[i, j] = A[i, j]

    # Relaxation parameter W calculation
    J = np.dot(np.linalg.inv(D), (L + U))
    w, v = np.linalg.eig(J)  # w here is the eigenvalues of J, not the same as relaxation parameter below
    w_abs = np.absolute(w)
    spectral = np.max(w_abs)
    w = 2 / (1 + (1-spectral**2)**0.5)

    # calculates terms 1, 2 and 3 as required for the final equation for x(k+1)
    t1 = np.linalg.inv(D + (w * L))
    t2 = (w * b)
    t3 = (w * U + (w - 1) * D)

    #Iteration
    for i in range(iterations):
        x = np.dot(t1, (t2 - np.dot(t3, x)))

    return x
