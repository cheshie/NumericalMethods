from functools import partial
from timeit import timeit
from numpy.linalg import solve
from numpy import eye, dot, ones, concatenate, max, abs
from numpy import triu, tril
from numpy.random import rand
import matplotlib.pyplot as plt


def jacobi_simple(A, b, X, e):
    W, Z = A, b
    WZ = concatenate((A, Z), axis=1)
    n  = max(A.shape)
    X  = X.copy()

    # Create WZ matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                WZ[i, :] = WZ[i, :] / W[i, i]
                WZ[i, :n] = WZ[i, :n] * -1
                WZ[i, i] = 0
    # Get W and Z matrices
    W, Z = WZ[:, :-1], WZ[:, -1]

    # Simple iterative method
    i = 0
    while True:
        i += 1
        X_prev = X.copy()
        X = dot(W, X) + Z.reshape(-1, 1)
        if max(abs(X - X_prev)) < e:
            break


    # Return solution
    # In our case solution should be always equal to 1
    # So we always return number of iterations
    return i
#

def gauss_seidel(A, b, X, e):
    # Example task - for this algorithm is tested
    # n = 2
    # A = array([[4, 1], [1, 2]], dtype=float64)
    # b = array([[8], [9]], dtype=float64)
    # X = zeros((n, 1))

    W, Z = A, b
    WZ = concatenate((A, Z), axis=1)
    n = max(A.shape)
    X = X.copy()

    # Create WZ matrix
    for i in range(n):
        for j in range(n):
            if i == j:
                WZ[i, :] = WZ[i, :] / W[i, i]
                WZ[i, :n] = WZ[i, :n] * -1
                WZ[i, i] = 0

    # Get W and Z matrices
    W, Z = WZ[:, :-1], WZ[:, -1]

    Wu = triu(WZ[:, :n])
    Wl = tril(WZ[:, :n])

    # Gauss Seidel method
    i = 0
    while True:
        i += 1
        X_prev = X.copy()

        X1_temp =  dot(Wu, X) + Z.reshape(-1, 1)
        # X1_temp += Z.reshape(-1, 1)[:X1_temp.shape[0]]
        X =  dot(Wl, X1_temp) + Z.reshape(-1, 1)

        if max(abs(X - X_prev)) < e:
            break

    # Return solution
    # In our case solution should be always equal to 1
    # So we always return number of iterations
    return i
#

# zadanie nr 2
n = [10, 100, 200, 500, 1000, 2000, 3000, 5000]
e = [0.1, 0.001]

# time results for simple, gauss, exact solution
res_e1 = [[], [], []]
res_e2 = [[], [], []]
for dim in n:
    for dokl in e:
        A = rand(dim, dim)
        # A = A + n * E
        A = A + dim * eye(dim)
        # column vector of ones - starting vector
        X = ones((dim, 1))
        # wyrazy wolne: b = A * x
        b = dot(A, X)

        """
        print("="*10, "SIZE: ", dim, "="*10)
        print("simple: ", timeit(partial(jacobi_simple, A, b, X, dokl), number=5), ' s')
        print("simple: ", jacobi_simple(A, b, X, dokl), ' iterations')
        print("-" * 10)
        print("seidel: ", timeit(partial(gauss_seidel, A, b, X, dokl), number=5), ' s')
        print("seidel: ", gauss_seidel(A, b, X, dokl), ' iterations')
        print("-" * 10)
        print("exact: ", timeit(partial(solve, A, b), number=5), ' s')
        """
        if dokl == 0.1:
            res_e1[0].append(timeit(partial(jacobi_simple, A, b, X, dokl), number=5))
            res_e1[1].append(timeit(partial(gauss_seidel, A, b, X, dokl), number=5))
            res_e1[2].append(timeit(partial(solve, A, b), number=5))
        else:
            res_e2[0].append(timeit(partial(jacobi_simple, A, b, X, dokl), number=5))
            res_e2[1].append(timeit(partial(gauss_seidel, A, b, X, dokl), number=5))
            res_e2[2].append(timeit(partial(solve, A, b), number=5))

fig, ax = plt.subplots()
plt.plot(n, res_e1[0], "b", label='simple')
plt.plot(n, res_e1[1], "r", label='gauss')
plt.plot(n, res_e1[2], "g", label='exact')
ax.set_ylabel('Time')
ax.set_title('Exec time for e = 0.1')
ax.legend()
plt.show()

fig, ax = plt.subplots()
plt.plot(n, res_e2[0], "b", label='simple')
plt.plot(n, res_e2[1], "r", label='gauss')
plt.plot(n, res_e2[2], "g", label='exact')
ax.set_ylabel('Time')
ax.set_title('Exec time for e = 0.001')
ax.legend()
plt.show()