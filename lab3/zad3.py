from numpy import array, triu, tril, trace
from numpy.linalg import det, eig, inv, matrix_rank
from numpy import array, triu, tril, trace, float64, cumprod, poly, roots, set_printoptions, sum, linspace, real, imag, arange
from numpy.linalg import det, eig, inv, eigvals
from itertools import accumulate
import matplotlib.pyplot as plt

set_printoptions(precision=3)
set_printoptions(suppress=True)

# from https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


# 1)
A = array([[1,-1,2], [2,1,-4], [3,0,6]], dtype=float64)
print("Macierz A: \n")
matprint(A)

# 2) Wspolczynniki wielomianu charakterystycznego:
print("wielomian charakterystyczny z A: ", poly(A))
# 3) Pierwiastki
print("Pierwiastki z A: ", roots(poly(A)))

# Plot
# T = linspace(0.0,0.9,100)
fig, ax = plt.subplots()
plt.plot(imag(roots(poly(A))), "b*", label='imag')
plt.plot(real(roots(poly(A))), "r*", label='real')
ax.set_ylabel('Error')
ax.set_title('Polynomial roots')
ax.legend()
plt.show()