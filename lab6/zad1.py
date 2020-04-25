from numpy import array, triu, tril, trace, eye, dot, ones
from numpy.linalg import det, eig, inv, matrix_rank
from numpy import array, triu, tril, trace, float64, cumprod, poly, roots, set_printoptions, sum
from numpy.linalg import det, eig, inv, eigvals
from numpy.random import rand
from itertools import accumulate

n = 3
A = rand(3,3)
# A = A + n * E
A = A + dot(n, eye(n))
# column vector of ones
X = ones((n, 1))
# Solution: b = A * x
b = dot(A, X)



