import numpy as np


"""
Numpy random matrix: 

x = np.random.randint(2, size=10) # vector with zeros and ones
x = np.random.randint(5, size=(2, 4)) # matrix of given size 
x = np.random.rand(3,2) # floats 
"""


"""
Matrix decomposition L/U:
https://www.quantstart.com/articles/LU-Decomposition-in-Python-and-NumPy/

import scipy.linalg   # SciPy Linear Algebra Library
A = np.random.randint(5, size=(2, 4))
P, L, U = scipy.linalg.lu(A)
"""

"""
Scipy: basic numerical operations: 
https://scipy-lectures.org/intro/numpy/operations.html
!!!! https://docs.scipy.org/doc/numpy/reference/routines.linalg.html#module-numpy.linalg

b = np.ones(4) + 1
j = np.arange(5)
print(2**(j + 1) - j)

Matrix operations: 
c = np.ones((3,3)) 
c * c <=== ELEMENT-WISE MULTIPLICATION
np.dot(c, c) <=== MATRIX MULTIPLICATION, for vectors it will return scalar product
np.cross(x,y) <=== cross product

Upper-triangular matrix and bottom-triangular: 
a = np.triu(np.ones((3, 3)), 1)  # or np.tril(np.ones(...))

Reductions: 
np.sum(A, axis=1)

Extrema: 
x.min() x.max() x.argmin() x.argmax() # latter two return indexes   

Conditions:
np.any(A != 0) or np.all(A != 0)
"""



"""
https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.solve.html
# gaussian elimination: 
https://gist.github.com/jgcastro89/49090cc69a499a129413597433b9baab
https://docs.sympy.org/dev/modules/solvers/solvers.html
http://poquitopicante.blogspot.com/2014/04/gaussian-elimination-using-lu.html
# Solvers, Derivatives :: https://docs.sympy.org/dev/modules/solvers/solvers.html
Solving linear equations: 
equations 3 * x0 + x1 = 9 and x0 + 2 * x1 = 8

a = np.array([[3,1], [1,2]])
b = np.array([9,8])
x = np.linalg.solve(a, b)
print(x) 
np.allclose(np.dot(a, x), b) # check solutions
"""


"""
Examples from the lecture: 
trace() => sum elements along the diagonal axis: 
https://docs.scipy.org/doc/numpy/reference/generated/numpy.trace.html
np.trace(np.eye(3))
a = np.arange(8).reshape((2,2,2))
array([6, 8])


# Extract diagonal matrix: 
x = np.arange(9).reshape((3,3))
np.diag(x) => returns vector
np.diag(np.diag(x)) => returns diagonal matrix extracted from a matrix (only elements from diagonal kept)

# macierz jednostkowa: 
a = np.eye(3)


# Tensor product, scalar product:
a = np.array([[1,2,3]]) # wektor wierszowy
b = np.array([[2,1,-1]]) # wektor kolumnowy
print(np.dot(a, b.T)[0]) # => scalar
print(np.dot(a.T, b).T) # => matrix

# Determinant:
a = np.array([[1, 2], [3, 4]])
np.linalg.det(a)

# Rank of matrix:
np.linalg.matrix_rank(np.eye(4)) # Full rank matrix

# Inverse matrix:
np.linalg.inv(np.eye(4))
"""



"""
Coloring matrix: 

import matplotlib.pyplot as plt
from matplotlib import cm

viridis = cm.get_cmap('viridis', 256)
data = np.random.randn(500, 500)
data1 = np.linalg.inv(data)
data3 = np.dot(data, data1)

fig, axs = plt.subplots(1, 3, figsize=(6, 3), constrained_layout=True)
psm = axs[0].pcolormesh(data, cmap=viridis, rasterized=True, vmin=-4, vmax=4)
psm = axs[1].pcolormesh(data1, cmap=viridis, rasterized=True, vmin=-4, vmax=4)
psm = axs[2].pcolormesh(data3, cmap=viridis, rasterized=True, vmin=-4, vmax=4)
plt.show()
"""




