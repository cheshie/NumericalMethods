import warnings
from functools import partial
from timeit import timeit

import matplotlib.pyplot as plt
from numpy import array, float64, polyfit, polyval, poly1d
from numpy.linalg import det, slogdet
from numpy.random import uniform

# Sometimes det() causes Runtime OverflowWarning
# So for bigger matrices we will use slogdet() which is more appropriate for larger matrices
# https://docs.scipy.org/doc/numpy/reference/generated/numpy.linalg.slogdet.html#numpy.linalg.slogdet

x = []
y = []
with warnings.catch_warnings():
    warnings.filterwarnings('error')
    for size in range(1, 1000, 10):
        mx = array(uniform(low=-size, high=size, size=(size, size)), dtype=float64)
        x += [size]

        try:
            y+= [timeit(partial(det, mx), number=1)]
        except RuntimeWarning:
            y += [timeit(partial(slogdet, mx), number=1)]  # (sign, logdet) = slogdet(x)

print("len: ", len(x), len(y))

fig, ax = plt.subplots()
pl = poly1d(polyfit(x, y, len(x) - 1))
vals = polyval(pl, x)
ax.plot(x, y, "b*", label='exact solution')
pfit, = plt.plot(x, pl(x), "r-", label='polyfit')
pval, = plt.plot(x, vals, "b--", label='polyval')

ax.set_ylabel('Time')
ax.set_xlabel('Size of matrix')
ax.set_title('Approximating degree of polynomial (time complexity)')
ax.legend()
plt.show()

# Above size 1100 gives following error for
# raise LinAlgError("SVD did not converge in Linear Least Squares")
# numpy.linalg.LinAlgError: SVD did not converge in Linear Least Squares


