from functools import partial
from timeit import timeit

import scipy.io
from numpy import array, diag, prod, where, float64
from numpy.linalg import det
from chio import chio_det
from echeon import echelon_det

# https://socratic.org/questions/how-do-i-find-the-determinant-of-a-matrix-using-row-echelon-form
# from https://gist.github.com/braingineer/d801735dac07ff3ac4d746e1f218ab75
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


mxes    = [array(scipy.io.loadmat('A_50.mat')['A_50'], dtype=float64),
           array(scipy.io.loadmat('A_100.mat')['A_100'], dtype=float64),
           array(scipy.io.loadmat('A_200.mat')['A_200'], dtype=float64)]
methods = [det, chio_det, echelon_det]

for mx in mxes:
    for m in methods:
        print("time: ", timeit(partial(m, mx), number=1))
        print("Result: {:e}".format(m(mx)))

