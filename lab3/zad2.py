from numpy import array, triu, tril, trace
from numpy.linalg import det, eig, inv, matrix_rank
from numpy import array, triu, tril, trace, float64, cumprod, poly, roots, set_printoptions, sum
from numpy.linalg import det, eig, inv, eigvals
from itertools import accumulate

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
A = array([[1,1,2], [2,1,4], [3,0,6]], dtype=float64)
print("Macierz A: \n")
matprint(A)
# 2)
print("Rzad macierzy A:    ", matrix_rank(A))
# 3)
print("Wartosci wlasne: \n", eigvals(A))
# 4)
print("Suma wartosci wlasnych: ", sum(eigvals(A)))#list(accumulate(eig(A)[-1], lambda a,b: a+b))[-1])
# 5) Wspolczynniki wielomianu charakterystycznego:
print("wielomian charakterystyczny z A: ", poly(A))
# 6) Pierwiastki
print("Pierwiastki z A: ", roots(poly(A)))
# 7) Porownac wyniki z 3 i 6
# 8) Slad macierzy A:
print("Slad z A: ", trace(A))
# 9) Porownac wyniki 4 i 8








