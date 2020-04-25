from numpy import array, triu, tril, trace, float64, set_printoptions, prod
from numpy.linalg import det, inv, eigvals

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
A = array([[-5, -100, 0, 0], [0.5, -1.15, -1.1, 0.15], [0, 0, -0.04, 0], [0, 0.0055, 0.0036, -0.0063]], dtype=float64)
print("Macierz A: \n")
matprint(A)
# 2)
print("Wyznacznik A:    ", det(A))
# 3)
print("Wartosci wlasne: \n", eigvals(A))
# 4)
print("Iloczyn wartosci wlasnych: ", prod(eigvals(A)))#list(accumulate(eig(A)[-1], lambda a,b: a*b))[-1])
# 5)
print("Wyznacz macierz odwrotna do A: ")
matprint(inv(A))
# 6)
print("Macierz gorna trójkatna: ")
matprint(triu(A))
print("Macierz dolna trójkatna: ")
matprint(tril(A))
# 7) Slad macierzy A
print("Slad macierzy A: ", trace(A))








