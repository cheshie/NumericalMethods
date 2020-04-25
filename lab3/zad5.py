from numpy import array, float64, set_printoptions, diag, dot
from numpy.linalg import det, eig, inv

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
A = array([[1, 1, 1], [2, 1, -1], [2, 0, 2]], dtype=float64)
print("Macierz A:")
matprint(A)

# 2) Wartosci wlasne i wektory wlasne:
vals, vecs = eig(A)
print("Wartosci wlasne A: ", vals)
print("Wektory wlasne A: ", vecs)

# 3) Wyznacznik:
print("Wyznacznik A: ", det(A))

# 4) Macierz podobna do A:
# AT = TB or taken from wikipedia: B = P^(-1) * A * P
# https://en.wikipedia.org/wiki/Matrix_similarity
B = dot(dot(inv(vecs), A), vecs)
print("Macierz podobna B: \n", B)

# 5) Wartosci na glownej przekatnej:
print("Wartosci diagonalne B: ", diag(B))

# 6)
print("Wyznacznik B: ", det(B))



