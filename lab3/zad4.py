import matplotlib.pyplot as plt
from numpy import array, float64, poly, roots, set_printoptions, real, imag
from numpy.linalg import eig

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
A = array([[0,1], [-2,-3]], dtype=float64)
print("Macierz A:")
matprint(A)

# 2) Wartosci wlasne i wektory wlasne:
vals, vecs = eig(A)
print("Wartosci wlasne A: ", vals)
print("Wektory wlasne A: ", vecs)

exit()
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