from numpy import linspace, array, float64, append
import matplotlib.pyplot as plt

vec = linspace(2 - 10 ** (-3), 2 + 10 ** (-3), 1000)

f1 = array([], dtype=float64)
f2 = array([], dtype=float64)
blad = array([], dtype=float64)
for x in vec:
    a = (x - 2)
    w1 = a ** 4
    f1 = append(f1, w1)

    w2 = x ** 4 - 8 * x ** 3 + 24 * x ** 2 - 32 * x + 16
    f2 = append(f2, w2)
    blad = append(blad, abs(w1 - w2))

plt.plot(vec, f1)
plt.plot(vec, f2, 'r')
plt.plot(vec, blad, 'g')
plt.show()