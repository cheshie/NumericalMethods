from functools import partial
from numpy import linspace, float32, float64, sum, array, arange, abs, subtract, exp, pi, sqrt, log10, dot
from scipy.special import psi
from timeit import timeit
import matplotlib
import matplotlib.pyplot as plt


X = array([exp(1), -pi, sqrt(2), -psi(1), log10(2)], dtype=float64)
Y = array([1486.2497, 878366.9879, -22.37492, 4773714.647, 0.000185049], dtype=float64)
rs = []

# W1 Monozenie skalarne wektorow
rs.append(dot(X, Y.T))
print("Result: {:e}".format(rs[0]))

# W2 Sumowanie ilocz z sum()
rs.append(sum(X * Y))
print("Result: {:e}".format(rs[1]))

# W3 Sumowanie z iloczynem
sum = 0
for x in zip(X,Y):
    sum += x[0] * x[1]

rs.append(sum)
print("Result: {:e}".format(rs[2]))

# W4 Sumowanie z iloczynem od tylu
sum = 0
for x in zip(X[::-1],Y[::-1]):
    sum += x[0] * x[1]

rs.append(sum)
print("Result: {:e}".format(rs[3]))

# W5 sumowanie parzystych i nieparzystych
sum = 0
for x in zip(X[::2],Y[::2]):
    sum += x[0] * x[1]
for x in zip(X[1::2],Y[1::2]):
    sum += x[0] * x[1]
rs.append(sum)
print("Result: {:e}".format(rs[4]))

labels = ['W1', 'W2', 'W3', 'W4', 'W5']
x = arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
errors = abs(subtract(rs, [rs[0] for x in range(5)]))
rects1 = ax.bar(x - width/2, errors, width, label='Errors - 64')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('abs(correct_result - approximate)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()