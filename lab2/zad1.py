from functools import partial
from numpy import linspace, float32, float64, sum, array, arange, abs, subtract
from timeit import timeit
import matplotlib
import matplotlib.pyplot as plt

# Zadanie 1 i 2 

# my_vec = range(1, 10 ** 6)
my_vec = [10 ** 6, 0.2, 0.2, 0.2, -10**6]

def A1(precision):
    sum = precision(0)
    for x in array(my_vec, dtype=precision):
        sum += precision(x)
    return sum
    #

def A2(precision):
    return sum(array(my_vec, dtype=precision))

# Kahan algorithm
# S => result
def A3(precision):
    x = array(my_vec, dtype=precision)
    n = len(x)
    S = precision(x[0])
    C = 0
    for i in range(1, n):
        Y = x[i] - C
        T = S + Y
        C = (T - S) - Y
        S = T
    return S
    #

# Gill-Moller algorithm
# S => result
def A4(precision):
    x = array(my_vec, dtype=precision)
    n = len(x)
    S = 0
    U = 0
    P = 0
    for i in range(n):
        S = U + x[i]
        P = U - S + x[i] + P
        U = S
    return S + P
    #

# Function to find sum of series.
def sumOfAP( a, d,n) :
    sum = 0
    i = 0
    while i < n :
        sum = sum + a
        a = a + d
        i = i + 1
    return sum

sum_correct = 0.6#sumOfAP(1, 1, 10 ** 6)
print("Arithmetic sequence (correct result): ", sum_correct)
print("="*10)
funs = [A1, A2, A3, A4]
res = []
res64 = []
for i, ex in enumerate(funs):
    # print("A"+str(i+1),"-"*10)
    print("Result (single): {:e}".format(ex(float32)))
    # print("Time   (single): {}".format(timeit(partial(ex, float32), number=5)))
    # print("---")
    print("Result (double): {:e}".format(ex(float64)))
    # print("Time   (double): {}".format(timeit(partial(ex, float64), number=5)))
    res.append(ex(float32))
    res64.append(ex(float64))
    # print()

labels = ['A1', 'A2', 'A3', 'A4']
x = arange(len(labels))  # the label locations
width = 0.35  # the width of the bars


fig, ax = plt.subplots()
errors = abs(subtract(res, [sum_correct for x in range(4)]))
errors64 = abs(subtract(res64, [sum_correct for x in range(4)]))
rects1 = ax.bar(x - width/2, errors, width, label='Errors - 32')
rects2 = ax.bar(x + width/2, errors64, width, label='Errors - 64')

# Add some text for labels, title and custom x-axis tick labels, etc.
ax.set_ylabel('Error')
ax.set_title('abs(correct_result - approximate)')
ax.set_xticks(x)
ax.set_xticklabels(labels)
ax.legend()
plt.show()