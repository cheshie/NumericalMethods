from numpy import float32, float64
from math import sqrt


for k in range(4, 10 + 1):
    x = float64(10 ** k)
    a = float64(sqrt(1 + x ** 2))
    w1 = float64(x - a)
    print("{:e}".format(w1))

print("\n\n--------------------------\n\n")

for k in range(4, 10 + 1):
    x = float64(10 ** k)
    a = float64(sqrt(1 + x ** 2))
    w2 = float64(-1 / (x + a))
    print("{:e}".format(w2))

print("\n\n--------------------------")
print("--------------------------\n\n")

for k in range(4, 10 + 1):
    x = float32(10 ** k)
    a = float32(sqrt(1 + x ** 2))
    w1 = float32(x - a)
    print("{:e}".format(w1))

print("\n\n--------------------------\n\n")

for k in range(4, 10 + 1):
    x = float32(10 ** k)
    a = float32(sqrt(1 + x ** 2))
    w2 = float32(-1 / (x + a))
    print("{:e}".format(w2))


