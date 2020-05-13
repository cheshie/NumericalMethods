from array import array
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from numpy import array as ar
from numpy import power as pwr, sum

# Here we do not have functions to evaluate, just a set of points
# and corresponding values of some function

# 3 points central
from sympy import symbols, cos, sin, N, diff, power


def num_3rd_c(f_k1, f_k_1, h):
    return (f_k1 - f_k_1) / (2*h)

def num_3rd_r(f_k, f_k1, f_k2, h):
    return (-3 * f_k + 4 * f_k1 - f_k2) / (2 * h)

def num_3rd_l(f_k_2, f_k_1, f_k, h):
    return (f_k_2 - 4 * f_k_1 - 3 * f_k) / (2 * h)


def num_deriv(x, y):
    # vectors must have the same length
    assert len(x) == len(y)

    y_deriv = array('f', [])
    for i in range(len(y)):
        if i < 1:
            y_deriv.append(num_3rd_r(y[i], y[i + 1], y[i + 2], abs(x[i+1] - x[i])))
        elif i >= 1 and i < len(y) - 1:
            y_deriv.append(num_3rd_c(y[i+1], y[i-1], abs(x[i + 1] - x[i])))
        else:
            y_deriv.append(num_3rd_l(y[i-2], y[i - 1], y[i], abs(x[i - 1] - x[i])))

    return y_deriv
#

x = eval(','.join("[0	0.500000000000000	1	1.50000000000000	2	2.50000000000000	3	3.50000000000000	4]".split()))
y = eval(','.join("[1	1.01972784447234	0.425324148260754	0.00750249000360903	0.255653805962070	0.882134329567183	1.10129029471023	0.403119026653685	-0.902302529116542]".split()))
y_deriv = num_deriv(x, y)

x_1 = eval(','.join("[0	0.200000000000000	0.400000000000000	0.600000000000000	0.800000000000000	1	1.20000000000000	1.40000000000000	1.60000000000000	"
                    "1.80000000000000	2	2.20000000000000	2.40000000000000	2.60000000000000	2.80000000000000	3	3.20000000000000	3.40000000000000	"
                    "3.60000000000000	3.80000000000000	4]".split()))
y_1 = eval(','.join("[1	1.11973032479795	1.08612505165582	0.927000227871709	0.688156568598234	0.425324148260754	0.194645370425981	0.0432273893198020	"
                    "0.00127882724675199	0.0770892145440482	0.255653805962070	0.501163533841171	0.762962163990597	0.984018043121841	1.11055402866615	"
                    "1.10129029471023	0.934810775330613	0.613856388322994	0.165830871237402				-0.360598048360463	-0.902302529116542]".split()))

y_deriv1 = num_deriv(x_1, y_1)


x_s, y_s = symbols('x y')
f1 = sin(x_s) + cos(2*x_s)
y_an = [N(diff(f1, x_s).subs(x_s, x_v)) for x_v in x]
y_an1 = [N(diff(f1, x_s).subs(x_s, x_v)) for x_v in x_1]

# Create figure with 3d plot (ax1) and contour plot (ax2)
fig = plt.figure(figsize=plt.figaspect(1 / 2.))
fig.suptitle('Derivatives')
ax1 = fig.add_subplot(1,2, 1)
ax2 = fig.add_subplot(1,2, 2)

print("Err 1:", (1 / len(x)) * sum(pwr(ar(y_an) - ar(y_deriv), 2)))
print("Err 2:", (1 / len(x)) * sum(pwr(ar(y_an1) - ar(y_deriv1), 2)))


blue_patch   = mpatches.Patch(color='blue', label='Derivative')
orange_patch = mpatches.Patch(color='orange', label='Exact solution')
ax1.legend(handles=[blue_patch, orange_patch])
ax2.legend(handles=[blue_patch, orange_patch])

ax1.plot(x, y_deriv)
ax1.plot(x, y_an, 'orange')

ax2.plot(x_1, y_deriv1)
ax2.plot(x_1, y_an1, 'orange')

plt.show()