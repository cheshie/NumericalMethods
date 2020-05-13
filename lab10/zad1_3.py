from numpy import array, float64, split, linspace, meshgrid, arange, where, abs, append, dstack, zeros, int32, power, sum
from numpy.linalg import solve, LinAlgError
from sympy import symbols, lambdify, cos, N, sin, diff
from collections import namedtuple
import matplotlib.pyplot as plt

# Wzor dwupunktowy
def num_2nd(f, x, h=0.001):
    return (N(f.f.subs(f.x, x + h)) - N(f.f.subs(f.x, x))) / h

# Wzor trojpunktowy
def num_3rd(f, x, h=0.001):
    return (N(f.f.subs(f.x, x + h)) - N(f.f.subs(f.x, x - h))) / (2 * h)

# Wzor pieciopunktowy
def num_5th(f, x, h=0.001):
    return 1/(12*h) *  (N(f.f.subs(f.x, x - 2*h)) - 8 * N(f.f.subs(f.x, x - h)) + 8 * N(f.f.subs(f.x, x + h)) - N(f.f.subs(f.x, x + 2*h)))

# druga pochodna
# Wzor trojpunktowy
def num2_3rd(f, x, h=0.001):
    return (1 / h ** 2) * N(f.f.subs(f.x, x + h)) - 2 * N(f.f.subs(f.x, x)) + N(f.f.subs(f.x, x - h))
# Wzor pieciopunktowy
def num2_5th(f, x, h=0.001):
    return 1/(12*h ** 2) *  (-N(f.f.subs(f.x, x - 2*h)) + 16 * N(f.f.subs(f.x, x - h)) - 30 * N(f.f.subs(f.x, x)) + 16 * N(f.f.subs(f.x, x+h)) - N(f.f.subs(f.x, x - 2*h)))

def plot_loop(axes, derivs, f1, methods, steps_values):
    ax_nr = 0
    for d, m in zip(derivs, methods):
        for h in steps_values:
            x_analytic = [x for x in arange(*f1.range, h)]
            y_analytic = [N(diff(f1.f, f1.x).subs(f1.x, x_val)) for x_val in x_analytic]
            x_vec = [x for x in arange(*f1.range, h)]
            y_vec = [d(f1, x_v, h) for x_v in x_vec]
            axes[ax_nr].plot(x_vec, y_vec)
            axes[ax_nr].plot(x_analytic, y_analytic, 'orange')
            axes[ax_nr].set_title(f'method: {m}, step: {h}')
            print(f"Err ({m}, {h}):", 1 / len(arange(*f1.range, h)) * sum(power(array(y_analytic) - array(y_vec), 2)))
            ax_nr += 1
    plt.show()

def zad1():
    # Declare special struct and symbols
    # f: equation, x: x symbol, y: y symbol (sympy library)
    F = namedtuple('F', ['f', 'x', 'range'])
    x, y = symbols('x y')

    f1 = F(f=cos(2 * x), x=x, range=(0,6))
    steps_values = [1, 0.5, 0.25]
    derivs = [num_2nd, num_3rd, num_5th]
    methods = ['num_2nd', 'num_3rd', 'num_5th']

    # Create figure with 3d plot (ax1) and contour plot (ax2)
    fig = plt.figure(figsize=plt.figaspect(1 / 2.))
    fig.suptitle('Derivatives')
    axes = [fig.add_subplot(3, 3, x) for x in range(1,10)]

    plot_loop(axes, derivs, f1, methods, steps_values)

def zad3():
    # Declare special struct and symbols
    # f: equation, x: x symbol, y: y symbol (sympy library)
    F = namedtuple('F', ['f', 'x', 'range'])
    x, y = symbols('x y')

    f1 = F(f=cos(4 * x) - (1/2) * x, x=x, range=(0,5))
    steps_values = [1, 0.5, 0.25]
    derivs = [num2_3rd, num2_5th]
    methods = ['num_3rd', 'num_5th']

    # Create figure with 3d plot (ax1) and contour plot (ax2)
    fig = plt.figure(figsize=plt.figaspect(1 / 2.))
    fig.suptitle('Derivatives 2nd')
    axes = [fig.add_subplot(3, 3, x) for x in range(1,7)]

    plot_loop(axes, derivs, f1, methods, steps_values)

    
zad3()


