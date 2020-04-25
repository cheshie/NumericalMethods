from numpy import array, float64, split, linspace, meshgrid, arange, where, abs
from numpy.linalg import solve
from sympy import symbols, lambdify
from collections import namedtuple
import matplotlib.pyplot as plt

def num_dfx(f, x, y, h=0.001):
    return (f.f.subs({f.x : x + h, f.y : y}) - f.f.subs({f.x : x - h, f.y : y})) / (2 * h)

def num_dfy(f, x, y, h=0.001):
    return (f.f.subs({f.x : x, f.y : y + h}) - f.f.subs({f.x : x, f.y : y - h})) / (2 * h)
# Exact solution i.e. f.diff(x).subs({x : 2, y : 2})


def plot_functions(U):
    x = linspace(*U['range']['x'], 100)
    y = linspace(*U['range']['y'], 100)
    X, Y = meshgrid(x, y)

    # Create figure with 3d plot (ax1) and contour plot (ax2)
    fig = plt.figure(figsize=plt.figaspect(1/2.))
    fig.suptitle('3D and contour diagrams of the system')
    ax1 = fig.add_subplot(1,2,1, projection='3d')
    ax2 = fig.add_subplot(1,2,2)

    # Iterate over equations in the system
    for fun in U['system']:
        # Create equation from sympy function
        F = lambdify((fun.x, fun.y), fun.f, 'numpy')
        # Calculate value
        Z = F(X, Y)
        # Print 3d
        ax1.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='viridis')
        # print contour
        C1 = ax2.contour(X, Y, Z)
        # set contour labels
        ax2.clabel(C1, inline=1, fontsize=10)

        # Choose points where function is very close to 0 and draw them
        X_0, Y_0 = where(abs(Z) < 0.1)
        # ax2.plot(Z[X_0, Y_0], 'r*')

    ax1.set_xlabel('x')
    ax1.set_ylabel('y')
    ax1.set_zlabel('z')
    plt.show()
#


# Solve system of non linear equations
def newton(U : "System of equations", h : "derivative step" = 0.001 , e : "Solution threshold" =0.005,
           x_start : "x - start point"=0.8, y_start : "y start point"=1.5, equations_nr : "number of equations in the system"=2):
    derivs = [num_dfx, num_dfy]
    X_prev = array([0, 0])

    while True:
        # Calculate values of F(x) in current point
        fx = array([F.f.subs({F.x : x_start, F.y : y_start}) for F in U['system']], dtype=float64)
        # Calculate values of dF(x) in current point
        J = array([d(F, x_start, y_start, h=h) for d in derivs for F in U['system']], dtype=float64).reshape((equations_nr, equations_nr), order='F')
        # Solve the system - solution is difference between next and current point
        X = solve(J, fx)

        # Check if the difference (offset) is smaller than our threshold (e), we must be close to the solution
        if max(abs(X - X_prev)) < e and abs(fx[0]) < 0.01 and abs(fx[1]) < 0.01:
            break

        # move points to the next offset
        x_start -= X[0]
        y_start -= X[1]

        X_prev = X

    return x_start, y_start
#


def main():
    # Declare special struct and symbols
    # f: equation, x: x symbol, y: y symbol (sympy library)
    F = namedtuple('F', ['f', 'x', 'y'])
    x, y = symbols('x y')

    # Range: x = 0.5, 3, y = -3, 3
    U1 = {'system': [F(f=x ** 2 - y ** 2 - 1, x=x, y=y),
          F(f=x ** 3 * y ** 2 - 1, x=x, y=y)],
          'range' : {'x': [0.5, 3], 'y' : [-3, 3]}}

    # Range: x = -6, 6, y = -6, 6
    U2 = {'system' : [F(f=90 * x ** 2 - 25 * y ** 2 - 225, x=x, y=y),
          F(f=9 *  x ** 4 + 25 *  y ** 3 - 50, x=x, y=y)],
          'range' : {'x': [-6, 6], 'y' : [-6,6]}}

    # print(newton(U1, x_start=9, y_start=3))
    plot_functions(U1)

main()


