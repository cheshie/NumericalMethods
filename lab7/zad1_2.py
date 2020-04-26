from numpy import array, float64, split, linspace, meshgrid, arange, where, abs, append, dstack, zeros, int32
from numpy.linalg import solve, LinAlgError
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


def classification(U, step=0.9, thr : "Setting threshold - this is how much computed result might differ from analytic solution" =0.3):
    # iterators over x and y
    x_range = arange(U['range']['x'][0], U['range']['x'][1], step)
    y_range = arange(U['range']['y'][0], U['range']['y'][1], step)

    # Examples of exact/analytic solutions in python:
    """
    >>> from sympy import symbols, Eq
    >>> from sympy.solvers import solve
    >>>
    >>> x, y = symbols('x y')
    >>> e1 = Eq(2 * x + 8, y)
    >>> e2 = Eq(-5 *x + 9, y)
    >>>
    >>> solve((e1, e2))
    {x: 1/7, y: 58/7}
    >>> 58/7
    8.285714285714286
    
    BUT! If we define:
    >>> e1 = Eq(x ** 2 - y ** 2, 1)
    >>> e2 = Eq(x ** 3 * y ** 2, 1)
    then:
    >>> solve((e1, e2))
    []
    
    Or even: 
    >>> from sympy import sqrt
    >>>
    >>> e1 = Eq(sqrt(x ** 2 - 1), y)
    >>> e2 = Eq(sqrt(1 / x ** 3), y)
    >>>
    >>> solve((e1, e2))
    [{x: CRootOf(y**10 + 3*y**8 + 3*y**6 + y**4 - 1, 1)**(-2/3), y: CRootOf(y**10 + 3*y**8 + 3*y**6 + y**4 - 1, 1)}]
    """

    # Color solutions if found first, second, or no solution found
    COLORS = {'first' : (255, 0, 0), 'second' : (0, 255, 0), 'none' : (0, 0, 0)}

    img = array([], dtype=int32)
    for x in x_range:
        for y in y_range:
            try:
                sol = array(newton(U, x_start=x, y_start=y))
            except LinAlgError:
                # If any error with computations - assign to image no solution (black colour)
                img = append(img, COLORS['none'])
                continue

            # If newton found 1st solution - assign another colour
            if max(abs(sol - U['analytic_sol'][0]))   < thr:
                img = append(img, COLORS['first'])
            # If found the 2nd - assign another colour
            elif max(abs(sol - U['analytic_sol'][1])) < thr:
                img = append(img, COLORS['second'])
            else:
            # If newton was not close to any of the solutions
                img = append(img, COLORS['none'])

    # x size and y size
    xs = len(list(x_range))
    ys = len(list(y_range))
    # create matrix in this matter [ r_matrix[xs, ys], g_matrix[xs, ys], b_matrix[xs, ys] ]
    rgb_matrix = dstack([img[::3].reshape(xs, ys), img[1::3].reshape(xs, ys), img[2::3].reshape(xs, ys)])

    # draw image
    from matplotlib import pyplot as plt
    img = plt.imshow(rgb_matrix, interpolation='nearest')
    img.set_cmap('hot')
    plt.axis('off')
    plt.savefig("test.png", bbox_inches='tight')
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
          'range' : {'x': [0.5, 3], 'y' : [-3, 3]},
           # I was not able to solve these equations analytically with sympy, so I just hardcoded these results
          'analytic_sol' : array([[1.2365, 0.7274], [1.2365, -0.7274]])}

    # Range: x = -6, 6, y = -6, 6
    U2 = {'system' : [F(f=90 * x ** 2 - 25 * y ** 2 - 225, x=x, y=y),
          F(f=9 *  x ** 4 + 25 *  y ** 3 - 50, x=x, y=y)],
          'range' : {'x': [-6, 6], 'y' : [-6,6]},
          'analytic_sol' : array([[-1.6447, -0.8591], [1.6447, -0.8591]])}

    # print(newton(U1, x_start=9, y_start=3))
    # plot_functions(U1)
    classification(U2)

main()


