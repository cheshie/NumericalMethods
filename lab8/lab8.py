from collections import namedtuple
from math import sqrt
import matplotlib.pyplot as plt
from numpy import meshgrid, array, linspace, split
from scipy.interpolate import interp2d


# Plot points that were read from diagram during class
def zad_1(st, x=None, y=None):
    if not x and not y:
        x, y = st.x, st.y
    X, Y = meshgrid(x, y)
    ax = plt.axes(projection='3d')
    ax.scatter3D(X, Y, st.z, c=st.z)
    plt.show()


# Plot interpolations for set of points
def zad_2(st):
    fig = plt.figure(figsize=plt.figaspect(1 / 2.))
    fig.suptitle('2d Interpolation techniques')
    axes = [fig.add_subplot(2, 2, 1, projection='3d'),
            fig.add_subplot(2, 2, 2, projection='3d'),
            fig.add_subplot(2, 2, 3, projection='3d'),
            fig.add_subplot(2, 2, 4, projection='3d')]

    # grid from prepared points
    X, Y = meshgrid(st.x, st.y)
    # extended grid used for plotting interpolation
    X_i, Y_i = meshgrid(linspace(st.x[0], st.x[-1], 17), linspace(st.y[0], st.y[-1], 17))
    titles = ['linear']  # , 'cubic']
    # objects used for calculating values of interpolated functions
    interps = [interp2d(X, Y, st.z, kind='linear')]  # , interp2d(X, Y, st.z, kind='cubic')]
    # generate values of interpolation functions for given range of points
    inter_vals = [i(linspace(st.x[0], st.x[-1], 17), linspace(st.y[0], st.y[-1], 17)) for i in interps]

    zad3(interps, titles)

    for i, it in enumerate(inter_vals):
        axes[i].set_title(titles[i])
        print(it.shape)
        axes[i].plot_surface(X_i, Y_i, it, rstride=1, cstride=1, cmap='viridis', edgecolor='none')
        axes[i].scatter3D(X, Y, st.z, c=st.z, cmap='BuPu')

    plt.show()


def zad3(interps, titles):
    # exact solutions - used for comparison of interpolation functions
    exact = [3.5, 5.6, 10.5]
    for i, t in enumerate(titles):
        err = 0
        # for each interpolation function enumeare given points (that correspond to exact values given above)
        # and calculate value of interpolation function for that function
        # lastly - compute quadratic error for given interpolation function
        for j, arg in enumerate([(7, 0.05), (11, 0.03), (15, 0.01)]):
            print("interp: ", t, " value: ", interps[i](arg[0], arg[1]))
            err += pow(interps[i](arg[0], arg[1]) - exact[j], 2)

        print("quadratic error: ", sqrt(err / len(exact)))
        print("-" * 5, "\n\n")


def main():
    X = [1, 5, 9, 13, 17]
    Y = [0.01, 0.02, 0.03, 0.04, 0.05]
    Z = array([[0, 6.3, 8.6, 10.05, 11.05],
               [0, 4.5, 6.2, 7.2, 8],
               [0, 3.8, 5.1, 6, 6.6],
               [0, 3.3, 4.4, 5.3, 5.9],
               [0, 3, 4, 4.7, 5.2]])
    SET1 = namedtuple('set', ['x', 'y', 'z'])(X, Y, Z)
    SET2 = namedtuple('set', ['x', 'y', 'z'])(X[::2], Y[::2],
                                              array(split(array([k for w in Z[::2] for k in w[::2]]), 3)))
    zad_1(SET1)
    # zad_2(SET2)


main()
