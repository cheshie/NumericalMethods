# https://rosettacode.org/wiki/Gaussian_elimination#Using_numpy
def echelon_det(a):
    a = a.copy()
    n = len(a)

    det = 1
    for i in range(n - 1):
        k = i
        for j in range(i + 1, n):
            if abs(a[j][i]) > abs(a[k][i]):
                k = j
        if k != i:
            a[i], a[k] = a[k], a[i]
            det = -det

        for j in range(i + 1, n):
            t = a[j][i] / a[i][i]
            for k in range(i + 1, n):
                a[j][k] -= t * a[i][k]

    for i in range(n - 1, -1, -1):
        det *= a[i][i]

    return det
#

"""
Tutaj jest pierwsza wersja (pisana z glowy), dziala idealnie dla przykladu
z pdfa, dla zadnej innej macierzy niestety nie

def echelon_determinant(a):
    w, k = a.shape
    assert w == k
    det = 1
    # Iterate over all rows
    for i in range(1, w):
        # Iterate all rows except current
        for wiersz in set(map(tuple, a)) - {tuple(a[i])}:
            # Iterate over elements in a row
            # break if current element greater than current row. This way, in second row 1st element is intreseting
            # in third both 1st and 2nd elements are interesting etc.
            for el in range(i):
                if array(wiersz)[el] != 0:
                    # Break if current element equals 0. If will be at the last iteration
                    # Of algorithm
                    if a[i][el] == 0:
                        break
                    # calculate element that the current row will be divided by
                    mmm = array(wiersz)[el] / a[i][el]
                    # divide current row by that divided element
                    print("array: ")
                    matprint(a)
                    print("row: ", array(wiersz))
                    print("x: ", where(a == array(wiersz)))
                    a[where(a == array(wiersz))[0][0]] /= mmm
                    # substract rows
                    a[i] = a[i] - (array(wiersz) / mmm)
                    # update det
                    det *= mmm

    return det * prod(diag(a))
#
"""