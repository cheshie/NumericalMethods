from numpy import array, float64, dot, transpose, zeros, fill_diagonal, diag
from numpy.linalg import inv, solve, cond
from scipy.linalg import lu, qr, det, svd
from gauss import gauss

# Miscellaneous to pretty print matrices
def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")
#

# Source: rosettacode.org/wiki/Cramer%27s_rule#Python
def cramer(A, B, C):
    X = []
    for i in range(0, len(B)):
        for j in range(0, len(B)):
            C[j][i] = B[j]
            if i > 0:
                C[j][i - 1] = A[j][i - 1]
        X.append(round(det(C) / det(A), 1))

    return X
#

X1_exact = array([2.0, 1.0], dtype=float64)

# Data : matrices:
A1 = array([[1, 2], [1, 2 + 0.01]] , dtype=float64)
A2 = array([[1, 2], [1, 2 + 0.0001]] , dtype=float64)

b1 = array([[4], [4 + 0.01]] , dtype=float64)
b2 = array([[4], [4 + 0.0001]] , dtype=float64)

print("COND: ")
print("A1", cond(A1))
print("A2", cond(A2))

# Cramer's rule :
X1 = cramer(A1.copy(), b1.copy(), A1.copy())
X2 = cramer(A2.copy(), b2.copy(), A2.copy())
print("Cramer's rule: ")
print('1-SET -> Cramer: x1={:e}'.format(X1[0]), 'x2={:e}'.format(X1[1]))
print("Error: ", abs(X1_exact - X1))
print('2-SET -> Cramer: x1={:e}'.format(X2[0]), 'x2={:e}'.format(X2[1]))
print("Error: ", abs(X1_exact - X2))
print("-"*10,"\n")

# Matrix inverse:
X1 = dot(inv(A1), b1)
X2 = dot(inv(A2), b2)
print("Matrix inverse: ")
print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("-"*10,"\n")


# Gauss elimination:
X1 = gauss(A1.copy(), b1.copy())
X2 = gauss(A2.copy(), b2.copy())
print("Gauss: ")
print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("-"*10,"Using NumPy:","\n")
X1 = solve(A1.copy(), b1.copy())
X2 = solve(A2.copy(), b2.copy())
print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("-"*10,"\n")


# LU Decomposition:
# P is permutation matrix
P, L, U = lu(A1)
X1 = dot(dot(inv(U), inv(L)), b1.copy())
P, L, U = lu(A2)
X2 = dot(dot(inv(U), inv(L)), b2.copy())
print("LU Decomposition - version 1: ")
print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("LU Decomposition - version 2: ")
P, L, U = lu(A1)
y = dot(inv(L), b1)
X1 = dot(inv(U), y)

P, L, U = lu(A2)
y = dot(inv(L), b2)
X2 = dot(inv(U), y)
print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("-"*10,"\n")

# QR Decomposition:
Q, R = qr(A1)
X1 = dot(dot(inv(R), transpose(Q)), b1.copy())
Q, R = qr(A2)
X2 = dot(dot(inv(R), transpose(Q)), b2.copy())
print("QR Decomposition - version 1: ")
print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("QR Decomposition - version 2: ")
Q, R = qr(A1)
y = dot(transpose(Q), b1)
X1 = dot(inv(R), y)

Q, R = qr(A2)
y = dot(transpose(Q), b2)
X2 = dot(inv(R), y)

print('1-SET -> Cramer: x1={:e}'.format(X1[0][0]), 'x2={:e}'.format(X1[1][0]))
print("Error: ", abs(X1_exact - transpose(X1)))
print('2-SET -> Cramer: x1={:e}'.format(X2[0][0]), 'x2={:e}'.format(X2[1][0]))
print("Error: ", abs(X1_exact - transpose(X2)))
print("-"*10,"\n")

# SVD (Singular Value) Decomposition:
# Source: https://personalpages.manchester.ac.uk/staff/timothy.f.cootes/MathsMethodsNotes/L3_linear_algebra3.pdf
U, W, V = svd(A1)
x1 = dot(dot(dot(V, inv(diag(W))), transpose(U)), b1)
U, W, V = svd(A2)
x2 = dot(dot(dot(V, inv(diag(W))), transpose(U)), b2)

print("SVD Decomposition: ")
print('1-SET -> Cramer: x1={:e}'.format(x1[0][0]), 'x2={:e}'.format(x1[1][0]))
print("Error: ", abs(X1_exact - transpose(x1)))
print('1-SET -> Cramer: x1={:e}'.format(x2[0][0]), 'x2={:e}'.format(x2[1][0]))
print("Error: ", abs(X1_exact - transpose(x2)))
print("-"*10,"\n")