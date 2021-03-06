from utils import *


def matrix_subtraction(a, b):
    """matrix_subtraction
    Matrix subtraction a[i][j]-b[i][j] for all i j
    :param a: first matrix
    :param b: second matrix
    :return: first-second result
    """
    c = [[0 for y in range(len(a[0]))] for x in range(len(a))]
    for i in range(len(a)):
        for j in range(len(a[0])):
            c[i][j] = a[i][j] - b[i][j]
    return c


def multiply_scalar(x, scalar):
    """multiply_scalar
    matrix multiply scalar for each element
    :param x: matrix
    :param scalar: scalar to multiply matrix
    :return: matrix after multiplying scalar
    """
    for i in range(len(x)):
        for j in range(len(x[0])):
            x[i][j] = scalar * x[i][j]
    return x


def newton(A, b, n):
    """newton
    Use Newton's method to get line fitting function coefficient
    :param A: power of x matrix
    :param b: original data points y axis
    :param n: polynomial bases
    :return: prediction of coefficient of the line fitting function
    """
    x = [[0.0] for y in range(n)]
    AT = transpose(A)
    ATb = matrix_multiplication(AT, b)
    ATA = matrix_multiplication(AT, A)

    # gradient: 2A.TAx-2A.Tb
    ATAx = matrix_multiplication(ATA, x)
    gradient = matrix_subtraction(multiply_scalar(ATAx, 2), multiply_scalar(ATb, 2))

    # Hessian: (2A.TA)^-1
    hessian = multiply_scalar(ATA, 2)
    upper, lower = LU_decomposition(hessian)
    hessian_inv = inverse(upper, lower)
    x = matrix_subtraction(x, matrix_multiplication(hessian_inv, gradient))
    return x
