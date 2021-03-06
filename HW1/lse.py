from utils import *


# (A^T*A+LAMBDA*I)^(-1)*A^Tb
def lse(A, b, _lambda):
    """lse
    Implementing rLSE algorithm for line fitting
    :param A: power of x matrix
    :param b: original data points y axis
    :param _lambda: regularization term
    :return: prediction of coefficient of the line fitting function
    """
    # matrix transpose
    AT = transpose(A)
    # matrix multiplication
    ATA = matrix_multiplication(AT, A)

    # identity matrix
    I = [[0 for y in range(len(ATA))] for x in range(len(ATA))]
    for i in range(len(ATA)):
        for j in range(len(ATA)):
            if i == j:
                I[i][j] = _lambda

    # matrix addition
    ATAlI = [[0 for y in range(len(ATA))] for x in range(len(ATA))]
    for i in range(len(ATA)):
        for j in range(len(ATA)):
            ATAlI[i][j] = ATA[i][j] + I[i][j]

    # LU decomposition
    upper, lower = LU_decomposition(ATAlI)

    # solve y: Ly=B
    ATAlI_inv = inverse(upper, lower)

    # (A^T*A+LAMBDA*I)^(-1)*A^Tb
    x = matrix_multiplication(ATAlI_inv, AT)
    x = matrix_multiplication(x, b)
    return x
