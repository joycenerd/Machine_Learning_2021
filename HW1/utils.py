def transpose(mat):
    """transpose
    matrix transpose A[i][j]=A.T[j][i]
    :param mat: original matrix
    :return: matrix after transpose
    """
    mat_t = [[0 for y in range(len(mat))] for x in range(len(mat[0]))]
    for i in range(len(mat)):
        for j in range(len(mat[0])):
            mat_t[j][i] = mat[i][j]
    return mat_t


def matrix_multiplication(a, b):
    """matrix_multiplication
    :param a: multiplicand matrix
    :param b: multiplier matrix
    :return: a multiplicated matrix
    """
    c = [[0 for y in range(len(b[0]))] for x in range(len(a))]
    for i in range(len(a)):
        for j in range(len(b[0])):
            for k in range(len(b)):
                c[i][j] += a[i][k] * b[k][j]
    return c


def LU_decomposition(mat):
    """LU_decomposition
    decompose original matrix to upper triangular matrix and lower triangular matrix: mat=upper*lower
    :param mat: original matrix
    :return: upper and lower triangular matrix
    """
    n = len(mat)
    lower = [[0 for y in range(n)] for x in range(n)]
    upper = [[0 for y in range(n)] for x in range(n)]
    # decomposing matrix into upper and lower triangular matrix
    for i in range(n):
        # upper triangular
        for k in range(i, n):
            # summation of L(i,j)*U(j,k)
            sum = 0
            for j in range(i):
                sum += lower[i][j] * upper[j][k]
            # evaluating U(i,k)
            upper[i][k] = mat[i][k] - sum
        # lower triangular
        for k in range(i, n):
            if i == k:
                lower[i][i] = 1  # diagonal as 1
            else:
                # summation of L(k,j)*U(j,i)
                sum = 0
                for j in range(i):
                    sum += lower[k][j] * upper[j][i]
                lower[k][i] = (mat[k][i] - sum) / upper[i][i]
    return upper, lower


def inverse(upper, lower):
    """inverse
    Solve inversion matrix by solving upper and lower matrix
    :param upper: upper triangular matrix
    :param lower: lower triangular matrix
    :return: inversion matrix
    """
    # solve y: Ly=B
    n = len(lower)
    y = []
    for i in range(n):
        y.append([0] * n)
    for i in range(0, n):
        for j in range(0, n):
            if i == j:
                B = 1.0
            else:
                B = 0.0
            for k in range(0, j):
                B -= lower[j][k] * y[k][i]
            y[j][i] = B

    # solve inverse: U*inverse=y
    n = len(upper)
    inv = []
    for i in range(n):
        inv.append([0] * n)
    for i in range(n - 1, -1, -1):
        for j in range(n - 1, -1, -1):
            temp = y[j][i]
            for k in range(j + 1, n):
                temp -= upper[j][k] * inv[k][i]
            inv[j][i] = temp / upper[j][j]
    return inv
