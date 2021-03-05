import numpy as np


# (A^T*A+LAMBDA*I)^(-1)*A^Tb
def lse(A,b,_lambda):

    # matrix transpose
    AT=[[0 for y in range(len(A))] for x in range(len(A[0]))]
    for i in range(len(A)):
        for j in range(len(A[0])):
            AT[j][i]=A[i][j]

    # matrix multiplication
    ATA=[[0 for y in range(len(A[0]))] for x in range(len(AT))]
    for i in range(len(AT)):
        for j in range(len(A[0])):
            for k in range(len(A)):
                ATA[i][j]+=AT[i][k]*A[k][j]

    # identity matrix
    I=[[0 for y in range(len(ATA))] for x in range(len(ATA))]
    for i in range(len(ATA)):
        for j in range(len(ATA)):
            if i==j:
                I[i][j]=_lambda

    # matrix addition
    ATAlI=[[0 for y in range(len(ATA))] for x in range(len(ATA))]
    for i in range(len(ATA)):
        for j in range(len(ATA)):
            ATAlI[i][j]=ATA[i][j]+I[i][j]