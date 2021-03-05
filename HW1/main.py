import numpy as np
from lse import lse


if __name__=='__main__':
    f=open('testfile.txt','r').readlines()
    X=[]
    Y=[]
    for i in range(0,len(f)):
        w=f[i].split(',')
        X.append(float(w[0]))
        Y.append(float(w[1]))


    cases=1
    while True:
        print("Case#",cases)
        print("Input polynomial bases n: ")
        n=int(input())
        print("Input lambda: ")
        _lambda=float(input())

        A=[]
        b=[]
        for i in range(0,len(X)):
            a=[]
            for exp in range(n-1,-1,-1):
                a.append(pow(X[i],exp))
            A.append(a)
            b.append(Y[i])

        lse(A,b,_lambda)