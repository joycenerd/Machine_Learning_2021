import numpy as np
from utils import Gaussian_data_gen


if __name__=='__main__':

    while True:
        toggle=input('Univariate Gaussian (0) or Polynomial basis linear mode (1): ')

        # Univariate Gaussian data generator
        if toggle=='0':
            mean=float(input("Input mean: "))
            variance=float(input("Input variance: "))
            rand_num=Gaussian_data_gen(mean,variance)
            print(rand_num)

        # Polynomial basis linear model data generator
        elif toggle=='1':
            n=int(input("Input n: "))
            a=float(input("Input a: "))
            w_str=input("Input w: ")
            w=list(map(float,w_str.split()))
            w=np.array(w).reshape(n,-1) # [n,1]
            x=np.random.uniform(-1.0,1.0)
            X=np.array([x**i for i in range(n)]).reshape(1,n) # [1,n]
            e=Gaussian_data_gen(0,a)
            y=X@w+e
            y=y.squeeze().squeeze()
            print(y)


