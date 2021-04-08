import numpy as np
from utils import Gaussian_data_gen, input_poly_basis


def poly_data_gen(n, a, w):
    x = np.random.uniform(-1.0, 1.0)
    X = np.array([(x ** deg) for deg in range(n)]).reshape(-1)
    W = np.array(w).reshape(-1)
    e = Gaussian_data_gen(0, a)
    return x, np.dot(np.transpose(W), X) + e


if __name__ == '__main__':

    while True:
        toggle = input('Univariate Gaussian (0) or Polynomial basis linear mode (1): ')

        # Univariate Gaussian data generator
        if toggle == '0':
            mean = float(input("Input mean: "))
            variance = float(input("Input variance: "))
            rand_num = Gaussian_data_gen(mean, variance)
            print(rand_num)

        # Polynomial basis linear model data generator
        elif toggle == '1':
            n, a, w = input_poly_basis()
            x, y = poly_data_gen(n, a, w)
            print(y)
