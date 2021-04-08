import numpy as np


def Gaussian_data_gen(mean, variance):
    """
    Gaussian_data_gen: generate standard Gaussian data with given mean and variance via Box Muller method
    :param mean: Gaussian mean
    :param variance: Gaussian variance
    :return: number generate from standard Gaussian
    """
    # box-muller
    u = np.random.uniform(0.0, 1.0)
    v = np.random.uniform(0.0, 1.0)
    z = np.sqrt(-2.0 * np.log(u)) * np.sin(2 * np.pi * v)
    return mean + np.sqrt(variance) * z


def input_poly_basis():
    n = int(input("Input n: "))
    a = float(input("Input a: "))
    w_str = input("Input w: ")
    w = list(map(float, w_str.split()))
    w = np.array(w).reshape(n, -1)  # [n,1]
    return n, a, w
