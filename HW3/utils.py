import numpy as np


def Gaussian_data_gen(mean,variance):
    """
    Gaussian_data_gen: generate standard Gaussian data with given mean and variance via Box Muller method
    :param mean: Gaussian mean
    :param variance: Gaussian variance
    :return: number generate from standard Gaussian
    """
    U = np.random.uniform(0,1)
    V = np.random.uniform(0,1)
    X=np.sqrt(-2*np.log(U))*np.cos(2*np.pi*V)
    rand_num = mean + np.sqrt(variance) * X
    return rand_num