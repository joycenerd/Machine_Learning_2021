import numpy as np
import matplotlib.pyplot as plt


def data_gen(mean, variance, n):
    data = np.zeros(n)
    for i in range(n):
        u = np.random.uniform(0.0, 1.0)
        v = np.random.uniform(0.0, 1.0)
        z = np.sqrt(-2.0 * np.log(u)) * np.sin(2 * np.pi * v)
        data[i] = mean + np.sqrt(variance) * z
    return data


def gradient_descent(num_iter, X, Y, X_mat):
    w = data_gen(0.0, 1.0, 3)
    for i in range(num_iter):
        # sigmoid, forward, backward


if __name__ == '__main__':

    # Enter param
    N = int(input("Enter N: "))
    mx1 = float(input("Enter mx1: "))
    my1 = float(input("Enter my1: "))
    mx2 = float(input("Enter mx2: "))
    my2 = float(input("Enter my2: "))
    vx1 = float(input("Enter vx1: "))
    vy1 = float(input("Enter vy1: "))
    vx2 = float(input("Enter vx2: "))
    vy2 = float(input("Enter vy2: "))

    # generate data
    x1 = data_gen(mx1, vx1, N)
    y1 = data_gen(my1, vy1, N)
    x2 = data_gen(mx2, vx2, N)
    y2 = data_gen(my2, vy2, N)
    label_1 = np.zeros(N)
    label_2 = np.ones(N)

    # organizing data
    X = np.concatenate((x1, x2))
    Y = np.concatenate((y1, y2))
    label = np.concatenate((label_1, label_2))

    X_mat = []
    for i in range(num_of_data):
        poly_X = [X[i]**j for j in range(3)]
        X_mat.append(poly_X)
    X_mat = np.array(X_mat)
