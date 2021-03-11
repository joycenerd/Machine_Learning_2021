import numpy as np
from lse import lse, matrix_multiplication
import matplotlib.pyplot as plt
from newton import newton


def flatten(x):
    """flatten
    flatten 2d array to 1d array
    :param x: initial array
    :return: array after flatten
    """
    x_flatten = []
    for i in range(len(x)):
        for j in range(len(x[0])):
            x_flatten.append(x[i][j])
    return x_flatten


def get_power_matrix(x):
    """power_matrix
    get the matrix form of all data points power x^n x^n-1...x^1,x^0
    :param x: the original data point x axis
    :return: matrix of data points power
    """
    pow_mat = []
    for i in range(0, len(x)):
        X = []
        for exp in range(n - 1, -1, -1):
            X.append(pow(x[i], exp))
        pow_mat.append(X)
    return pow_mat


def print_predict(x):
    """print_predict
    print the prediction function
    :param x: function coefficient
    """
    x_flatten = flatten(x)
    # n=len(x_flatten)
    exp = n - 1
    output_str = ""
    for i in range(n):
        if exp == 0 and x_flatten[i] < 0:
            output_str += " {}".format(x_flatten[i])
        elif exp == 0 and x_flatten[i] >= 0:
            output_str += " + {}".format(x_flatten[i])
        elif exp == n - 1:
            output_str += " {}X^{}".format(x_flatten[i], exp)
        elif x_flatten[i] < 0:
            output_str += " {}X^{}".format(x_flatten[i], exp)
        else:
            output_str += " + {}X^{}".format(x_flatten[i], exp)
        exp -= 1
    print(output_str)


def calculate_error(y_pred, y):
    """calculate error
    calculate the least square error between prediction and ground truth
    :param y_pred: prediction
    :param y: ground truth
    :return: least square error
    """
    y_pred = flatten(y_pred)
    y = flatten(y)
    error = 0.0
    for i in range(len(y)):
        error += (y_pred[i] - y[i]) * (y_pred[i] - y[i])
    return error


if __name__ == '__main__':
    f = open('testfile.txt', 'r').readlines()
    X = []
    Y = []
    for i in range(0, len(f)):
        w = f[i].split(',')
        X.append(float(w[0]))
        Y.append(float(w[1]))

    cases = 1
    while True:
        print("Case#", cases)
        print("Input polynomial bases n: ")
        n = int(input())
        print("Input lambda: ")
        _lambda = float(input())

        A = []
        b = []
        A = get_power_matrix(X)
        for i in range(len(Y)):
            b.append([Y[i]])

        # lse
        x1 = lse(A, b, _lambda)
        print("LSE:")
        print("Fitting line:", end=" ")
        print_predict(x1)

        # Ax=y_pred
        y_pred = matrix_multiplication(A, x1)

        # error=sum((b-y_pred)^2)
        error = calculate_error(y_pred, b)
        print("Total error: ", error)
        print("\n")

        # newton's method
        x2 = newton(A, b, n)
        print("Newton's Method: ")
        print("Fitting line:", end=" ")
        print_predict(x2)

        # Ax=y_pred
        y_pred = matrix_multiplication(A, x2)

        # error=sum((b-y_pred)^2)
        error = calculate_error(y_pred, b)
        print("Total error: ", error)
        print("\n")

        # plot
        fig, (ax1, ax2) = plt.subplots(2, figsize=(8, 7))
        # plot lse result
        ax1.plot(X, Y, 'o', color="red")
        x_continuous = np.linspace(-6, 6, 1000).tolist()
        A_fit = get_power_matrix(x_continuous)
        y_continuous = matrix_multiplication(A_fit, x1)
        ax1.plot(x_continuous, y_continuous)
        ax1.title.set_text("LSE result")
        # plot newton's method result
        ax2.plot(X, Y, 'o', color="red")
        y_continuous = matrix_multiplication(A_fit, x2)
        ax2.plot(x_continuous, y_continuous)
        ax2.title.set_text("Newton's Method result")
        plt.show()

        cases+=1
