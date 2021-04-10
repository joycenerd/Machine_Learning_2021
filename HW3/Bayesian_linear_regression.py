from utils import input_poly_basis
import numpy as np
from random_data_generater import poly_data_gen
import matplotlib.pyplot as plt


def get_poly(X, w, n):
    X_mat = []
    for x in X:
        X_mat.append([(x ** deg) for deg in range(n)])
    X_mat = np.array(X_mat)
    W = np.array(w).reshape(-1)
    Y = X_mat @ W
    return Y


def get_predictive_distribution(x, m, s, n, a):
    y = np.zeros(len(x))
    err = np.zeros(len(x))
    for i in range(len(x)):
        X = np.array([x[i] ** deg for deg in range(n)])
        # predictive distribution ~ N(X*μ, 1/a+X*Λ^(-1)*X^T)
        y[i] = np.dot(X, m)
        err[i] = 1 / a + np.dot(X, np.dot(s, X.T))
    return y, err


if __name__ == '__main__':
    b = float(input("Input b: "))
    n, a, w = input_poly_basis()
    cnt = 1
    m = np.zeros([n, 1])  # inital prior mean (nx1)
    S = b * np.identity(n)  # inital inverse of prior variance
    pred_mean = 0.0
    pred_var = 0.0

    X_list = []
    Y_list = []
    while (1):
        # generate polynomial random number
        pt_x, y = poly_data_gen(n, a, w)
        X_list.append(pt_x)
        Y_list.append(y)

        X = np.array([(pt_x ** deg) for deg in range(n)]).reshape(1, -1)

        # posterior Λ = a * X^T * X + (bI or S)
        _lambda = a * X.T @ X + S
        post_var = np.linalg.inv(_lambda)

        # posterior μ = Λ^(-1)*(a * X^T * y + S * m)  # 1st iteration: S * m == 0
        post_mean = np.linalg.inv(_lambda) @ (a * X.T * y + S @ m)

        old_pred_mean = pred_mean
        old_pred_var = pred_var

        # predictive distribution ~ N(X*mu, 1/a+X*Λ^(-1)*X^T)
        pred_mean = X @ post_mean
        pred_var = 1.0 / a + X @ np.linalg.inv(_lambda) @ X.T

        print('Add data point ({:.5f}, {:.5f}):\n'.format(pt_x, y))
        print('Posterior mean:')  # n x 1
        for i in range(n):
            print('  {:.10f}'.format(post_mean[i][0]))
        print('\nPosterior variance:')  # n x n
        for i in range(n):
            for j in range(n):
                print('  {:.10f}'.format(post_var[i][j]), end=',')
            print()
        print('\nPredictive distribution ~ N({:.5F}, {:.5F})'.format(pred_mean[0][0], pred_var[0][0]))
        print('-----------------------------------------------')

        if (cnt == 10):
            m_10, S_10 = post_mean, post_var
        elif (cnt == 50):
            m_50, S_50 = post_mean, post_var

        if (cnt >= 100 and abs(pred_var - old_pred_var) < 1e-4 and abs(pred_mean - old_pred_mean) < 1e-2):
            m_final, S_final = post_mean, post_var
            break

        S = _lambda  # inverse of prior covariance
        m = post_mean  # prior mean
        cnt += 1

    # When converge visualize the results
    X = np.linspace(-2.0, 2.0, 1000)

    # plot ground truth
    Y = get_poly(X, w, n)
    plt.subplot(2, 2, 1)
    plt.plot(X, Y, color="black")
    plt.plot(X, Y + a, color="red")
    plt.plot(X, Y - a, color="red")
    plt.title("Ground truth")
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)

    # plot predict results
    plt.subplot(2, 2, 2)
    plt.plot(X, Y, color="black")
    plt.plot(X, Y + a, color="red")
    plt.plot(X, Y - a, color="red")
    plt.scatter(X_list, Y_list, color="tab:blue")
    plt.title("Predict result")
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)

    # at the time that have seen 10 data points
    plt.subplot(2, 2, 3)
    Y, var = get_predictive_distribution(X, m_10, S_10, n, a)
    plt.plot(X, Y, color="black")
    plt.plot(X, Y + var, color="red")
    plt.plot(X, Y - var, color="red")
    plt.scatter(X_list[:10], Y_list[:10], color="tab:blue")
    plt.title("After 10 incomes")
    plt.xlim(-2, 2)
    plt.ylim(-20, 30)

    plt.show()
