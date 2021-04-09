from utils import input_poly_basis
import numpy as np
from random_data_generater import poly_data_gen

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
        pred_var = 1.0/a + X @ np.linalg.inv(_lambda) @ X.T

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
