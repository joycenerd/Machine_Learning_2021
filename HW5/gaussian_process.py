import numpy as np
import matplotlib.pyplot as plt
import scipy.optimize


INPUT_SIZE = 34
DATA_PATH = "./data/"
BETA = 5
RESULT_PATH = "./results/"


def get_kernel(x, x_prime, theta):
    # rational quadratic kernel: sigma^2*(1+(x-x_prime)^2/(2*alpha*length_scale^2))^(-alpha)
    sigma, alpha, length_scale = theta
    x = x.reshape(-1, 1)
    x_prime = x_prime.reshape(1, -1)
    divisor = 1+(x-x_prime)*(x-x_prime)
    dividend = 2*alpha*length_scale**2
    kernel = sigma**2*np.power(divisor/dividend, -alpha)
    return kernel


def predict(test_x, train_x, train_y, theta):
    kernel = get_kernel(train_x, train_x, theta)  # k(x,x)
    C = kernel + 1/BETA*np.eye(INPUT_SIZE)
    k_x_xstar = get_kernel(train_x, test_x, theta)  # k(x,x*)
    k_xstar_xstar = get_kernel(test_x, test_x, theta)  # k(x*,x*)

    # predictive distribution
    # predictive_mean=k(x,x*)^T*C^(-1)*y
    pred_mean = k_x_xstar.T@np.linalg.inv(C)@train_y
    pred_mean = pred_mean.reshape(-1)
    # predictive_variance=k(x*,x*)-k(x,x*)^T*C^(-1)*k(x,x*)
    pred_var = k_xstar_xstar-k_x_xstar.T@np.linalg.inv(C)@k_x_xstar
    pred_var = np.sqrt(np.diag(pred_var))

    return pred_mean, pred_var


def get_log_likelihood(theta, *args):
    train_x, train_y = args
    kernel = get_kernel(train_x, train_x, theta)
    C = kernel + 1/BETA*np.eye(INPUT_SIZE)

    # log(p(y|X)) = min( 0.5 * (y.T*C^(-1)*y + log(det(C)) + N*log(2*pi)))
    log_likelihood = train_y.T@np.linalg.inv(C)@train_y+np.sum(
        np.log(np.diagonal(np.linalg.cholesky(kernel))))+INPUT_SIZE*np.log(2*np.pi)
    log_likelihood /= 2.0
    return log_likelihood


if __name__ == "__main__":
    train_x = np.zeros(INPUT_SIZE)
    train_y = np.zeros(INPUT_SIZE)
    data = open(DATA_PATH+"input.data")
    for i, coordinate in enumerate(data):
        train_x[i], train_y[i] = coordinate.strip("\n").split()

    test_x = np.linspace(-60, 60, 500)
    theta = np.ones(3)
    pred_mean, pred_var = predict(test_x, train_x, train_y, theta)

    # plot the result
    plt.figure(figsize=(10, 10))
    plt.scatter(train_x, train_y)
    plt.plot(test_x, pred_mean)
    plt.fill_between(test_x, pred_mean+2*pred_var,
                     pred_mean-2*pred_var, alpha=0.3)
    plt.title(
        f"Initial Gaussian Process sigma={theta[0]}, alpha={theta[1]}, length scale={theta[2]}")
    plt.savefig(RESULT_PATH+"initial_gaussian_process.jpg")
    plt.show()

    # Optimize the kernel parameters
    x0 = np.ones(3)
    opt_param = scipy.optimize.minimize(
        get_log_likelihood, args=(train_x, train_y), x0=x0, method='CG').x
    pred_mean, pred_var = predict(test_x, train_x, train_y, opt_param)

    # plot the result
    plt.figure(figsize=(10, 10))
    plt.scatter(train_x, train_y)
    plt.plot(test_x, pred_mean)
    plt.fill_between(test_x, pred_mean+2*pred_var,
                     pred_mean-2*pred_var, alpha=0.3)
    plt.title(
        f"Optimize Gaussian Process sigma={opt_param[0]}, opt_param={theta[1]}, length scale={opt_param[2]}")
    plt.savefig(RESULT_PATH+"optimize_gaussian_process.jpg")
    plt.show()
