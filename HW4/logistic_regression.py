import numpy as np
import matplotlib.pyplot as plt
import pandas as pd


def data_gen(mean, variance, n):
    data = np.zeros(n)
    for i in range(n):
        u = np.random.uniform(0.0, 1.0)
        v = np.random.uniform(0.0, 1.0)
        z = np.sqrt(-2.0 * np.log(u)) * np.sin(2 * np.pi * v)
        data[i] = mean + np.sqrt(variance) * z
    return data


def sigmoid(x):
    for i in range(x.shape[0]):
        try:
            x[i] = 1.0 / (1.0 + np.exp(-1.0 * x[i]))
        except OverflowError:
            x[i] = 0.0
    return x


def gradient_descent(num_iter, X, Y):
    num_of_data = X.shape[0]
    w = data_gen(0.0, 1.0, 3)  # [3,]
    for i in range(num_iter):
        # sigmoid
        z = X @ w
        sigmoid_z = sigmoid(z)

        # backward
        gradient = X.T @ (Y - sigmoid_z)
        old_w = w
        w = w + gradient
        # print(w)
        if sum(abs(w - old_w)) < 1e-3:
            print(f"Gradinet descent converge in {i}th iteration!")
            return w
    return w


def get_hessian(X, w, z):
    D = np.eye(X.shape[0])
    for i in range(D.shape[0]):
        try:
            D[i][i] = np.exp(-1.0 * z[i]) / (1.0 + np.exp(-1.0 * z[i])) ** 2
        except OverflowError:
            D[i][i] = 0.0
    hessian = X.T @ D @ X
    return hessian


def newton_method(num_iter, X, Y):
    lr_rate = 0.01
    # w = data_gen(0.0, 1.0, 3)  # [3,]
    w = np.zeros(3, dtype=float)
    for i in range(100000):
        xw = X @ w
        hessian = get_hessian(X, w, xw)
        g = X.T @ (Y - sigmoid(xw))

        if np.linalg.det(hessian) != 0:
            # x1 = x0 + Hessian^(-1) * gradient
            new_w = w + np.dot(np.linalg.inv(hessian), g)
        else:
            # use Steepest gradient descent singular when not invertible --> determinant == 0
            new_w = w + lr_rate * g
        if sum(abs(new_w - w)) < 1e-3:
            print(f"Newton's method converge in {i}th iteration!")
            return new_w
        w = new_w
    return new_w


def get_predict_value(X, Y, w):
    pred_Y = np.zeros(Y.shape[0])
    out = X @ w
    for i in range(Y.shape[0]):
        if out[i] < 0.5:
            pred_Y[i] = 0
        else:
            pred_Y[i] = 1
    return pred_Y


def get_confusion_matrix(Y, pred_Y):
    tp = 0
    tn = 0
    fp = 0
    fn = 0
    for i in range(Y.shape[0]):
        if Y[i] == pred_Y[i] and Y[i] == 0.0:
            tp += 1
        elif Y[i] == pred_Y[i] and Y[i] == 1.0:
            tn += 1
        elif Y[i] == 0.0 and pred_Y[i] == 1.0:
            fn += 1
        else:
            fp += 1
    return tp, tn, fp, fn


if __name__ == "__main__":

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
    X = np.concatenate(
        (
            [[x1[i], y1[i], 1.0] for i in range(N)],
            [[x2[i], y2[i], 1.0] for i in range(N)],
        ),
        axis=0,
    )  # [N*2,3]
    label = np.concatenate((label_1, label_2))  # [N*2,1]

    w_gradient = gradient_descent(100000, X, label)
    pred_Y_gradient = get_predict_value(X, label, w_gradient)
    tp_gradient, tn_gradient, fp_gradient, fn_gradient = get_confusion_matrix(
        label, pred_Y_gradient
    )

    w_newton = newton_method(100000, X, label)
    pred_Y_newton = get_predict_value(X, label, w_newton)
    tp_newton, tn_newton, fp_newton, fn_newton = get_confusion_matrix(
        label, pred_Y_newton
    )

    gradient_x1 = []
    gradient_y1 = []
    gradient_x2 = []
    gradient_y2 = []

    newton_x1 = []
    newton_y1 = []
    newton_x2 = []
    newton_y2 = []

    for i, [x, y, _] in enumerate(X):
        if pred_Y_gradient[i] == 0.0:
            gradient_x1.append(x)
            gradient_y1.append(y)
        else:
            gradient_x2.append(x)
            gradient_y2.append(y)

        if pred_Y_newton[i] == 0.0:
            newton_x1.append(x)
            newton_y1.append(y)
        else:
            newton_x2.append(x)
            newton_y2.append(y)

    # Output
    print("Gradient descent:\n")
    print("w:")
    for value in w_gradient:
        print(f"\t{value}")
    print(f"Confusion Matrix:")
    confusion_mat = [[tp_gradient, fn_gradient], [fp_gradient, tn_gradient]]
    header1 = ["Predict cluster 1", "Predict cluster 2"]
    header2 = ["Is cluster 1", "Is cluster 2"]
    print(pd.DataFrame(confusion_mat, header2, header1))
    print("")
    print(
        f"Sensitivity (Successfully predict cluster 1): {float(tp_gradient)/(tp_gradient+fn_gradient)}"
    )
    print(
        f"Specificity (Successfully predict cluster 2): {float(tn_gradient)/(tn_gradient+fp_gradient)}"
    )
    print("")
    print("----------------------------------------")

    print("Newton's Method\n")
    print("w:")
    for value in w_newton:
        print(f"\t{value}")
    print(f"Confusion Matrix:")
    confusion_mat = [[tp_newton, fn_newton], [fp_newton, tn_newton]]
    header1 = ["Predict cluster 1", "Predict cluster 2"]
    header2 = ["Is cluster 1", "Is cluster 2"]
    print(pd.DataFrame(confusion_mat, header2, header1))
    print("")
    print(
        f"Sensitivity (Successfully predict cluster 1): {float(tp_newton)/(tp_newton+fn_newton)}"
    )
    print(
        f"Specificity (Successfully predict cluster 2): {float(tn_newton)/(tn_newton+fp_newton)}"
    )
    print("")
    print("----------------------------------------")

    # plot the result
    # ground truth
    plt.figure(figsize=(10, 8))
    plt.subplot(1, 3, 1)
    plt.scatter(x1, y1, c="b")
    plt.scatter(x2, y2, c="r")
    plt.title("Ground truth")
    # gradient descent
    plt.subplot(1, 3, 2)
    plt.scatter(gradient_x1, gradient_y1, c="b")
    plt.scatter(gradient_x2, gradient_y2, c="r")
    plt.title("Gradient descent")
    # newton's method
    plt.subplot(1, 3, 3)
    plt.scatter(newton_x1, newton_y1, c="b")
    plt.scatter(newton_x2, newton_y2, c="r")
    plt.title("Newton's method")
    plt.show()
