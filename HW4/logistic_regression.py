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


def gradient_descent(num_iter, X, Y):
    num_of_data = X.shape[0]
    w = data_gen(0.0, 1.0, 3)  # [3,]
    for i in range(num_iter):
        # sigmoid
        z = X @ w
        sigmoid_z = np.zeros(num_of_data)
        for i, value in enumerate(z):
            if value >= 0:
                sigmoid_z[i] = 1.0 / 1 + np.exp(-value)
            else:
                sigmoid_z[i] = np.exp(value) / (1 + np.exp(value))

        # backward
        gradient = X.T @ (Y - sigmoid_z)
        old_w = w
        w = w + gradient
        # print(w)
        if sum(abs(w - old_w)) < 1e-3:
            print(f"Converge in {i}th iteration!")
            return w
    return w


def newton_method(num_iter,X,Y):
    num_of_data=X.shape[0]
    w=data_gen(0.0,1.0,3)
    for i in range(num_iter):
        


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
        ([[x1[i], y1[i], 1.0] for i in range(N)], [[x2[i], y2[i], 1.0] for i in range(N)]), axis=0
    )  # [N*2,3]
    label = np.concatenate((label_1, label_2))  # [N*2,1]

    w = gradient_descent(100000, X, label)
    pred_Y = get_predict_value(X, label, w)
    tp, tn, fp, fn = get_confusion_matrix(label, pred_Y)

    # Output
    print("Gradient descent:\n")
    print("w:")
    for value in w:
        print(f"\t{value}")
    print(f"Confusion Matrix:")
    confusion_mat = [[tp, fn], [fp, tn]]
    header1 = ["Predict cluster 1", "Predict cluster 2"]
    header2 = ["Is cluster 1", "Is cluster 2"]
    print(pd.DataFrame(confusion_mat, header2, header1))
    print("")
    print(f"Sensitivity (Successfully predict cluster 1): {float(tp)/(tp+fn)}")
    print(f"Specificity (Successfully predict cluster 2): {float(tn)/(tn+fp)}")
    print("")
    print("----------------------------------------")

