import numpy as np
from libsvm.svmutil import *
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import cdist
import os
import csv
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--part2-kernel-type", type=str,
                    default="RBF", help="choose the kernel type for part2")
parser.add_argument("--part", type=int, default=3,
                    help="Which part do you want to execute")
parser.add_argument("--gamma", type=float, default=0.05,
                    help="Part 3 gamma value for self defined kernel")
args = parser.parse_args()


DATA_PATH = "./data/"
RESULT_PATH = "./results/"


def read_data():
    # parse train
    X_train_f = open(DATA_PATH + "X_train.csv", "r")
    X_train = list(csv.reader(X_train_f))
    X_train = np.array(X_train).astype(np.float64)
    Y_train_f = open(DATA_PATH + "Y_train.csv", "r")
    Y_train = list(csv.reader(Y_train_f))
    Y_train = np.array(Y_train).astype(np.int32)

    # parse test
    X_test_f = open(DATA_PATH + "X_test.csv", "r")
    X_test = list(csv.reader(X_test_f))
    X_test = np.array(X_test).astype(np.float64)
    Y_test_f = open(DATA_PATH + "Y_test.csv", "r")
    Y_test = list(csv.reader(Y_test_f))
    Y_test = np.array(Y_test).astype(np.int32)

    return X_train, Y_train, X_test, Y_test


def cmp_kernels(X_train, Y_train, X_test, Y_test):
    kernel_types = {"linear": "-t 0", "polynomial": "-t 1", "RBF:": "-t 2"}
    for _type, param in kernel_types.items():
        m = svm_train(Y_train, X_train, "-q " + param)
        pred_label, pred_acc, pred_vals = svm_predict(Y_test, X_test, m, "-q")
        print(f"{_type} kernel accuracy: {pred_acc[0]}%")


def grid_search(X_train, Y_train, X_test, Y_test, cost, gamma, degree, coeff0, option):
    max_acc = 0
    best_g = 0
    best_c = 0
    best_d = 0
    best_r = 0
    best_param = []
    confusion_mat = 0

    if option == "rbf":
        confusion_mat = np.zeros((len(cost), len(gamma)))
        for i in range(len(cost)):
            for j in range(len(gamma)):
                param = f"-q -s 0 -t 2 -g {gamma[i]} -c {cost[i]} -v 5"
                acc = svm_train(Y_train, X_train, param)
                confusion_mat[i][j] = acc
                if acc > max_acc:
                    max_acc = acc
                    best_g = gamma[i]
                    best_c = cost[i]
        best_param = [best_c, best_g]

    elif option == "linear":
        for c in cost:
            param = f"-q -s 0 -t 0 -c {c} -v 5"
            acc = svm_train(Y_train, X_train, param)
            if acc > max_acc:
                max_acc = acc
                best_c = c
        best_param = [best_c]

    elif option == "polynomial":
        for c in cost:
            for g in gamma:
                for d in degree:
                    for r in coeff0:
                        param = f"-q -s 0 -t 1 -c {c} -g {g} -d {d} -r {r} -v 5"
                        acc = svm_train(Y_train, X_train, param)
                        if acc > max_acc:
                            max_acc = acc
                            best_c = c
                            best_g = g
                            best_d = d
                            best_r = r
        best_param = [best_c, best_g, best_d, best_r]

    return best_param, confusion_mat


def plot_confusion_mat(confusion_mat, cost, gamma, option):
    cost_param = []
    for val in cost:
        cost_param.append(str(val))
    gamma_param = []
    for val in gamma:
        gamma_param.append(str(val))

    confusion_mat /= 100
    df = pd.DataFrame(confusion_mat, index=cost_param, columns=gamma_param)
    sns.heatmap(df, annot=True, cmap="YlGnBu", fmt=".2%")
    plt.xlabel("gamma")
    plt.ylabel("cost")
    plt.title("RBF C-SVC Parameters")
    plt.savefig(RESULT_PATH + f"{option}_confusion_matrix.jpg")
    plt.show()


def predict(X_train, Y_train, X_test, Y_test, param, _type):
    m = svm_train(Y_train, X_train, param)
    pred_label, pred_acc, pred_vals = svm_predict(Y_test, X_test, m, "-q")
    print(f"{_type} kernel with best parameters accuracy: {pred_acc[0]}%")


def get_linear_rbf_kernel(x, x_prime, gamma):
    linear_kernel = x@x_prime.T
    rbf_kernel = np.exp(-gamma*cdist(x, x_prime, 'sqeuclidean'))
    linear_rbf_kernel = linear_kernel+rbf_kernel
    linear_rbf_kernel = np.hstack(
        (np.arange(1, len(x)+1).reshape(-1, 1), linear_rbf_kernel))
    return linear_rbf_kernel


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data()
    Y_train = Y_train.ravel()
    Y_test = Y_test.ravel()
    print(f"Num of train: {len(X_train)}")
    print(f"Num of test: {len(X_test)}")

    # Part 1: Compare linear, polynomial and RBF kernel
    if args.part == 1:
        cmp_kernels(X_train, Y_train, X_test, Y_test)

    # Part2: find the best param for C-SVC
    elif args.part == 2:
        option = args.part2_kernel_type
        cost = [1, 10, 20, 30]
        gamma = [0.01, 0.05, 0.1, 0.2]
        degree = [2, 3, 4, 5]
        coef0 = [0, 5, 10, 15]

        if option == "rbf":
            best_param, confusion_mat = grid_search(
                X_train, Y_train, X_test, Y_test, cost, gamma, -1, -1, option
            )
            print(f"best cost: {best_param[0]}, best gamma: {best_param[1]}")
            plot_confusion_mat(confusion_mat, cost, gamma, option)
            param = f"-q -s 0 -t 2 -c {best_param[0]} -g {best_param[1]}"
            predict(X_train, Y_train, X_test, Y_test, param, option)

        elif option == "linear":
            best_param, _ = grid_search(
                X_train, Y_train, X_test, Y_test, cost, -1, -1, -1, option
            )
            print(f"best cost: {best_param[0]}")
            param = f"-q -s 0 -t 0 -c {best_param[0]}"
            predict(X_train, Y_train, X_test, Y_test, param, option)

        elif option == "polynomial":
            best_param, _ = grid_search(
                X_train, Y_train, X_test, Y_test, cost, gamma, degree, coef0, option
            )
            print(
                f"best cost: {best_param[0]}, best gamma: {best_param[1]}, best degree: {best_param[2]}, best coeff0: {best_param[3]}")
            param = f"-q -s 0 -t 1 -c {best_param[0]} -g {best_param[1]} -d {best_param[2]} -r {best_param[3]}"
            predict(X_train, Y_train, X_test, Y_test, param, option)

    # Part3: self-defined kernel: linear+rbf
    elif args.part == 3:
        train_kernel = get_linear_rbf_kernel(X_train, X_train, args.gamma)
        test_kernel = get_linear_rbf_kernel(X_test, X_train, args.gamma)
        param = "-q -t 4"
        predict(train_kernel, Y_train, test_kernel,
                Y_test, param, "linear+rbf")
