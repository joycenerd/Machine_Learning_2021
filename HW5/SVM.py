import os
import csv
import numpy as np
from libsvm.svmutil import *


DATA_PATH = "./data/"


def read_data():
    # parse train
    X_train_f = open(DATA_PATH+"X_train.csv", 'r')
    X_train = list(csv.reader(X_train_f))
    X_train = np.array(X_train).astype(np.float64)
    Y_train_f = open(DATA_PATH+"Y_train.csv", 'r')
    Y_train = list(csv.reader(Y_train_f))
    Y_train = np.array(Y_train).astype(np.int32)

    # parse test
    X_test_f = open(DATA_PATH+"X_test.csv", 'r')
    X_test = list(csv.reader(X_test_f))
    X_test = np.array(X_test).astype(np.float64)
    Y_test_f = open(DATA_PATH+"Y_test.csv", 'r')
    Y_test = list(csv.reader(Y_test_f))
    Y_test = np.array(Y_test).astype(np.int32)

    return X_train, Y_train, X_test, Y_test


def cmp_kernels(X_train, Y_train, X_test, Y_test):
    kernel_types = {"linear": "-t 0", "polynomial": "-t 1", "RBF:": "-t 2"}
    for _type, param in kernel_types.items():
        m = svm_train(Y_train, X_train, "-q "+param)
        pred_label, pred_acc, pred_vals = svm_predict(
            Y_test, X_test, model, "-q")
        prit(f"{_type} kernel accuracy: {pred_acc}")


if __name__ == "__main__":
    X_train, Y_train, X_test, Y_test = read_data()
    print(f"Num of train: {len(X_train)}")
    print(f"Num of test: {len(X_test)}")

    # Compare linear, polynomial and RBF kernel
    cmp_kernels(X_train, Y_train, X_test, Y_test)
