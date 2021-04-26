import gzip
import numpy as np
import math
import time
import pandas as pd


start = time.time()
total_class = 10
cols = 28
rows = 28
total_px = cols * rows
dt = np.dtype(np.uint8).newbyteorder(">")


def loadMNIST(data_file, label_file):
    f_ti = open(data_file, "rb")
    f_tl = open(label_file, "rb")

    _ = f_ti.read(4)  # magic_number(4 bytes)
    img_num = int.from_bytes(f_ti.read(4), "big")
    rows = int.from_bytes(f_ti.read(4), "big")
    cols = int.from_bytes(f_ti.read(4), "big")

    _ = f_tl.read(8)  # magic_number(4 bytes), item number(4 bytes)

    img_pixels = np.zeros((img_num, rows * cols), dtype=int)
    img_label = np.zeros(img_num, dtype=int)
    for n in range(img_num):
        pixels = f_ti.read(rows * cols)
        img_pixels[n] = np.frombuffer(pixels, dtype=dt)
        img_label[n] = int.from_bytes(f_tl.read(1), "big")
    f_ti.close()
    f_tl.close()

    return img_pixels, img_label


def printNumber(print_str, guess, labels=np.arange(10)):
    for c in range(total_class):
        print(print_str + " {}:".format(c))
        for px_idx in range(total_px):
            if px_idx % rows == cols - 1:
                print()
            else:
                print("{:d}".format(guess[labels[c]][px_idx] >= 0.5), end=" ")
        print()


def printResult(train_x, train_y, _lambda, px_prob, iterations):

    # (ground truth labels, highest possibility label)
    # count the number of ground truth c, with highest possibility class p
    count_cls = np.zeros((total_class, total_class))
    pred_y = np.zeros(train_x.shape[0], dtype=int)

    px_prob_comp = 1.0 - px_prob
    train_x_comp = 1 - train_x
    for n in range(train_x.shape[0]):
        # compute the prediction (highest probability)
        prob_cls = _lambda.copy()
        for k in range(total_class):
            prob_cls[k] = prob_cls[k] * (
                np.prod(
                    np.multiply(px_prob[k, :], train_x[n, :])
                    + np.multiply(px_prob_comp[k, :], train_x_comp[n, :])
                )
            )

        pred_y[n] = np.argmax(prob_cls)
        count_cls[train_y[n]][pred_y[n]] += 1

    # find the corresponding number
    table = count_cls.copy()
    labels = np.full(total_class, -1, dtype=int)
    for k in range(total_class):
        max_idx = np.argmax(count_cls, axis=None)
        (lb_num, lb) = np.unravel_index(max_idx, count_cls.shape)
        labels[lb_num] = lb
        count_cls[lb_num, :] = -1
        count_cls[:, lb] = -1  # set impossible value

    printNumber("labeled class", px_prob, labels)

    # plot confusion matrix
    cmx = np.zeros((total_class, 2, 2), dtype=int)
    total_correct = 0.0
    for i, c in enumerate(labels):
        cmx[i][1][1] = table[i][c]  # tp
        cmx[i][0][1] = np.sum(table[:, c]) - table[i][c]  # fp
        cmx[i][1][0] = np.sum(table[i, :]) - table[i][c]  # fn
        cmx[i][0][0] = (
            np.sum(table) - np.sum(table[:, c]) - np.sum(table[i, :]) + table[i][c]
        )  # tn
        total_correct += cmx[i][1][1]

    for c in range(10):
        print("---------------------------------------------------------------\n")
        print(f"Confusion matrix {c}")
        header1 = [f"Predict number {c}", f"Predict not number {c}"]
        header2 = [f"Is number {c}", f"Isn't number {c}"]
        confusion_mat = [[cmx[c][1][1], cmx[c][1][0]], [cmx[c][0][1], cmx[c][0][0]]]
        print(pd.DataFrame(confusion_mat, header2, header2))
        print(
            f"Sensitivity (Successfully predict number {c}): {cmx[c][1][1] / (cmx[c][1][1] + cmx[c][1][0])}"
        )
        print(
            f"Specificity (Successfully predict not number {c}): {cmx[c][0][0] / (cmx[c][0][0] + cmx[c][0][1])}"
        )
    print("Total iteration to converge: {:d}".format(iterations))
    print(
        "Total error rate: {}".format(
            float(train_x.shape[0] - total_correct) / train_x.shape[0]
        )
    )


def binning(x):
    # map pixel<128 to 0 and >=128 to 1
    shape = (x.shape[0], x.shape[1])
    x = x.reshape(-1)
    new_x = (x >= 128).astype(int)
    new_x = np.reshape(new_x, shape)
    return new_x


def initialize(n):
    # # initialize
    _lambda = np.full(total_class, 1.0 / total_class, dtype=float)
    # # px_prob[k][i] = probability of i^th pixel = 1 in no.k cluster
    px_prob = np.random.rand(total_class, total_px).astype(float)
    new_px_prob = np.zeros_like(px_prob, dtype=float)
    # z = np.full(( n, total_class), 1.0/total_class, dtype=float)
    z = np.zeros((n, total_class), dtype=float)
    return _lambda, px_prob, new_px_prob, z


def EM(train_x, max_cnt=50):
    train_x_comp = 1 - train_x

    # initialize
    _lambda, px_prob, new_px_prob, z = initialize(train_x.shape[0])

    cnt = 1
    true_iter = 1
    while cnt <= max_cnt:

        # E step
        px_prob_comp = 1.0 - px_prob
        for n in range(z.shape[0]):  # 60000
            u_frac = _lambda.copy()
            for k in range(z.shape[1]):
                u_frac[k] = u_frac[k] * (
                    np.prod(
                        np.multiply(px_prob[k, :], train_x[n, :])
                        + np.multiply(px_prob_comp[k, :], train_x_comp[n, :])
                    )
                )
            marginal = float(np.sum(u_frac))
            if marginal == 0:
                marginal = 1
            z[n] = u_frac / marginal

        # M step: update lambda and mu
        new_N = np.sum(z, axis=0)
        new_lambda = new_N / train_x.shape[0]
        new_N[new_N == 0] = 1
        for k in range(px_prob.shape[0]):  # 10
            for p in range(px_prob.shape[1]):  # 28*28
                new_px_prob[k][p] = np.dot(train_x[:, p], z[:, k]) / new_N[k]

        # check if it is a valid update (_lambda should not be 0)
        if np.count_nonzero(new_lambda) != total_class:
            new_lambda, new_px_prob, _, z = initialize(train_x.shape[0])
        else:
            true_iter += 1

        diff = np.sum(np.abs(new_px_prob - px_prob)) + np.sum(
            np.abs(new_lambda - _lambda)
        )

        printNumber("class", new_px_prob)
        print("\nNo. of Iteration: {}, Difference: {}\n".format(cnt, diff))
        print("---------------------------------------------------------------\n")

        if diff < 0.02 and true_iter >= 8 and np.sum(new_lambda) > 0.95:
            break

        _lambda = new_lambda
        px_prob = new_px_prob
        cnt += 1

    print("---------------------------------------------------------------")
    print("---------------------------------------------------------------")

    return new_lambda, new_px_prob, cnt


if __name__ == "__main__":
    train_img_file = "./data/train-images-idx3-ubyte"
    train_label_file = "./data/train-labels-idx1-ubyte"
    train_pixels, train_labels = loadMNIST(train_img_file, train_label_file)

    new_train_pixels = binning(train_pixels)
    _lambda, px_prob, iterations = EM(new_train_pixels)
    printResult(new_train_pixels, train_labels, _lambda, px_prob, iterations)
