from scipy.spatial.distance import cdist
from numpy.linalg import eig, norm, pinv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import argparse
import ntpath
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument("--option", type=str, default="PCA",
                    help="Choose which task to do: [PCA, LDA]")
parser.add_argument("--img-size", type=int, default=50,
                    help="image resize shape")
parser.add_argument("--kernel-type", type=str, default="linear",
                    help="kernel type for PCA/LDA: [linear, polynomial, rbf]")
parser.add_argument("--gamma", type=float, default=1,
                    help="gamma value for polynomial or rbf kernel")
parser.add_argument("--coeff", type=int, default=2,
                    help="coeff value for polynomial kernel")
parser.add_argument("--degree", type=int, default=20,
                    help="degree value for polynomial kernel")
args = parser.parse_args()


DATA_PATH = "/eva_data/zchin/Yale_Face_Database/"
SAVE_PATH = "./results/"


def read_data(data_path):
    img_size = args.img_size
    data = []
    filepath = []
    label = []

    for file in glob.glob(data_path+"*"):
        # file path (135,)
        filepath.append(file)

        # data (135,10000)
        image = Image.open(file)
        image = image.resize((img_size, img_size), Image.ANTIALIAS)
        image = np.array(image)
        data.append(image.ravel())

        # label (135,)
        _, tail = ntpath.split(file)
        label.append(int(tail[7:9]))

    return np.array(data), filepath, np.array(label)


def get_eig(data, method, kernel_type="none"):
    # get eigenvalue and eigenvector by np.linalg.eig()
    eigval_file = DATA_PATH+method+"_"+kernel_type+"_eigval.npy"
    eigvec_file = DATA_PATH+method+"_"+kernel_type+"_eigvec.npy"

    """if os.path.isfile(eigval_file) and os.path.isfile(eigvec_file):
        eigval = np.load(eigval_file)
        eigvec = np.load(eigvec_file)
    else:"""
    eigval, eigvec = eig(data)
    # sort by decreasing order of eigenvalues
    idx = eigval.argsort()[::-1]
    eigval = eigval[idx]
    eigvec = eigvec[:, idx]
    np.save(eigval_file, eigval)
    np.save(eigvec_file, eigvec)

    return eigval, eigvec


def get_kernel(X):
    kernel_type = args.kernel_type
    gamma = args.gamma
    coeff = args.coeff
    degree = args.degree

    if kernel_type == "linear":
        kernel = X@X.T
    elif kernel_type == "polynomial":
        kernel = np.power(gamma*(X@X.T)+coeff, degree)
    elif kernel_type == "rbf":
        kernel = np.exp(-gamma*cdist(X, X, metric="sqeuclidean"))
    return kernel


def pca(x, kernel_type=None, kernel=None):
    if kernel_type == None:
        x_bar = np.mean(x, axis=0)
        cov = (x-x_bar)@(x-x_bar).T
        eigval, eigvec = get_eig(cov, "pca")
        # project data
        eigvec = (x-x_bar).T@eigvec

    else:
        x_bar = 0

        # cetralize the kernel
        n = kernel.shape[0]
        one = np.ones((n, n), dtype=float)
        one *= 1.0/n
        kernel = kernel - one @ kernel - kernel @ one + one @ kernel @ one

        eigval, eigvec = get_eig(kernel, "pca", kernel_type)

    for i in range(eigvec.shape[1]):
        eigvec[:, i] *= 1/norm(eigvec[:, i], 1)
    # get the top 25 eigenvectors
    W = eigvec[:, :25].real

    return x_bar, W


def draw_eigenface(W, name):
    img_size = args.img_size

    # save eigenface in 5x5 grid
    for i in range(5):
        for j in range(5):
            idx = i * 5 + j
            plt.subplot(5, 5, idx + 1)
            plt.imshow(W[:, idx].reshape((img_size, img_size)), cmap='gray')
            plt.axis('off')
    plt.savefig(SAVE_PATH+name+".jpg")


def lda(X, label, kernel_type="none", dims=25):
    (n, d) = X.shape
    label = np.asarray(label)
    c = np.unique(label)
    mu = np.mean(X, axis=0)
    S_w = np.zeros((d, d), dtype=np.float64)
    S_b = np.zeros((d, d), dtype=np.float64)

    # Sw=(xi-mj)*(xi-mj)^T
    # Sb=nj*(mj-m)*(mj-m)^T
    for i in c:
        X_i = X[np.where(label == i)[0], :]
        mu_i = np.mean(X_i, axis=0)
        S_w += (X_i - mu_i).T @ (X_i - mu_i)
        S_b += X_i.shape[0] * ((mu_i - mu).T @ (mu_i - mu))

    # get eigenvalues and eigenvectors
    S = pinv(S_w) @ S_b
    eigen_val, eigen_vec = get_eig(S, "lda", kernel_type)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / norm(eigen_vec[:, i])

    W = eigen_vec[:, :25].real
    return W


def reconstruct(data, W, method, m=None):
    img_size = args.img_size

    if method == "pca":
        reconstruction = (data-m)@W@W.T+m
    elif method == "lda":
        reconstruction = data@W@W.T

    idx = 1
    for i in range(2):
        for j in range(5):
            plt.subplot(2, 5, idx)
            plt.imshow(reconstruction[idx-1, :].reshape(
                (img_size, img_size)), cmap='gray')
            plt.axis('off')
            idx += 1
    plt.savefig(SAVE_PATH+method+"_reconstruction"+".jpg")


def face_recognition(train_data, train_label, test_data, test_label):
    num_of_train = train_label.shape[0]
    num_of_test = test_label.shape[0]
    dist_mat = np.zeros((num_of_test, num_of_train), dtype=float)

    # calculate distance
    for i in range(num_of_test):
        dist = np.zeros(num_of_train, dtype=float)
        for j in range(num_of_train):
            dist[j] = np.sum((test_data[i, :]-train_data[j, :])**2)
        dist = np.argsort(dist)
        dist_mat[i, :] = label[dist]

    # KNN
    K = [1, 3, 5, 7, 9, 11]
    best_acc = 0.0
    for k in K:
        correct = 0.0
        for i in range(num_of_test):
            dist = dist_mat[i, :]
            dist = dist[:k]
            val, cnt = np.unique(dist, return_counts=True)
            most_cnt = np.argmax(cnt)
            pred = val[most_cnt]
            if pred == test_label[i]:
                correct += 1

        acc = correct/num_of_test
        print(f"Face recognition accuracy when K={k}: {acc:.4}")
        if acc > best_acc:
            best_acc = acc
            best_K = k
    print(f"Best K: {best_K}\tBest accuracy: {best_acc:.4}")


def project(train_data, test_data, W, m=0):
    # data dimensionality reductionn
    option = args.option

    if option == "PCA":
        train_proj = (train_data-m)@W
        test_proj = (test_data-m)@W

    elif option == "LDA":
        train_proj = train_data@W
        test_proj = test_data@W

    return train_proj, test_proj


if __name__ == "__main__":
    option = args.option
    kernel_type = args.kernel_type

    # read training and testing data
    train_data, train_filepath, train_label = read_data(DATA_PATH+"Training/")
    test_data, test_filepath, test_label = read_data(DATA_PATH+"Testing/")
    data = np.vstack((train_data, test_data))  # (165,10000)
    filepath = np.hstack((train_filepath, test_filepath))  # (165,)
    label = np.hstack((train_label, test_label))  # (165,)
    num_of_data = label.shape[0]
    print(f"Num of data: {num_of_data}")

    if option == "PCA":
        rand_idx = np.random.randint(num_of_data, size=10)
        samples = data[rand_idx, :]  # (10,10000)

        x_bar, W = pca(data)
        draw_eigenface(W, "eigenface")
        print("eigenface completed...")

        reconstruct(samples, W, "pca", x_bar)
        print("reconstruction completed...")

        train_proj, test_proj = project(train_data, test_data, W, x_bar)
        face_recognition(train_proj, train_label, test_proj, test_label)
        print("pca face recognition completed...\n")

        # python kernel_eigenface.py --option PCA --kernel-type polynomial --gamma 5 --coeff 1 --degree 2
        # python kernel_eigenface.py --option  PCA --kernel-type rbf --gamma 1e-7
        kernel = get_kernel(data)
        _, W = pca(data, kernel_type, kernel)
        train_kernel = kernel[:train_label.shape[0], :]
        test_kernel = kernel[train_label.shape[0]:, :]
        train_proj, test_proj = project(train_kernel, test_kernel, W)
        face_recognition(train_proj, train_label, test_proj, test_label)
        print(
            f"kernel pca with {kernel_type} kernel face recognition completed...")

    if option == "LDA":
        rand_idx = np.random.randint(num_of_data, size=10)
        samples = data[rand_idx, :]  # (10,10000)

        W = lda(data, label)
        draw_eigenface(W, "fisherface")
        print("fisherface completed...")

        reconstruct(samples, W, "lda")
        print("reconstruction completed...")

        train_proj, test_proj = project(train_data, test_data, W)
        face_recognition(train_proj, train_label, test_proj, test_label)
        print("lda face recognition completed...\n")

        # python kernel_eigenface.py --option LDA --kernel-type polynomial --gamma 1 --coeff 2 --degree 20
        # python kernel_eigenface.py --option  PCA --kernel-type rbf --gamma 1e-4
        kernel = get_kernel(data.T)
        W = lda(kernel, kernel_type)
        train_kernel = kernel[:train_label.shape[0], :]
        test_kernel = kernel[train_label.shape[0]:, :]
        train_proj, test_proj = project(train_kernel, test_kernel, W)
        face_recognition(train_proj, train_label, test_proj, test_label)
        print(
            f"kernel lda with {kernel_type} kernel face recognition completed...")
