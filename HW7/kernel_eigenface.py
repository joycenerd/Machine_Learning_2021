from numpy.core.fromnumeric import mean
from numpy.linalg import eig, norm, pinv
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import argparse
import ntpath
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument("--option", type=str, default="PCA/LDA eigenface",
                    help="Choose which task to do: [eigenface, PCA/LDA face recognition, Kernel PCA/LDA face recognition]")
parser.add_argument("--img-size", type=int, default=100,
                    help="image resize shape")
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


def get_eig(data, method):
    # get eigenvalue and eigenvector by np.linalg.eigh()
    if method == "lda":
        eigval_file = DATA_PATH+"lda_eigval.npy"
        eigvec_file = DATA_PATH+"lda_eigvec.npy"
    elif method == "pca":
        eigval_file = DATA_PATH+"pca_eigval.npy"
        eigvec_file = DATA_PATH+"pca_eigvec.npy"

    if os.path.isfile(eigval_file) and os.path.isfile(eigvec_file):
        eigval = np.load(eigval_file)
        eigvec = np.load(eigvec_file)
    else:
        eigval, eigvec = eig(data)
        # sort by decreasing order of eigenvalues
        idx = eigval.argsort()[::-1]
        eigval = eigval[idx]
        eigvec = eigvec[:, idx]
        np.save(eigval_file, eigval)
        np.save(eigvec_file, eigvec)

    return eigval, eigvec


def pca(x):
    x_bar = np.mean(x, axis=0)
    cov = (x-x_bar)@(x-x_bar).T
    eigval, eigvec = get_eig(cov, "pca")
    # project data
    P = (x-x_bar).T@eigvec
    for i in range(P.shape[1]):
        P[:, i] *= 1/norm(P[:, i], 1)
    # get the top 25 eigenvectors
    W = P[:, :25].real
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


def lda(X, label, dims=25):
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
    eigen_val, eigen_vec = get_eig(S, "lda")
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / norm(eigen_vec[:, i])

    W = eigen_vec[:, :25].real
    return W


def reconstruct(data, W, method, mean=None):
    img_size = args.img_size

    if method == "pca":
        reconstruction = (data-mean)@W@W.T+mean
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

    for i in range(num_of_test):
        dist = np.zeros(num_of_train, dtype=float)
        for j in range(num_of_train):
            dist[j] = np.sum((test_data[i, :]-train_data[j, :])**2)
        dist_mat[i, :] = np.argsort(dist)


if __name__ == "__main__":
    option = args.option

    # read training and testing data
    train_data, train_filepath, train_label = read_data(DATA_PATH+"Training/")
    test_data, test_filepath, test_label = read_data(DATA_PATH+"Testing/")
    data = np.vstack((train_data, test_data))  # (165,10000)
    filepath = np.hstack((train_filepath, test_filepath))  # (165,)
    label = np.hstack((train_label, test_label))  # (165,)
    num_of_data = label.shape[0]
    print(f"Num of data: {num_of_data}")

    if option == "PCA/LDA eigenface":
        rand_idx = np.random.randint(num_of_data, size=10)
        samples = data[rand_idx, :]  # (10,10000)

        x_bar, W = pca(data)
        draw_eigenface(W, "eigenface")
        reconstruct(samples, W, "pca", x_bar)
        face_recognition(train_data, train_label, test_data, test_label)
        print("PCA completed...")

        """W = lda(data, label)
        draw_eigenface(W, "fisherface")
        reconstruct(samples, W, "lda")
        print("LDA completed...")"""
