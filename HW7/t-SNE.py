#
#  tsne.py
#
# Implementation of t-SNE in Python. The implementation was tested on Python
# 2.7.10, and it requires a working installation of NumPy. The implementation
# comes with an example on the MNIST dataset. In order to plot the
# results of this example, a working installation of matplotlib is required.
#
# The example can be run by executing: `ipython tsne.py`
#
#
#  Created by Laurens van der Maaten on 20-12-08.
#  Copyright (c) 2008 Tilburg University. All rights reserved.

from PIL import Image
from scipy.spatial.distance import cdist
from numpy.linalg import pinv, eig, norm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

import argparse
import pylab
import os


parser = argparse.ArgumentParser()
parser.add_argument("--max_iter", type=int, default=1000,
                    help="Maximum iteration")
parser.add_argument("--method", type=str, default="t-sne",
                    help="Which sne method: [t-sne, s-sne]")
parser.add_argument("--dim-reduction", type=str, default="pca",
                    help="Which dimension reduction method: [pca,lda]")
parser.add_argument("--perplexity", type=int, default=50,
                    help="perplexity for sne: [30,50,70]")
args = parser.parse_args()


DATA_PATH = "/eva_data/zchin/data/"
SAVE_PATH = "./results/"


def Hbeta(D=np.array([]), beta=1.0):
    """
        Compute the perplexity and the P-row for a specific value of the
        precision of a Gaussian distribution.
    """

    # Compute P-row and corresponding perplexity
    P = np.exp(-D.copy() * beta)
    sumP = sum(P)
    H = np.log(sumP) + beta * np.sum(D * P) / sumP
    P = P / sumP
    return H, P


def x2p(X=np.array([]), tol=1e-5, perplexity=30.0):
    """
        Performs a binary search to get P-values in such a way that each
        conditional Gaussian has the same perplexity.
    """
    perplexity = args.perplexity

    # Initialize some variables
    print("Computing pairwise distances...")
    (n, d) = X.shape
    sum_X = np.sum(np.square(X), 1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)
    P = np.zeros((n, n))
    beta = np.ones((n, 1))
    logU = np.log(perplexity)

    # Loop over all datapoints
    for i in range(n):

        # Print progress
        if i % 500 == 0:
            print("Computing P-values for point %d of %d..." % (i, n))

        # Compute the Gaussian kernel and entropy for the current precision
        betamin = -np.inf
        betamax = np.inf
        Di = D[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))]
        (H, thisP) = Hbeta(Di, beta[i])

        # Evaluate whether the perplexity is within tolerance
        Hdiff = H - logU
        tries = 0
        while np.abs(Hdiff) > tol and tries < 50:

            # If not, increase or decrease precision
            if Hdiff > 0:
                betamin = beta[i].copy()
                if betamax == np.inf or betamax == -np.inf:
                    beta[i] = beta[i] * 2.
                else:
                    beta[i] = (beta[i] + betamax) / 2.
            else:
                betamax = beta[i].copy()
                if betamin == np.inf or betamin == -np.inf:
                    beta[i] = beta[i] / 2.
                else:
                    beta[i] = (beta[i] + betamin) / 2.

            # Recompute the values
            (H, thisP) = Hbeta(Di, beta[i])
            Hdiff = H - logU
            tries += 1

        # Set the final row of P
        P[i, np.concatenate((np.r_[0:i], np.r_[i+1:n]))] = thisP

    # Return final P-matrix
    print("Mean value of sigma: %f" % np.mean(np.sqrt(1 / beta)))
    return P


def pca(X=np.array([]), no_dims=50):
    """
        Runs PCA on the NxD array X in order to reduce its dimensionality to
        no_dims dimensions.
    """

    print("Preprocessing the data using PCA...")
    (n, d) = X.shape
    X = X - np.tile(np.mean(X, 0), (n, 1))
    (l, M) = np.linalg.eig(np.dot(X.T, X))
    Y = np.dot(X, M[:, 0:no_dims])
    return Y


def lda(X, label, no_dims=50):
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
    eigen_val, eigen_vec = eig(S)
    for i in range(eigen_vec.shape[1]):
        eigen_vec[:, i] = eigen_vec[:, i] / norm(eigen_vec[:, i])

    W = eigen_vec[:, :no_dims].real
    Y = X@W
    return Y


def sne(X, labels, no_dims=2, initial_dims=50, perplexity=30.0):
    """
        Runs t-SNE on the dataset in the NxD array X to reduce its
        dimensionality to no_dims dimensions. The syntaxis of the function is
        `Y = tsne.tsne(X, no_dims, perplexity), where X is an NxD NumPy array.
    """
    method = args.method
    dim_reduction = args.dim_reduction
    perplexity = args.perplexity

    # Check inputs
    if isinstance(no_dims, float):
        print("Error: array X should have type float.")
        return -1
    if round(no_dims) != no_dims:
        print("Error: number of dimensions should be an integer.")
        return -1

    # Initialize variables
    if dim_reduction == "pca":
        X = pca(X, initial_dims).real
    elif dim_reduction == "lda":
        X = lda(X, labels)
    (n, d) = X.shape
    max_iter = args.max_iter
    initial_momentum = 0.5
    final_momentum = 0.8
    eta = 500
    min_gain = 0.01
    Y = np.random.randn(n, no_dims)
    dY = np.zeros((n, no_dims))
    iY = np.zeros((n, no_dims))
    gains = np.ones((n, no_dims))

    # Compute P-values
    P = x2p(X, 1e-5, perplexity)
    P = P + np.transpose(P)
    P = P / np.sum(P)
    P = P * 4.									# early exaggeration
    P = np.maximum(P, 1e-12)

    C_list = []
    Y_list = []

    # Run iterations
    for iter in range(max_iter):

        # Compute pairwise affinities
        """sum_Y = np.sum(np.square(Y), 1)
        num = -2. * np.dot(Y, Y.T)
        num = 1. / (1. + np.add(np.add(num, sum_Y).T, sum_Y))"""
        # (1+||yi-yj||^2)^(-1)
        if method == "t-sne":
            num = 1/(1+cdist(Y, Y, metric="sqeuclidean"))
        # exp(-||yi-yj||^2)
        elif method == "s-sne":
            num = np.exp(-cdist(Y, Y, metric="sqeuclidean"))
        num[range(n), range(n)] = 0.
        Q = num / np.sum(num)
        Q = np.maximum(Q, 1e-12)

        # Compute gradient
        PQ = P - Q
        for i in range(n):
            if method == "t-sne":
                dY[i, :] = np.sum(np.tile(PQ[:, i] * num[:, i],
                                          (no_dims, 1)).T * (Y[i, :] - Y), 0)
            elif method == "s-sne":
                dY[i, :] = np.sum(
                    (np.tile(PQ[:, i], (no_dims, 1)).T) * (Y[i, :] - Y), axis=0)

        # Perform the update
        if iter < 20:
            momentum = initial_momentum
        else:
            momentum = final_momentum
        gains = (gains + 0.2) * ((dY > 0.) != (iY > 0.)) + \
                (gains * 0.8) * ((dY > 0.) == (iY > 0.))
        gains[gains < min_gain] = min_gain
        iY = momentum * iY - eta * (gains * dY)
        Y = Y + iY
        Y = Y - np.tile(np.mean(Y, 0), (n, 1))
        Y_list.append(Y)

        # Compute current value of cost function
        if (iter + 1) % 10 == 0:
            C = np.sum(P * np.log(P / Q))
            print("Iteration %d: error is %f" % (iter + 1, C))
            C_list.append(C)

        # Stop lying about P-values
        if iter == 100:
            P = P / 4.

    # Return solution
    return Y_list, P, Q, C_list


def plot_result(Y):
    method = args.method
    perplexity = args.perplexity
    dim_reduction = args.dim_reduction
    max_iter = args.max_iter

    # save plots
    save_dir = f"{SAVE_PATH}{method}_{perplexity}_{dim_reduction}/"
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)

    for idx, y in enumerate(Y):
        if (idx+1) % 10 == 0:
            plt.clf()
            plt.cla()
            plt.close()
            scatter = plt.scatter(y[:, 0], y[:, 1], 5, labels)
            plt.legend(*scatter.legend_elements(),
                       loc='lower left', title='Digit')
            plt.title(
                f'{method}, perplexity: {perplexity}, dim_reduction: {dim_reduction} iteration: {idx}')
            plt.tight_layout()
            plt.savefig(f'{save_dir}{idx}.png')

    # make gif
    vid_dir = SAVE_PATH+"video/"
    imgs = []
    for i in range(max_iter):
        if (i+1) % 10 == 0:
            image_name = f"{save_dir}{i}.png"
            imgs.append(Image.open(image_name))
    imgs[0].save(f"{vid_dir}{method}_{perplexity}_{dim_reduction}.gif", format='GIF', append_images=imgs[1:], loop=0,
                 save_all=True, duration=300)


def plot_similarity(P, Q):
    method = args.method
    perplexity = args.perplexity
    dim_reduction = args.dim_reduction

    # x = P or Q
    plt.clf()
    x = np.log(P)
    index = np.argsort(labels)
    x = x[index, :][:, index]
    img = plt.imshow(x, cmap='viridis', vmin=np.min(x), vmax=np.max(x))
    plt.colorbar(img)
    plt.savefig(f'{SAVE_PATH}{method}_{perplexity}_{dim_reduction}_High-D.png')

    plt.clf()
    x = np.log(Q)
    index = np.argsort(labels)
    x = x[index, :][:, index]
    img = plt.imshow(x, cmap='viridis', vmin=np.min(x), vmax=np.max(x))
    plt.colorbar(img)
    plt.savefig(f'{SAVE_PATH}{method}_{perplexity}_{dim_reduction}_Low-D.png')


if __name__ == "__main__":
    method = args.method
    max_iter = args.max_iter
    dim_reduction = args.dim_reduction

    print("Run Y = tsne.tsne(X, no_dims, perplexity) to perform t-SNE on your dataset.")
    print("Running example on 2,500 MNIST digits...")
    X = np.loadtxt(DATA_PATH+"mnist2500_X.txt")
    labels = np.loadtxt(DATA_PATH+"mnist2500_labels.txt")
    Y, P, Q, err_list = sne(X, labels, 2, 50, 20.0)
    plot_result(Y)
    iter_list = list(range(int(max_iter/10)))
    plot_similarity(P, Q)
    # pylab.scatter(Y[:, 0], Y[:, 1], 5, labels)
    # pylab.savefig(f"{SAVE_PATH}{method}_{dim_reduction}.jpg")
    # pylab.clf()
    # pylab.cla()
    # pylab.close()
    # pylab.plot(iter_list, err_list)
    # pylab.savefig(f"{SAVE_PATH}{method}_{dim_reduction}_error.jpg")
    # pylab.show()
