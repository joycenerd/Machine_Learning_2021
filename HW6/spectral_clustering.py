from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from numpy.linalg import eig
from PIL import Image
import numpy as np

import argparse
import glob
import time
import os


parser = argparse.ArgumentParser()
parser.add_argument("--kernel-type", type=str,
                    default="rbf", help="kernel function type")
parser.add_argument("--gamma-s", type=float, default=2.5,
                    help="hyperparameter gamma_s in the rbf kernel")
parser.add_argument("--gamma-c", type=float, default=2.5,
                    help="hyperparameter gamma_c in the rbf kernel")
parser.add_argument("--sigma", type=float, default=0.1,
                    help="Sigma value for Laplace rbf kernel")
parser.add_argument("--cut", type=str, default="normalized",
                    help="ratio or normalized cut")
parser.add_argument("--K", type=int, default=2, help="number of clusters")
parser.add_argument("--init-mode", type=str, default="k-means++",
                    help="initialize cluster mode")
parser.add_argument("--iterations", type=str, default=50,
                    help="Maximum iterations for K-means to run")
args = parser.parse_args()
print("".join(f"{k}={v}\n" for k, v in vars(args).items()))


DATA_PATH = "./data/"
SAVE_PATH = "./results/"


def get_kernel(img, h, w):
    img = img.reshape(h * w, 3)
    img = img / 255.0

    coor = []
    for i in range(w):
        for j in range(h):
            coor.append([i, j])
    coor = np.array(coor, dtype=float)
    coor = coor / 100.0

    if args.kernel_type == "rbf":
        pix_dist = cdist(img, img, "sqeuclidean")
        spatial_dist = cdist(coor, coor, "sqeuclidean")
        # e^-gamma_s*spatial_dist x e^-gamma_c*color_dist
        g_s = args.gamma_s
        g_c = args.gamma_c
        gram_matrix = np.multiply(
            np.exp(-g_s * spatial_dist), np.exp(-g_c * pix_dist))

    elif args.kernel_type == "Laplace_rbf":
        sigma = args.sigma
        pix_dist = cdist(img, img, metric="minkowski", p=1)
        spatial_dist = cdist(coor, coor, metric="minkowski", p=1)
        gram_matrix = np.multiply(
            np.exp(-1 / sigma * spatial_dist), np.exp(-1/sigma * pix_dist))
        print("finish gram matrix")

    return gram_matrix


def get_img_name(img_path):
    img_path = os.path.normpath(img_path)
    path_list = img_path.split(os.sep)
    img_name = path_list[-1][:-4]
    return img_name


def get_graph_Laplacian(W):
    cut_type = args.cut
    d = np.sum(W, axis=1)
    D = np.diag(d)  # degree matrix D=[dii]
    if cut_type == "ratio":
        L = D - W
    elif cut_type == "normalized":
        L = np.sqrt(D) @ (D - W) @ np.sqrt(D)
    return L


def eigen_decomposition(img_name, L):
    cut = args.cut
    K = args.K
    kernel_type = args.kernel_type

    eigval_f = DATA_PATH + f"eigval_{img_name}_{cut}_{kernel_type}.npy"
    eigvec_f = DATA_PATH + f"eigvec_{img_name}_{cut}_{kernel_type}.npy"
    if os.path.exists(eigval_f):
        eigval = np.load(eigval_f)
        eigvec = np.load(eigvec_f)
    else:
        eigval, eigvec = eig(L)
        np.save(eigval_f, eigval)
        np.save(eigvec_f, eigvec)

    order = np.argsort(eigval)
    sorted_eigvec = eigvec[:, order]
    U = sorted_eigvec[:, 1: K + 1]
    T = U.copy()
    if cut == "normalized":
        for i, u in enumerate(U):
            T[i, :] = u / np.sqrt(np.sum(u ** 2))
    return T


def init_cluster(data, img):
    K = args.K
    mode = args.init_mode

    if mode == "random":
        rand_idx = np.random.choice(data.shape[0], size=K)
        mean = data[rand_idx]
        dist = cdist(mean, data, metric="sqeuclidean")
        cluster = np.argmin(dist, axis=0)

    elif mode == "k-means++":
        # 1. Choose one center uniformly at random among the data points.
        # 2. For each data point x not chosen yet, compute D(x), the distance between x and the nearest center that has already been chosen.
        # 3. Choose one new data point at random as a new center, using a weighted probability distribution where a point x is chosen with probability proportional to D(x)2.
        # 4. Repeat Steps 2 and 3 until k centers have been chosen.
        img = img.reshape(-1, 3)
        img = img / 255.0
        first_mean = np.random.choice(h * w, size=1)
        center = np.full(K, first_mean, dtype=int)
        center_val = img[center]
        for i in range(1, K):
            dist = cdist(center_val, img, metric="sqeuclidean")
            min_dist = np.min(dist, axis=0)
            center[i] = np.random.choice(
                h * w, size=1, p=min_dist ** 2 / np.sum(min_dist ** 2))
            center_val = img[center]

        dist = cdist(center_val, img, metric="sqeuclidean")
        cluster = np.argmin(dist, axis=0)

    return cluster


def run(data, h, w, img):
    iterations = args.iterations
    K = args.K

    all_alpha = []
    alpha = init_cluster(data, img)
    all_alpha.append(alpha.reshape(h, w))

    for iter in range(iterations):
        cnt = np.zeros(K, dtype=float)
        for i in range(K):
            cnt[i] = np.count_nonzero(alpha == i)
            if cnt[i] == 0:
                cnt[i] = 1
        mean = np.zeros((K, K), dtype=float)
        for i in range(K):
            mean[i] = np.sum(data[alpha == i, :], axis=0)
            mean[i] = mean[i]/cnt[i]

        dist = cdist(mean, data, metric="sqeuclidean")
        new_alpha = np.argmin(dist, axis=0)
        all_alpha.append(new_alpha.reshape(h, w))

        if np.array_equal(alpha, new_alpha):
            print(f"Converge in {iter+1}th iterations!")
            break

        alpha = new_alpha

    all_alpha = np.array(all_alpha)

    return all_alpha


def plot_result(all_alpha, img_name, data):
    K = args.K
    mode = args.init_mode
    kernel_type = args.kernel_type
    cut = args.cut
    img_name += f"_{cut}_k{K}_{kernel_type}_{mode}"

    # export video .gif
    save_dir = SAVE_PATH + img_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    imgs = []
    for i in range(len(all_alpha)):
        out_img = color[all_alpha[i]]
        out_img = out_img.reshape((h, w, 3))
        plt.imsave(f"{save_dir}/{img_name}_{i}.png", out_img)
        imgs.append(Image.fromarray(np.uint8(out_img * 255)))
    video_path = SAVE_PATH + "spectral_video/" + img_name + ".gif"
    imgs[0].save(video_path, format="GIF", append_images=imgs[1:],
                 loop=0, save_all=True, duration=300)

    # plot eigenspace
    alpha = all_alpha[-1]
    alpha = np.array(alpha)
    alpha = alpha.reshape(-1)
    if K == 2:
        plt.figure(figsize=(10, 10))
        plt.scatter(data[alpha == 0, 0], data[alpha == 0, 1], c='yellow')
        plt.scatter(data[alpha == 1, 0], data[alpha == 1, 1], c='blue')
        plt.title(f"Eigendspace {cut} K={K} {kernel_type} {mode}")
        eigen_path = SAVE_PATH+"eigenspace/"+img_name+".png"
        plt.savefig(eigen_path)
        plt.show()


if __name__ == "__main__":
    start_time = time.time()

    for img_path in glob.glob(DATA_PATH + "*.png"):
        img_name = get_img_name(img_path)
        img = Image.open(img_path, "r")
        img = np.array(img)
        h, w, _ = img.shape
        W = get_kernel(img, h, w)
        L = get_graph_Laplacian(W)
        T = eigen_decomposition(img_name, L)
        #all_alpha = run(T, h, w, img)
        #plot_result(all_alpha, img_name, T)

    #print(f"--- {time.time()-start_time} seconds ---")
