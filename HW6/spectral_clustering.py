from scipy.spatial.distance import cdist
from numpy.linalg import eig
from PIL import Image
import numpy as np

import argparse
import glob
import os


parser = argparse.ArgumentParser()
parser.add_argument("--kernel-type", type=str, default="tanh", help="kernel function type")
parser.add_argument("--gamma-s", type=float, default=2.5, help="hyperparameter gamma_s in the rbf kernel")
parser.add_argument("--gamma-c", type=float, default=2.5, help="hyperparameter gamma_c in the rbf kernel")
parser.add_argument("--kappa", type=float, default=0.25, help="kappa value for hyperbolic tangent kernal")
parser.add_argument("--c", type=float, default=0.3, help="constant value for hyperbolic tangent kernel")
parser.add_argument("--cut", type=str, default="normalized", help="ratio or normalize cut")
parser.add_argument("--K", type=int, default=2, help="number of clusters")
args = parser.parse_args()
print("".join(f"{k}={v}\n" for k, v in vars(args).items()))


DATA_PATH = "./data/"


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
        gram_matrix = np.multiply(np.exp(-g_s * spatial_dist), np.exp(-g_c * pix_dist))

    elif args.kernel_type == "tanh":
        kappa = args.kappa
        c = args.c
        # tanh(kappa*xi*xj+c)
        pix_dist = np.tanh(kappa * img @ img.T + c)
        spatial_dist = np.tanh(kappa * coor @ coor.T + c)
        gram_matrix = np.multiply(pix_dist, spatial_dist)

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
    U = sorted_eigvec[:, 1 : K + 1]
    T = U.copy()
    if cut == "normalized":
        for i, u in enumerate(U):
            T[i, :] = u / np.sqrt(np.sum(u ** 2))
    return T


def K_means(data):
    print(data.shape)


if __name__ == "__main__":
    for img_path in glob.glob(DATA_PATH + "*.png"):
        img_name = get_img_name(img_path)
        img = Image.open(img_path, "r")
        img = np.array(img)
        h, w, _ = img.shape
        W = get_kernel(img, h, w)
        L = get_graph_Laplacian(W)
        T = eigen_decomposition(img_name, L)
        K_means(T)
