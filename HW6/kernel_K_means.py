from scipy.spatial.distance import cdist
from PIL import Image
import numpy as np
from glob import glob
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--clusters", type=int, default=2, help="Number of clusters")
parser.add_argument("--gamma-s", type=float, default=2.5, help="hyperparameter gamma_s in the kernel")
parser.add_argument("--gamma-c", type=float, default=2.5, help="hyperparameter gamma_c in the kernel")
parser.add_argument("--iterations", type=int, default=100, help="Maximum iteration for K-means")
args = parser.parse_args()


DATA_PATH = "./data/"


def get_kernel(img, h, w):
    img = img.reshape(h * w, c)

    # color similarity
    pix_dist = cdist(img, img, "sqeuclidean")

    # spatial similarity
    coor = []
    for i in range(w):
        for j in range(h):
            coor.append([i, j])
    coor = np.array(coor, dtype=float)
    spatial_dist = cdist(coor, coor, "sqeuclidean")

    # e^-gamma_s*spatial_dist x e^-gamma_c*color_dist
    g_s = args.gamma_s
    g_c = args.gamma_c
    gram_matrix = np.multiply(np.exp(-g_s * spatial_dist), np.exp(-g_c * pix_dist))

    return gram_matrix


def init_cluster(h, w):
    cluster = np.random.randint(args.clusters, size=h * w)
    return cluster


def run(h, w, gram_matrix):
    K = args.clusters
    all_cluster = []

    # initialize the clusters
    alpha = init_cluster(h, w)
    all_cluster.append(alpha.reshape(h, w))

    # Kernel K-means
    for iter in range(args.iterations):
        first_term = np.diag(gram_matrix).reshape(-1, 1)

        # 2/|C_k|*sum_n(alpha_kn*k(xj,xn))
        C = np.zeros(K, dtype=float)
        for i in range(K):
            C[i] = np.count_nonzero(alpha == i)
        second_term = np.zeros((h * w, K), dtype=float)
        for j in range(h * w):
            for k in range(K):
                for n in range(h * w):
                    if alpha[n] == k:
                        second_term[j, k] += gram_matrix[j, n]
            second_term[j, k] *= 2 / C[k]

        # 1/|C_k|^2 alpha_kp*alpha_kq*k(xp,xq)
        third_term = np.zeros(K, dtype=float)
        for k in range(K):
            for p in range(h * w):
                for q in range(h * w):
                    if alpha[i] == k and alpha[j] == k:
                        third_term[k] += gram_matrix[p, q]
            third_term[k] /= C[k]

        new_alpha = np.argmin(first_term - second_term + third_term, axis=1)

        if np.equal(alpha, new_alpha):
            break
        alpha = new_alpha


if __name__ == "__main__":
    for img_path in glob(DATA_PATH + "*"):
        img = Image.open(img_path)
        img = np.array(img)
        h, w, c = img.shape
        gram_matrix = get_kernel(img, h, w)
        run(h, w, gram_matrix)
        break
