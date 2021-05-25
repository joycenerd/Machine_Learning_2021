from scipy.spatial.distance import cdist
from PIL import Image
import numpy as np

import argparse
import glob


parser = argparse.ArgumentParser()
parser.add_argument("--kernel_type", type=str,
                    default="rbf", help="kernel function type")
parser.add_argument("--gamma-s", type=float, default=2.5,
                    help="hyperparameter gamma_s in the rbf kernel")
parser.add_argument("--gamma-c", type=float, default=2.5,
                    help="hyperparameter gamma_c in the rbf kernel")
parser.add_argument("--kappa", type=float, default=0.25,
                    help="kappa value for hyperbolic tangent kernal")
parser.add_argument("--c", type=float, default=0.3,
                    help="constant value for hyperbolic tangent kernel")
args = parser.parse_args()


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
        gram_matrix = np.multiply(
            np.exp(-g_s * spatial_dist), np.exp(-g_c * pix_dist))

    elif args.kernel_type == "tanh":
        kappa = args.kappa
        c = args.c
        # tanh(kappa*xi*xj+c)
        pix_dist = np.tanh(kappa * img @ img.T + c)
        spatial_dist = np.tanh(kappa * coor @ coor.T + c)
        gram_matrix = np.multiply(pix_dist, spatial_dist)

    return gram_matrix


if __name__ == "__main__":
    for img_path in glob.glob(DATA_PATH+"*"):
        img = Image.open(img_path, "r")
        img = np.array(img)
        h, w, _ = img.shape
        W = get_kernel(img, h, w)
        d = np.sum(W, axis=1)
        D = np.diag(d)  # degree matrix D=[dii]
        L = D-W
