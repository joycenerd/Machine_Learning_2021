from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from glob import glob
import argparse
import os


parser = argparse.ArgumentParser()
parser.add_argument("--clusters", type=int, default=2,
                    help="Number of clusters")
parser.add_argument("--gamma-s", type=float, default=2.5,
                    help="hyperparameter gamma_s in the kernel")
parser.add_argument("--gamma-c", type=float, default=2.5,
                    help="hyperparameter gamma_c in the kernel")
parser.add_argument("--iterations", type=int, default=50,
                    help="Maximum iteration for K-means")
args = parser.parse_args()


DATA_PATH = "./data/"
SAVE_PATH = "./results/"


def get_kernel(img, h, w):
    img = img.reshape(h * w, c)
    img=img/255.0

    # color similarity
    pix_dist = cdist(img, img, "sqeuclidean")

    # spatial similarity
    coor = []
    for i in range(w):
        for j in range(h):
            coor.append([i, j])
    coor = np.array(coor, dtype=float)
    coor=coor/100.0
    spatial_dist = cdist(coor, coor, "sqeuclidean")

    # e^-gamma_s*spatial_dist x e^-gamma_c*color_dist
    g_s = args.gamma_s
    g_c = args.gamma_c
    gram_matrix = np.multiply(
        np.exp(-g_s * spatial_dist), np.exp(-g_c * pix_dist))

    return gram_matrix


def init_cluster(h, w):
    cluster = np.random.randint(args.clusters, size=h * w)
    return cluster


def run(h, w, gram_matrix):
    K = args.clusters
    all_alpha = []

    # initialize the clusters
    alpha = init_cluster(h, w)
    all_alpha.append(alpha.reshape(h, w))

    # Kernel K-means
    for iter in range(1, args.iterations+1):
        first_term = np.diag(gram_matrix).reshape(-1, 1)

        # 2/|C_k|*sum_n(alpha_kn*k(xj,xn))
        C = np.zeros(K, dtype=float)
        for i in range(K):
            C[i] = np.count_nonzero(alpha == i)
            if C[i] == 0:
                C[i] = 1
        second_term = np.zeros((h*w, K), dtype=float)
        for k in range(K):
            second_term[:, k] = np.sum(gram_matrix[:, alpha == k], axis=1)
        second_term *= (-2.0 / C)

        # 1/|C_k|^2 alpha_kp*alpha_kq*k(xp,xq)
        third_term = np.zeros(K, dtype=float)
        for k in range(K):
            third_term[k] = np.sum(gram_matrix[alpha == k, :][:, alpha == k])
        third_term = third_term / (C ** 2)

        new_alpha = np.argmin(first_term + second_term + third_term, axis=1)
        all_alpha.append(new_alpha)

        if np.array_equal(alpha, new_alpha):
            print(f"Converge in {iter}th iterations!")
            break
        alpha = new_alpha

        #print(f"Iteration #{iter} complete...")

    return all_alpha


def plot_result(all_alpha, img_name):
    save_dir = SAVE_PATH+img_name
    if not os.path.isdir(save_dir):
        os.mkdir(save_dir)
    color = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0]], dtype=float)
    imgs = []
    for i in range(len(all_alpha)):
        out_img = color[all_alpha[i]]
        out_img = out_img.reshape((h, w, 3))
        plt.imsave(f"{save_dir}/{img_name}_{i}.png", out_img)
        imgs.append(Image.fromarray(np.uint8(out_img * 255)))
    imgs[0].save(f'{save_dir}/video.gif', format='GIF',
                 append_images=imgs[1:], loop=0, save_all=True, duration=300)


if __name__ == "__main__":
    for img_path in glob(DATA_PATH + "*"):
        print(f"Start processing {img_path}")
        img = Image.open(img_path)
        img_path = os.path.normpath(img_path)
        path_list = img_path.split(os.sep)
        img_name = path_list[-1][:-4]
        img = np.array(img)
        h, w, c = img.shape
        gram_matrix = get_kernel(img, h, w)
        print("Get gram matrix complete...")
        all_alpha = run(h, w, gram_matrix)
        img_name += f'_{args.clusters}'
        plot_result(all_alpha, img_name)
        print("Plotting complete...")
