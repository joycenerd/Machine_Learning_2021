from numpy.linalg import eigh
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


def extract_label(path):
    head, tail = ntpath.split(path)


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


def get_eig(data):
    # get eigenvalue and eigenvector by np.linalg.eigh()
    eigval_file = DATA_PATH+"eigval.npy"
    eigvec_file = DATA_PATH+"eigvec.npy"

    if os.path.isfile(eigval_file) and os.path.isfile(eigvec_file):
        eigval = np.load(eigval_file)
        eigvec = np.load(eigvec_file)

    else:
        eigval, eigvec = eigh(data)

    return eigval, eigvec


def pca(x):
    x_bar = np.mean(x, axis=0)
    cov = (x-x_bar)@(x-x_bar).T
    eigval, eigvec = get_eig(cov)
    # project data
    P = (x-x_bar).T@eigvec


if __name__ == "__main__":
    option = args.option

    # read training and testing data
    train_data, train_filepath, train_label = read_data(DATA_PATH+"Training/")
    test_data, test_filepath, test_label = read_data(DATA_PATH+"Testing/")
    data = np.vstack((train_data, test_data))  # (165,10000)
    filepath = np.hstack((train_filepath, test_filepath))  # (165,)
    label = np.hstack((train_label, test_label))  # (165,)

    if option == "PCA/LDA eigenface":
        pca(data)
