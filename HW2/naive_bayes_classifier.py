import numpy as np
import math


def data_preprocessing(image_path, label_path):
    """
    data_preprocessing: extract MNIST data and label from binary big endian file
    :param image_path: the path of image data
    :param label_path: the path of label data
    :return: MNIST images and labels, total number of data, each image row and column counts
    """
    # extract image header
    image_file = open(image_path, "rb")
    magic_num = int.from_bytes(image_file.read(4), byteorder="big")
    num_of_data = int.from_bytes(image_file.read(4), byteorder="big")
    rows = int.from_bytes(image_file.read(4), byteorder="big")
    cols = int.from_bytes(image_file.read(4), byteorder="big")

    # extract label header
    label_file = open(label_path, "rb")
    magic_num = int.from_bytes(label_file.read(4), byteorder="big")
    num_of_data = int.from_bytes(label_file.read(4), byteorder="big")

    # extract image and label
    image = np.zeros((num_of_data, rows, cols), dtype=int)
    label = np.zeros(num_of_data, dtype=int)

    for i in range(num_of_data):
        for j in range(rows):
            for k in range(cols):
                image[i][j][k] = int.from_bytes(image_file.read(1), byteorder="big")
        label[i] = int.from_bytes(label_file.read(1), byteorder="big")

    return image, label, num_of_data, rows, cols


def tally_freq(num_of_data, rows, cols, data):
    """
    tally_freq: tally [0,255] into 32 discrete bins
    :param num_of_data: total data counts
    :param rows: the number of rows of each image
    :param cols: the numbr of columns of each image
    :param data: the original imag data
    :return: 32 bins for each image
    """
    bin = np.zeros((num_of_data, rows, cols), dtype=int)
    for i in range(num_of_data):
        for j in range(rows):
            for k in range(cols):
                bin[i][j][k] = int(data[i][j][k] / 8)
    return bin


def get_prior(data_label, num_of_data):
    """
    get_prior: the the prior (each_class/num_of_data)
    :param data_label: ground truth label
    :param num_of_data: number of training or testing data
    :return: the total number of data in each class and prior of each class
    """
    class_cnt = np.zeros(10, dtype=np.double)
    prior = np.zeros(10, dtype=np.double)
    for label in data_label:
        class_cnt[label] += 1
    for i, classes in enumerate(class_cnt):
        prior[i] = classes / num_of_data
    return class_cnt, prior


def print_posterior_calc_error(test_label, all_posterior):
    """
    print_posterior_calc_error: print the calculated posterior, the prediction and ground truth label and calculate the error rate (predction!=label)
    :param test_label: ground truth label
    :param all_posterior: posterior of each image and all the possible class
    :return: error rate
    """
    err = 0.0
    for i, posterior in enumerate(all_posterior):
        print("Posterior (in log scale):")
        for classes, class_prob in enumerate(posterior):
            print('{}: {}'.format(classes, class_prob))
        pred = np.argmin(posterior)
        label = test_label[i]
        print("Prediction: {}, Ans: {}".format(pred, label))
        if pred != label:
            err += 1
        print("")
    return err


def print_err(err, num_of_data):
    """
    print_err: print error rate
    :param err: error rate
    :param num_of_data: number of training or testing data
    :return: no return value
    """
    err_rate = err / num_of_data
    print("Error rate: " + str(err_rate))


def get_posterior(mode,test_image,prior,likelihood=None,mean=None,variance=None):
    """
    get_posterior: calculate the posterior of every testing image
    :param mode: discrete:0 or continuous:1
    :param test_image: discrete: each image has 32 bins, continuous: all the original testing image
    :param prior: the prior
    :param likelihood: the likelihood in discrete mode
    :param mean: mean of each pixel in each class (continuos mode)
    :param variance: variance of each pixel in each class (continuous mode)
    :return: all the posterior for every image every class (each image will have 10 possible class)
    """
    posterior_list = []

    for n, px_list in enumerate(test_image):
        posterior = np.zeros(10, dtype=np.double)
        for c in range(10):
            posterior[c] += math.log(max(1e-4,prior[c]))
            for px_idx, px in enumerate(px_list):
                if mode==0:
                    posterior[c]+=math.log(max(1e-4, likelihood[c][px_idx][int(px)]))
                elif mode==1:
                    if variance[c][px_idx] == 0.0:  # prevent from divide by 0
                        variance[c][px_idx] = 1e-8
                    g = math.log(1.0 / math.sqrt(2.0 * math.pi * variance[c][px_idx])) - (
                            (px - mean[c][px_idx]) ** 2 / (2.0 * variance[c][px_idx]))
                    posterior[c] += g
        posterior = posterior / np.sum(posterior)  # marginalize
        posterior_list.append(posterior)
    return posterior_list


if __name__ == "__main__":
    # Read in training set
    train_image, train_label, num_of_train, rows, cols = data_preprocessing("./data/train-images-idx3-ubyte",
                                                                            "./data/train-labels-idx1-ubyte")
    print("num_of_train:", num_of_train)
    print("rows:", rows)
    print("cols:", cols)

    # Read in testing set
    test_image, test_label, num_of_test, rows, cols = data_preprocessing("./data/t10k-images-idx3-ubyte",
                                                                         "./data/t10k-labels-idx1-ubyte")
    print("num_of_test:", num_of_test)
    print("rows:", rows)
    print("cols:", cols)

    # calculate prior
    class_cnt, prior = get_prior(train_label, num_of_train)
    
    while True:
        toggle = input("Enter toggle option: ")

        # Discrete mode
        if toggle == "0":

            # gray level [0-255], convert to 32 bin
            train_bin = tally_freq(num_of_train, rows, cols, train_image)

            # Calculate likelihood
            train_bin = train_bin.reshape(num_of_train, -1)  # [60000,784]
            likelihood = np.zeros((10, rows * cols, 32), dtype=np.double)  # [10,784,32]
            likelihood_sum = np.zeros((10, rows * cols))  # [10,784]

            bin_total = np.zeros([10, rows * cols, 32], dtype=float)
            for img_idx, label in enumerate(train_label):
                for px_idx, value in enumerate(train_bin[img_idx]):
                    bin_total[int(label)][px_idx][int(value)] += 1

            likelihood = np.zeros([10, rows * cols, 32])
            for label in range(10):
                for px_idx in range(rows * cols):
                    for bin in range(32):
                        if prior[label] != 0:
                            likelihood[label][px_idx][bin] = float(bin_total[label][px_idx][bin] / class_cnt[label])
                        else:
                            likelihood[cls][px_idx][bin] = 1e-8

            # Calculate posterior
            test_bin = tally_freq(num_of_test, rows, cols, test_image)
            test_bin = test_bin.reshape(num_of_test, -1)
            posterior_list=get_posterior(0,test_bin,prior,likelihood)

            # Print posterior
            err = print_posterior_calc_error(test_label, posterior_list)

            # print imagination number --> from likelihood
            likelihood = likelihood.reshape(10, rows, cols, -1)
            for i in range(10):
                print(str(i) + ':')
                for j in range(rows):
                    for k in range(cols):
                        zero_prob = 0
                        one_prob = 0
                        for l in range(16):
                            zero_prob += likelihood[i][j][k][l]
                            one_prob += likelihood[i][j][k][l + 16]
                        if zero_prob > one_prob:
                            print('0', end=' ')
                        else:
                            print('1', end=' ')
                    print('')
                print('')

            # print error rate
            print_err(err, num_of_test)

        # Continuous mode
        if toggle == "1":
            # Calculate prior
            class_cnt, prior = get_prior(train_label, num_of_train)

            # Calculate likelihood
            train_image = train_image.reshape((num_of_train, -1))  # [60000,784]
            mean = np.zeros((10, rows * cols), dtype=np.double)  # [10,784]
            variance = np.zeros((10, rows * cols), dtype=np.double)  # [10,784]
            sum_of_train = np.zeros((10, rows * cols), dtype=np.double)  # [10,784]

            # get mean, variance
            for i in range(num_of_train):
                for j in range(rows * cols):
                    label = train_label[i]
                    sum_of_train[label][j] += train_image[i][j]
                    variance[label][j] += train_image[i][j] ** 2

            for i in range(10):
                for j in range(rows * cols):
                    mean[i][j] = sum_of_train[i][j] / class_cnt[i]
                    variance[i][j] = variance[i][j] / class_cnt[i] - (mean[i][j] ** 2) + 1000  # var=sum(x)^2/n-mean^2

            # Calculate posterior
            test_image = test_image.reshape(num_of_test, -1)  # [10000,784]
            posterior_list=get_posterior(1,test_image,prior,None,mean,variance)

            # print posterior and calculate error rate
            err = print_posterior_calc_error(test_label, posterior_list)

            # print imagination number
            mean = mean.reshape((10, rows, cols))
            for i in range(10):
                print(str(i) + ":")
                for j in range(rows):
                    for k in range(cols):
                        if mean[i][j][k] < 128:
                            print("0", end=" ")
                        else:
                            print("1", end=" ")
                    print("")
                print("")

            # print error rate
            print_err(err, num_of_test)
            print("")
