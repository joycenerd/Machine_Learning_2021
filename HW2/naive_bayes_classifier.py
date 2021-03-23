import numpy as np

def data_preprocessing(image_path,label_path):
    # extract image header
    image_file = open(image_path, "rb")
    magic_num = int.from_bytes(image_file.read(4), byteorder="big")
    num_of_data = int.from_bytes(image_file.read(4), byteorder="big")
    rows = int.from_bytes(image_file.read(4), byteorder="big")
    cols = int.from_bytes(image_file.read(4), byteorder="big")

    # extract label header
    label_file= open(label_path, "rb")
    magic_num = int.from_bytes(label_file.read(4), byteorder="big")
    num_of_data = int.from_bytes(label_file.read(4), byteorder="big")

    #extract image and label
    image = np.zeros((num_of_data, rows, cols), dtype=int)
    label = np.zeros((num_of_data), dtype=int)

    for i in range(num_of_data):
        for j in range(rows):
            for k in range(cols):
                image[i][j][k] = int.from_bytes(image_file.read(1), byteorder="big")
        label[i] = int.from_bytes(label_file.read(1), byteorder="big")

    return image,label,num_of_data,rows,cols


if __name__=="__main__":
    # Read in training set
    train_image,train_label,num_of_train,rows,cols=data_preprocessing("./data/train-images-idx3-ubyte","./data/train-labels-idx1-ubyte")
    print("num_of_train:",num_of_train)
    print("rows:",rows)
    print("cols:",cols)

    # Read in testing set
    test_image,test_label,num_of_test,rows,cols=data_preprocessing("./data/t10k-images-idx3-ubyte","./data/t10k-labels-idx1-ubyte")
    print("num_of_test:", num_of_test)
    print("rows:", rows)
    print("cols:", cols)

    toggle=input("Enter toggle option: ")

    # Discrete case
    if toggle=="0":
        # gray level [0-255], convert to 32 bins
        bin=np.zeros((num_of_train,rows,cols))
        for i in range(num_of_train):
            for j in range(rows):
                for k in range(cols):
                    bin[i][j][k]=int(train_image[i][j][k]/8)

        # Calculate likelihood

