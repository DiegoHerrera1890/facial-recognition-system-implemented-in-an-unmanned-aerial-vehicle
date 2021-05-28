from PIL import Image
import numpy as np
import os
from PIL import Image as im


# /home/diego/facial-recognition-system-implemented-in-an-unmanned-aerial-vehicle/face_recognition_siamese/pyimagesearch/dataset/training


def data_generation_1():
    # from keras.utils import plot_model
    # os.path.abspath("C:/example/cwd/mydir/myfile.txt")
    base_dir = os.path.abspath(
        r'/home/diego/facial-recognition-system-implemented-in-an-unmanned-aerial-vehicle/face_recognition_siamese/pyimagesearch/dataset/training/')
    train_test_split = 0.7
    no_of_files_in_each_class = 100

    # Read all the folders in the directory
    folder_list = os.listdir(base_dir)
    print(len(folder_list), "categories found in the training dataset")

    # Declare training array
    cat_list = []
    x = []
    y = []
    y_label = 0

    # Using just 5 images per category
    for folder_name in folder_list:
        files_list = os.listdir(os.path.join(base_dir, folder_name))
        temp = []
        for file_name in files_list[:no_of_files_in_each_class]:
            temp.append(len(x))
            x.append(
                np.asarray(
                    Image.open(os.path.join(base_dir, folder_name, file_name)).convert('RGB').resize((128, 128))))
            y.append(y_label)
        y_label += 1
        cat_list.append(temp)

    cat_list = np.asarray(cat_list)
    x_train = np.asarray(x) / 255.0
    y_train = np.asarray(y)
    # print('X, Y shape', x.shape, y.shape, cat_list.shape)
    print('X shape', x_train.shape)
    print('Y shape', y_train.shape)
    print('Cat_list shape', cat_list.shape)
    # print('X', x)

    # X_i, Y_i = make_pairs(x, y)
    # print('A', A)
    # print('B', B)
    # print('A', A.shape)
    # print('B', B.shape)
    # print('done')
    return x_train, y_train


def data_generation_2():
    # from keras.utils import plot_model
    base_dir = os.path.abspath(
        r'/home/diego/facial-recognition-system-implemented-in-an-unmanned-aerial-vehicle/face_recognition_siamese/pyimagesearch/dataset/testing/')
    train_test_split = 0.7
    no_of_files_in_each_class = 100

    # Read all the folders in the directory
    folder_list = os.listdir(base_dir)
    print(len(folder_list), "categories found in the testing dataset")

    # Declare training array
    cat_list = []
    x = []
    y = []
    y_label = 0

    # Using just 5 images per category
    for folder_name in folder_list:
        files_list = os.listdir(os.path.join(base_dir, folder_name))
        temp = []
        for file_name in files_list[:no_of_files_in_each_class]:
            temp.append(len(x))
            x.append(
                np.asarray(
                    Image.open(os.path.join(base_dir, folder_name, file_name)).convert('RGB').resize((128, 128))))
            y.append(y_label)
        y_label += 1
        cat_list.append(temp)

    cat_list = np.asarray(cat_list)
    x_test = np.asarray(x) / 255.0
    y_test = np.asarray(y)
    # print('X, Y shape', x.shape, y.shape, cat_list.shape)
    print('X_test shape', x_test.shape)
    print('Y_test shape', y_test.shape)
    print('Cat_list shape', cat_list.shape)
    # print('X', x)

    # X_i, Y_i = make_pairs(x, y)
    # print('A', A)
    # print('B', B)
    # print('A', A.shape)
    # print('B', B.shape)
    # print('done')
    return x_test, y_test
