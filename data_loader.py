import glob
import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import cv2

size = (96, 96)


def UTKFace_data(size=(128, 128)):
    print("load data started")
    all_images = glob.glob("E:\\Datasets\\UTKFace\\*")

    X = []
    y = []

    for i, image in enumerate(all_images):
        # # take half data
        # if i%20 != 0:
        #     continue

        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])

        img = mpimage.imread(image)
        img = cv2.resize(img, (size[0], size[1]))

        X.append(img)
        y.append(min(int(age / 5), 19))

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int)
    print("data loaded")

    return X, y


def UTKFace_male(size=(128, 128)):
    print("load data started")
    all_images = glob.glob("E:\\Datasets\\UTKFace\\*")

    X = []
    y = []

    for i, image in enumerate(all_images):
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])
        gender = int(only_name.split('_')[1])

        if gender == 0:
            img = mpimage.imread(image)
            img = cv2.resize(img, (size[0], size[1]))

            X.append(img)
            y.append(min(int(age / 5), 19))

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int)
    print("data loaded")

    return X, y


def UTKFace_female(size=(128, 128)):
    print("load data started")
    all_images = glob.glob("E:\\Datasets\\UTKFace\\*")

    X = []
    y = []

    for i, image in enumerate(all_images):
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])
        gender = int(only_name.split('_')[1])

        if gender == 1:
            img = mpimage.imread(image)
            img = cv2.resize(img, (size[0], size[1]))

            X.append(img)
            y.append(min(int(age / 5), 19))

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int)
    print("data loaded")

    return X, y


def UTKFace_male_5cat(size=(128, 128)):
    print("load data started")
    all_images = glob.glob("E:\\Datasets\\UTKFace\\*")

    X = []
    y = []

    for i, image in enumerate(all_images):
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])
        gender = int(only_name.split('_')[1])

        if gender == 0:
            img = mpimage.imread(image)
            img = cv2.resize(img, (size[0], size[1]))

            X.append(img)

            age_class = 0
            if age > 18:
                age_class = 1
            if age > 30:
                age_class = 2
            if age > 50:
                age_class = 3
            if age > 65:
                age_class = 4

            y.append(age_class)

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int)
    print("data loaded")

    return X, y
