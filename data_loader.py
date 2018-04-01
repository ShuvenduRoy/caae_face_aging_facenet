import glob
import cv2
import numpy as np


def load_data():
    all_images = glob.glob("E:\\Datasets\\UTKFace\\*")

    X = []
    y = []

    for image in all_images:
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])

        img = cv2.imread(image)
        X.append(img)
        y.append(int(age / 5))

    X = np.array(X)
    y = np.array(y)

    return X, y
