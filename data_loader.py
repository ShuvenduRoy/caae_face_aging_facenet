import glob
import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import cv2


def load_data():
    print("load data started")
    all_images = glob.glob("E:\\Datasets\\UTKFace\\*")

    X = []
    y = []

    for image in all_images:
        only_name = image.split('\\')[-1]
        age = int(only_name.split('_')[0])

        img = mpimage.imread(image)
        img = cv2.resize(img, (64, 64))

        X.append(img)
        y.append(int(age / 5))

    X = np.array(X).astype(np.float32)
    y = np.array(y).astype(np.int)
    print("data loaded")

    return X, y
