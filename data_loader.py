import glob
import numpy as np
import matplotlib.image as mpimage
import matplotlib.pyplot as plt
import cv2


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
