# -*- coding: utf-8 -*-

# standard module
import glob
import os

# 3rd party module
from PIL import Image
from sklearn import model_selection
import numpy as np


CLASSES = ['MORE', 'Ray', 'VERY']
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE_W = 50
IMAGE_SIZE_H = 100


def image_data():
    # read images and save data as type of npy
    X = []
    Y = []
    for index, classlabel in enumerate(CLASSES):
        images_dir = '../downloads/' + classlabel
        files = glob.glob(images_dir + '/*.jpg')
        for i, file in enumerate(files):
            if i > 100:
                break
            else:
                image = Image.open(file)
                image = image.convert('RGB')
                image = image.resize((IMAGE_SIZE_W, IMAGE_SIZE_H))
                data = np.asarray(image)
                X.append(data)
                Y.append(index)
    X = np.array(X)
    Y = np.array(Y)
    X_train, X_test, Y_train, Y_test = model_selection.train_test_split(X, Y)
    xy = (X_train, X_test, Y_train, Y_test)
    np.save('./visual.npy', xy)


if __name__ == '__main__':
    image_data()
