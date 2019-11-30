# -*- coding: utf-8 -*-

# 3rd party module
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.utils import np_utils
from keras.models import Sequential, load_model
from PIL import Image
import keras
import numpy as np
import sys


CLASSES = ['MORE', 'Ray', 'VERY']
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE_W = 50
IMAGE_SIZE_H = 100


def build_model():
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=(100, 50, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Conv2D(64, (3, 3), padding='same'))
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(3))
    model.add(Activation('softmax'))

    opt = keras.optimizers.rmsprop(lr=0.0001, decay=1e-6)

    model.compile(loss='categorical_crossentropy',
                  optimizer=opt, metrics=['accuracy'])

    # モデルの保存
    model = load_model('./visual_cnn.h5')

    return model


def main(img):
    # image = Image.open(sys.argv[1])
    image = Image.open(img)
    image = image.convert('RGB')
    image = image.resize((IMAGE_SIZE_W, IMAGE_SIZE_H))
    data = np.asarray(image)
    X = []
    X.append(data)
    X = np.array(X)
    # model = build_model()
    model = load_model('./visual_cnn.h5')

    result = model.predict([X])[0]
    predicted = result.argmax()
    percentage = int(result[predicted] * 100)
    print(f'{"="*25}')
    print('{0}({1}%)'.format(CLASSES[predicted], percentage))
    print(f'{"="*25}')


if __name__ == '__main__':
    # build_model()
    main('sample02.jpg')
