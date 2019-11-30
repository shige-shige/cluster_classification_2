# -*- coding: utf-8 -*-

# 3rd party module
from keras.models import Sequential
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import MaxPooling2D
from keras.utils import np_utils
import keras
import numpy as np


CLASSES = ['MORE', 'Ray', 'VERY']
NUM_CLASSES = len(CLASSES)
IMAGE_SIZE_W = 50
IMAGE_SIZE_H = 100


def main():
    # イメージデータを読み込む
    X_train, X_test, y_train, y_test = np.load('./visual.npy')

    # 最大値が1になるよう調整
    X_train = X_train.astype('float') / 256
    X_test = X_test.astype('float') / 256

    # カテゴリー情報を付与
    y_train = np_utils.to_categorical(y_train, NUM_CLASSES)
    y_test = np_utils.to_categorical(y_test, NUM_CLASSES)

    # トレーニング＆評価
    model = model_train(X_train, y_train)
    model_eval(model, X_test, y_test)


def model_train(X, y):
    """
    Args: X(train data), y(train data)
    Return: model
    """
    model = Sequential()
    model.add(Conv2D(32, (3, 3), padding='same', input_shape=X.shape[1:]))
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
    model.fit(X, y, batch_size=32, epochs=100)

    model.save('./visual_cnn.h5')

    return model


def model_eval(model, X, y):
    scores = model.evaluate(X, y, verbose=1)
    print('Test Loss: ', scores[0])
    print('Test Accuracy: ', scores[1])


if __name__ == '__main__':
    main()
